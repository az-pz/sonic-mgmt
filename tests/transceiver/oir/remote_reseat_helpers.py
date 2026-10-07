"""Remote reseat operation and verification primitives.

Backs the Remote reseat test cases in
``docs/testplan/transceiver/online_insertion_removal_testplan.md``.

A remote reseat simulates an OIR on-device in seven steps, each with its own
expected result: disable DOM monitoring, shut the ports down, reset the module,
put it in and take it out of low-power mode, start the ports up again and
re-enable DOM monitoring.  :func:`perform_remote_reseat` runs every step across
all modules under test before the next step (one bulk cycle, like the other
scenario operations) and returns the per-step failures.  The verifiers below
complement the OIR primitives in :mod:`tests.transceiver.oir.oir_helpers`:
periodic tables updated after the reseat, LLDP neighbor identity, and the link
peers' state.  They return failure strings for the caller to aggregate into one
``pytest.fail``.
"""
import logging
import re
import time
from collections import defaultdict, namedtuple

from natsort import natsorted

from tests.common.platform.interface_utils import get_dut_interfaces_status, wait_ports_oper_status
from tests.transceiver.attribute_parser.attribute_keys import (
    REMOTE_RESEAT_ATTRIBUTES_KEY,
    SYSTEM_ATTRIBUTES_KEY,
)
from tests.transceiver.common import cli_helpers, db_helpers, scenario_ops
from tests.transceiver.common.verification import check_lldp_neighbors_present
from tests.transceiver.dom.dom_helpers import DOM_POLLING_DISABLED_VALUE, DOM_POLLING_ENABLED_VALUES
from tests.transceiver.eeprom import recovery
from tests.transceiver.oir import oir_helpers

logger = logging.getLogger(__name__)

# The remote reseat procedure's pause after resetting the modules.
RESET_SETTLE_SEC = 5
# xcvrd's default DomInfoUpdateTask period between two DOM sweeps.
XCVRD_DOM_UPDATE_PERIOD_SEC = 60
DEFAULT_LLDP_NEIGHBOR_WAIT_SEC = 60
REFRESH_POLL_INTERVAL_SEC = 10
LLDP_POLL_INTERVAL_SEC = 3

_STEPS = {
    1: "disable DOM monitoring",
    2: "port shutdown",
    3: "module reset",
    4: "low-power mode on",
    5: "low-power mode off",
    6: "port startup",
    7: "enable DOM monitoring",
}

# Tables xcvrd republishes, with a new ``last_update_time``, on every DOM sweep
# of a module whose DOM monitoring is enabled.  Threshold tables carry one too
# but are only published when a module is inserted, so they are not listed.
PERIODIC_TABLES = (
    "TRANSCEIVER_DOM_SENSOR",
    "TRANSCEIVER_DOM_FLAG",
    "TRANSCEIVER_STATUS",
    "TRANSCEIVER_STATUS_FLAG",
    "TRANSCEIVER_VDM_REAL_VALUE",
    "TRANSCEIVER_VDM_HALARM_FLAG",
    "TRANSCEIVER_VDM_LALARM_FLAG",
    "TRANSCEIVER_VDM_HWARN_FLAG",
    "TRANSCEIVER_VDM_LWARN_FLAG",
)

_RX_SIGNAL_FLAG_RE = re.compile(r"^rx\d+(los|cdrlol)$")
# xcvrd publishes "N/A" for a flag the module does not support.
_SUPPORTED_FLAG_VALUES = ("true", "false")

ReseatWaits = namedtuple("ReseatWaits", ("shutdown", "startup", "reset_settle", "lpm_settle", "publication"))
PeerBaseline = namedtuple("PeerBaseline", ("flap_counts", "tables", "rx_signal_flag_counts"))


def modules_of(lports, lport_to_first_subport_mapping):
    """Return the first sub-ports of the modules ``lports`` belong to."""
    return natsorted({lport_to_first_subport_mapping.get(port, port) for port in lports})


def _group_by_namespace(host, ports):
    ports_by_namespace = defaultdict(list)
    for port in ports:
        ports_by_namespace[db_helpers.resolve_port_namespace(host, port)].append(port)
    return ports_by_namespace


def reseat_waits(port_attributes_dict, lports, modules):
    """Return the settle budgets of a bulk remote reseat of ``lports``.

    The slowest port sets each budget, and the port settle waits are scaled to
    the bulk operation.  ``transceiver_reset_i2c_recover_sec`` is the I2C
    recovery time after a reset or low-power mode change; the reset settle
    never drops below the procedure's 5s.  ``publication`` adds the DOM
    republish budget (``dom_info_recover_sec``, at least one xcvrd DOM sweep)
    to the oper-up wait.
    """
    system_attrs = [port_attributes_dict[port][SYSTEM_ATTRIBUTES_KEY] for port in lports]
    startup = scenario_ops.scale_bulk_wait(
        max(attrs["port_startup_wait_sec"] for attrs in system_attrs), len(lports))
    i2c_recover = max(
        port_attributes_dict[module][SYSTEM_ATTRIBUTES_KEY]["transceiver_reset_i2c_recover_sec"]
        for module in modules)
    return ReseatWaits(
        shutdown=scenario_ops.scale_bulk_wait(
            max(attrs["port_shutdown_wait_sec"] for attrs in system_attrs), len(lports)),
        startup=startup,
        reset_settle=max(RESET_SETTLE_SEC, i2c_recover),
        lpm_settle=i2c_recover,
        publication=max(
            recovery.dom_republish_wait(port_attributes_dict, startup, lports),
            startup + XCVRD_DOM_UPDATE_PERIOD_SEC),
    )


def reseat_timeout_sec(port_attributes_dict, lports):
    """Return the slowest port's ``remote_reseat_timeout_min``, scaled to the bulk reseat."""
    timeout_min = max(
        port_attributes_dict[port][REMOTE_RESEAT_ATTRIBUTES_KEY]["remote_reseat_timeout_min"] for port in lports)
    return scenario_ops.scale_bulk_wait(timeout_min * 60, len(lports))


def lldp_port_timeouts(port_attributes_dict, lports):
    """Return ``{port: lldp_neighbor_wait_sec}`` for the ports with ``verify_lldp_on_link_up``."""
    timeouts = {}
    for port in lports:
        system_attrs = port_attributes_dict[port].get(SYSTEM_ATTRIBUTES_KEY, {})
        if system_attrs.get("verify_lldp_on_link_up", True):
            timeouts[port] = system_attrs.get("lldp_neighbor_wait_sec", DEFAULT_LLDP_NEIGHBOR_WAIT_SEC)
    return timeouts


# ──────────────────────────────────────────────────────────────────────
# DOM monitoring
# ──────────────────────────────────────────────────────────────────────


def read_dom_polling(duthost, modules):
    """Return ``({module: CONFIG_DB dom_polling, '' when unset}, errors)``."""
    values, errors = {}, []
    for module in modules:
        value, err = db_helpers.get_db_hash_field(
            duthost, "CONFIG_DB", "PORT", module, "dom_polling",
            namespace=db_helpers.resolve_port_namespace(duthost, module))
        if err:
            errors.append(f"{module}: {err}")
        else:
            values[module] = (value or "").strip().lower()
    return values, errors


def set_dom_monitoring(duthost, modules, enable):
    """Enable or disable DOM monitoring on ``modules`` and verify CONFIG_DB took it."""
    failures = []
    for module in modules:
        err = cli_helpers.set_dom_polling(
            duthost, module, enable, namespace=db_helpers.resolve_port_namespace(duthost, module))
        if err:
            failures.append(err)
    values, errors = read_dom_polling(duthost, modules)
    expected = DOM_POLLING_ENABLED_VALUES if enable else (DOM_POLLING_DISABLED_VALUE,)
    failures += errors
    failures += [
        f"{module}: dom_polling is {value!r}, expected DOM monitoring {'enabled' if enable else 'disabled'}"
        for module, value in values.items() if value not in expected
    ]
    return failures


def _clear_dom_polling(duthost, module):
    """Remove ``module``'s ``dom_polling`` field; no CLI unsets it."""
    namespace = db_helpers.resolve_port_namespace(duthost, module)
    ns_flag = f" -n {namespace}" if namespace else ""
    cmd = f'sonic-db-cli{ns_flag} CONFIG_DB hdel "PORT|{module}" dom_polling'
    result = duthost.shell(cmd, module_ignore_errors=True)
    if result.get("rc", 1) != 0:
        return f"{cmd} failed with rc={result.get('rc')}: {(result.get('stderr') or '').strip()}"
    return None


def restore_dom_polling(duthost, original):
    """Return each module's ``dom_polling`` to its exact ``original`` (:func:`read_dom_polling`) value."""
    current, failures = read_dom_polling(duthost, list(original))
    changed = [module for module, value in original.items() if module in current and current[module] != value]
    for module in changed:
        value = original[module]
        logger.warning("Restoring dom_polling of %s from %r to %r", module, current[module], value or "<unset>")
        if value:
            err = cli_helpers.set_dom_polling(
                duthost, module, value != DOM_POLLING_DISABLED_VALUE,
                namespace=db_helpers.resolve_port_namespace(duthost, module))
        else:
            err = _clear_dom_polling(duthost, module)
        if err:
            failures.append(err)
    restored, errors = read_dom_polling(duthost, changed)
    failures += errors
    failures += [
        f"{module}: dom_polling is {value!r} after restoration, expected {original[module] or '<unset>'!r}"
        for module, value in restored.items() if value != original[module]
    ]
    return failures


# ──────────────────────────────────────────────────────────────────────
# Periodic table updates
# ──────────────────────────────────────────────────────────────────────


def capture_update_times(host, ports, tables=PERIODIC_TABLES):
    """Return ``({(table, port): last_update_time}, errors)`` for the ``tables`` entries of ``ports``."""
    update_times, errors = {}, []
    for namespace, namespace_ports in _group_by_namespace(host, ports).items():
        for table in tables:
            entries, err = db_helpers.get_state_db_table(host, table, namespace=namespace)
            if err:
                errors.append(err)
                continue
            for port in namespace_ports:
                update_time = (entries.get(port) or {}).get("last_update_time")
                if update_time:
                    update_times[(table, port)] = update_time
    return update_times, errors


def verify_tables_refreshed(host, baseline, wait_sec):
    """Poll until every ``baseline`` (:func:`capture_update_times`) entry is republished."""
    if not baseline:
        return []
    ports = natsorted({port for _, port in baseline})
    tables = sorted({table for table, _ in baseline})

    def _check():
        current, failures = capture_update_times(host, ports, tables)
        for (table, port), before in sorted(baseline.items()):
            after = current.get((table, port))
            if after is None:
                failures.append(f"{port}: {table} entry or its last_update_time disappeared")
            elif after == before:
                failures.append(
                    f"{port}: {table} not republished after DOM monitoring was re-enabled "
                    f"(last_update_time still {before})")
        return failures

    return scenario_ops.poll_ports_recovered(_check, wait_sec, REFRESH_POLL_INTERVAL_SEC, "periodic table update")


# ──────────────────────────────────────────────────────────────────────
# Remote reseat operation
# ──────────────────────────────────────────────────────────────────────


def _step_failures(step, failures):
    return [f"step {step} ({_STEPS[step]}): {failure}" for failure in failures]


def _check_lldp_present(duthost, lldp_timeouts):
    """Step 6: every port that linked up is seen in the LLDP table."""
    if not lldp_timeouts:
        return []
    up_ports = set(oir_helpers.get_oper_up_ports(duthost, list(lldp_timeouts)))
    results = check_lldp_neighbors_present(
        duthost, {port: timeout for port, timeout in lldp_timeouts.items() if port in up_ports})
    return [result["details"] for result in results.values() if not result["passed"]]


def _set_lpmode(duthost, modules, low_power, settle_sec):
    """Steps 4 and 5: move ``modules`` into or out of low-power mode, let their
    I2C recover, then confirm each module reports the new mode."""
    failures, switched = [], []
    for module in modules:
        elapsed, err = cli_helpers.sfputil_set_lpmode(duthost, module, low_power)
        logger.info("Port %s: lpmode %s took %ss", module, "on" if low_power else "off", elapsed)
        if err:
            failures.append(err)
        else:
            switched.append(module)
    if switched:
        time.sleep(settle_sec)
    for module in switched:
        failures += scenario_ops.verify_lpmode(duthost, module, low_power)
    return failures


def perform_remote_reseat(duthost, port_attributes_dict, lports, modules, waits, lldp_timeouts, timeout_sec):
    """Remotely reseat ``modules`` (every port of which is in ``lports``).

    Runs the seven steps of the remote reseat procedure, each across every
    module before the next, and checks each step's expected result.  Steps 6
    and 7 run however the earlier steps end, so a port is never left down or
    unmonitored.  The whole reseat must complete within ``timeout_sec``.

    Returns:
        tuple: ``(failures, update_times)`` - the per-step failure strings, and
        the :func:`capture_update_times` snapshot of the modules taken just
        before DOM monitoring is re-enabled, from which a later update of the
        periodic tables can only come from the re-enabled monitoring.
    """
    started_at = time.monotonic()
    logger.info("Remote reseat of %d module(s) across %d port(s)", len(modules), len(lports))
    failures = _step_failures(1, set_dom_monitoring(duthost, modules, enable=False))
    try:
        failures += _step_failures(2, scenario_ops.perform_ports_shutdown(duthost, lports, waits.shutdown))

        reset_failures = []
        for module in modules:
            elapsed, err = cli_helpers.sfputil_reset(duthost, module)
            logger.info("sfputil reset of %s took %ss", module, elapsed)
            if err:
                reset_failures.append(err)
        time.sleep(waits.reset_settle)
        failures += _step_failures(3, reset_failures)

        lpm_modules = [
            module for module in modules
            if port_attributes_dict[module][SYSTEM_ATTRIBUTES_KEY]["low_power_mode_supported"]
        ]
        failures += _step_failures(4, _set_lpmode(duthost, lpm_modules, True, waits.lpm_settle))
        failures += _step_failures(5, _set_lpmode(duthost, lpm_modules, False, waits.lpm_settle))
    finally:
        try:
            startup_failures = scenario_ops.perform_ports_startup(duthost, lports, waits.startup)
            failures += _step_failures(6, startup_failures + _check_lldp_present(duthost, lldp_timeouts))
            update_times, errors = capture_update_times(duthost, modules)
            failures += [f"periodic table snapshot: {error}" for error in errors]
        finally:
            failures += _step_failures(7, set_dom_monitoring(duthost, modules, enable=True))

    elapsed = time.monotonic() - started_at
    logger.info("Remote reseat of %d module(s) took %.1fs", len(modules), elapsed)
    if elapsed > timeout_sec:
        failures.append(
            f"remote reseat took {elapsed:.0f}s, longer than its {timeout_sec}s timeout (remote_reseat_timeout_min)")
    return failures, update_times


# ──────────────────────────────────────────────────────────────────────
# LLDP neighbor identity
# ──────────────────────────────────────────────────────────────────────


def peer_lldp_port_ids(peers):
    """Return ``{port: {LLDP port IDs the peer may advertise}}``: its port name and alias."""
    status_by_device = {}
    port_ids = {}
    for port, peer in peers.items():
        if peer.device not in status_by_device:
            status_by_device[peer.device] = get_dut_interfaces_status(peer.host)
        alias = (status_by_device[peer.device].get(peer.port) or {}).get("alias")
        port_ids[port] = {peer.port, alias} - {None, ""}
    return port_ids


def verify_lldp_neighbors(duthost, lldp_timeouts, peers):
    """Verify each port's LLDP neighbor is its connection-graph peer.

    The neighbor's system name must be the peer device and its port ID the
    peer port's name or alias, the latter being what SONiC's lldpd advertises.
    """
    if not lldp_timeouts:
        return []
    port_ids = peer_lldp_port_ids({port: peers[port] for port in lldp_timeouts})
    ports_by_namespace = _group_by_namespace(duthost, lldp_timeouts)

    def _check():
        failures = []
        for namespace, namespace_ports in ports_by_namespace.items():
            neighbors, err = db_helpers.get_db_table(
                duthost, "APPL_DB", "LLDP_ENTRY_TABLE", namespace=namespace, sep=":")
            if err:
                failures += [f"{port}: {err}" for port in namespace_ports]
                continue
            for port in namespace_ports:
                peer = peers[port]
                expected = f"{peer.device} port {' or '.join(natsorted(port_ids[port]))}"
                neighbor = neighbors.get(port)
                if not neighbor:
                    failures.append(f"{port}: no LLDP neighbor, expected {expected}")
                    continue
                system_name = neighbor.get("lldp_rem_sys_name")
                port_id = neighbor.get("lldp_rem_port_id")
                if system_name != peer.device or port_id not in port_ids[port]:
                    failures.append(f"{port}: LLDP neighbor is {system_name} port {port_id}, expected {expected}")
        return failures

    return scenario_ops.poll_ports_recovered(
        _check, max(lldp_timeouts.values()), LLDP_POLL_INTERVAL_SEC, "LLDP neighbor identity")


# ──────────────────────────────────────────────────────────────────────
# Link peers
# ──────────────────────────────────────────────────────────────────────


def _peers_by_device(peers):
    by_device = defaultdict(list)
    for peer in peers.values():
        by_device[peer.device].append(peer)
    return by_device


def _parse_count(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def capture_rx_signal_flag_counts(host, modules):
    """Return ``({module: {flag: change count}}, errors)`` for each module's supported
    Rx LOS / Rx CDR LOL flags in ``TRANSCEIVER_STATUS_FLAG``."""
    counts, errors = {}, []
    for namespace, namespace_modules in _group_by_namespace(host, modules).items():
        flags, flag_err = db_helpers.get_state_db_table(host, "TRANSCEIVER_STATUS_FLAG", namespace=namespace)
        changes, change_err = db_helpers.get_state_db_table(
            host, "TRANSCEIVER_STATUS_FLAG_CHANGE_COUNT", namespace=namespace)
        if flag_err or change_err:
            errors += [err for err in (flag_err, change_err) if err]
            continue
        for module in namespace_modules:
            module_changes = changes.get(module) or {}
            counts[module] = {
                flag: _parse_count(module_changes.get(flag))
                for flag, value in (flags.get(module) or {}).items()
                if _RX_SIGNAL_FLAG_RE.match(flag) and str(value).strip().lower() in _SUPPORTED_FLAG_VALUES
            }
    return counts, errors


def verify_rx_signal_flags_changed(host, baseline, wait_sec):
    """Poll until each module in ``baseline`` (:func:`capture_rx_signal_flag_counts`)
    records a change of at least one of its Rx LOS / Rx CDR LOL flags.

    The flags latch while the link is dark, so xcvrd records the change on its
    next read even after the link is back.  Modules without a supported flag
    are not checked.
    """
    tracked = {module: counts for module, counts in baseline.items() if counts}
    for module in natsorted(set(baseline) - set(tracked)):
        logger.info("%s publishes no supported Rx LOS / Rx CDR LOL flag; skipping its flag check", module)
    if not tracked:
        return []

    def _check():
        current, failures = capture_rx_signal_flag_counts(host, list(tracked))
        for module, before in natsorted(tracked.items()):
            after = current.get(module, {})
            if not any(after.get(flag, 0) > count for flag, count in before.items()):
                failures.append(
                    f"{module}: TRANSCEIVER_STATUS_FLAG_CHANGE_COUNT recorded no change of "
                    f"{', '.join(natsorted(before))} while the link was down")
        return failures

    return scenario_ops.poll_ports_recovered(_check, wait_sec, REFRESH_POLL_INTERVAL_SEC, "peer Rx signal flags")


def capture_peer_baseline(peers):
    """Snapshot each link peer's flap count, transceiver tables and Rx signal flag counts.

    Returns ``({device: PeerBaseline}, errors)`` for ``peers`` (``{port: PeerInfo}``).
    """
    baseline, errors = {}, []
    for device, device_peers in _peers_by_device(peers).items():
        host = device_peers[0].host
        ports = natsorted({peer.port for peer in device_peers})
        modules = natsorted({peer.primary_port for peer in device_peers})
        tables, table_errors = oir_helpers.capture_state_tables(host, natsorted(set(ports) | set(modules)))
        rx_counts, rx_errors = capture_rx_signal_flag_counts(host, modules)
        baseline[device] = PeerBaseline(oir_helpers.get_flap_counts(host, ports), tables, rx_counts)
        errors += [f"peer {device}: {error}" for error in table_errors + rx_errors]
    return baseline, errors


def rebase_peer_rx_signal_flags(peers, baseline):
    """Return ``(baseline, errors)`` with fresh Rx signal flag counts, so that only
    flag changes from the reseats to come are counted."""
    rebased, errors = dict(baseline), []
    for device, device_peers in _peers_by_device(peers).items():
        modules = natsorted({peer.primary_port for peer in device_peers})
        rx_counts, rx_errors = capture_rx_signal_flag_counts(device_peers[0].host, modules)
        rebased[device] = baseline[device]._replace(rx_signal_flag_counts=rx_counts)
        errors += [f"peer {device}: {error}" for error in rx_errors]
    return rebased, errors


def verify_peers(peers, baseline, reseats, startup_wait, publication_wait):
    """Verify the link peers of the reseated ports after ``reseats`` remote reseats.

    Each peer port is oper up and flapped exactly once down and once up per
    reseat; its module, untouched, keeps every transceiver table it had with
    ``TRANSCEIVER_STATUS_SW`` READY; and its Rx LOS / Rx CDR LOL flags recorded
    the outage.  The peer's DOM Rx power and PM averages are not checked: a
    reseat can keep the link dark for less than one xcvrd DOM sweep, so STATE_DB
    need not show them drop, whereas the latched flags always record the outage.
    """
    failures = []
    peers_by_device = _peers_by_device(peers)
    for device in natsorted(peers_by_device):
        device_peers = peers_by_device[device]
        host = device_peers[0].host
        ports = natsorted({peer.port for peer in device_peers})
        modules = natsorted({peer.primary_port for peer in device_peers})
        device_failures = wait_ports_oper_status(host, ports, "up", startup_wait)
        device_failures += oir_helpers.verify_flap_count_increment(
            host, ports, baseline[device].flap_counts, expected_increment=2 * reseats)
        device_failures += oir_helpers.verify_state_tables_present(
            host, natsorted(set(ports) | set(modules)), [], publication_wait,
            baseline_tables=baseline[device].tables, event="the remote reseat")
        device_failures += verify_rx_signal_flags_changed(
            host, baseline[device].rx_signal_flag_counts, publication_wait)
        failures += [f"peer {device}: {failure}" for failure in device_failures]
    return failures
