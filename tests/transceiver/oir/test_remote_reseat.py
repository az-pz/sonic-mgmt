"""Remote reseat.

Implements the Remote reseat test cases TC1-TC2 from
``docs/testplan/transceiver/online_insertion_removal_testplan.md``.

A remote reseat runs the plan's seven-step procedure (DOM monitoring off, port
shutdown, module reset, low-power mode on and off, port startup, DOM monitoring
on) on every module under test together, checking each step's expected result
as it runs.  After the last reseat, the TC1 expected results are verified on
every port under test and its link peer:

  1. EEPROM content from the show and sfputil CLIs (exit code 0) matches the inventory.
  2. DOM values are republished and valid; VDM and PM tables are back.
  3. The ports are oper up within ``port_startup_wait_sec`` and the system healthy.
  4. No link flap for ``link_flap_monitor_timeout_sec``; each reseat takes a
     port down and up exactly once.
  5. Optics and media SI settings match the inventory.
  6. The LLDP neighbor is the port's connection-graph peer.
  7. The transceiver state and flag tables of the ports and their peers are updated.
  8. No transceiver kernel error, when ``monitor_kernel_errors`` is set.

No other inventory port may flap.  Per-port failures are aggregated into a
single ``pytest.fail`` so one run surfaces every issue across every port.
"""
import logging
import time
from collections import namedtuple

import pytest

from tests.transceiver.attribute_parser.attribute_keys import REMOTE_RESEAT_ATTRIBUTES_KEY
from tests.transceiver.common.health_checks import capture_baseline, verify_health
from tests.transceiver.common.si_settings import check_media_si_settings, check_optics_si_settings
from tests.transceiver.common.verification import standard_port_recovery_and_verification
from tests.transceiver.dom import dom_helpers
from tests.transceiver.eeprom import recovery
from tests.transceiver.oir import oir_helpers, remote_reseat_helpers

logger = logging.getLogger(__name__)

pytestmark = pytest.mark.usefixtures("gold_fw_verified", "_restore_reseat_state")

_ReseatScope = namedtuple(
    "_ReseatScope", ("lports", "modules", "waits", "timeout_sec", "lldp_timeouts", "peers", "remote_peers"))
_ReseatBaseline = namedtuple("_ReseatBaseline", ("tables", "peers", "flap_counts", "health", "watermark"))


def _remaining_wait(deadline):
    """Return the non-negative time remaining before ``deadline``."""
    return max(0, deadline - time.monotonic())


def _failed_details(results):
    """Return the details of the failed ``{port: {'passed', 'details'}}`` results."""
    return [result["details"] for result in results.values() if not result["passed"]]


def _reseat_scope(duthost, port_attributes_dict, lport_to_first_subport_mapping,
                  reseat_pport_to_lports, reseat_link_peers):
    lports = [port for ports in reseat_pport_to_lports.values() for port in ports]
    modules = remote_reseat_helpers.modules_of(lports, lport_to_first_subport_mapping)
    under_test = set(lports)
    return _ReseatScope(
        lports=lports,
        modules=modules,
        waits=remote_reseat_helpers.reseat_waits(port_attributes_dict, lports, modules),
        timeout_sec=remote_reseat_helpers.reseat_timeout_sec(port_attributes_dict, lports),
        lldp_timeouts=remote_reseat_helpers.lldp_port_timeouts(port_attributes_dict, lports),
        peers=reseat_link_peers,
        # A peer that is itself reseated is verified as a port under test.
        remote_peers={
            lport: peer for lport, peer in reseat_link_peers.items()
            if not (peer.device == duthost.hostname and peer.port in under_test)
        },
    )


def _verify_reseat(duthost, port_attributes_dict, lport_to_first_subport_mapping,
                   scope, baseline, update_times, reseats):
    """TC1 expected results after the last of ``reseats`` remote reseats."""
    lports, modules, waits = scope.lports, scope.modules, scope.waits
    recovery_start = time.monotonic()
    startup_deadline = recovery_start + waits.startup
    publication_deadline = recovery_start + waits.publication

    # 3. Oper up within port_startup_wait_sec, with every monitored process healthy.
    port_recovery = standard_port_recovery_and_verification(
        duthost, lports, {port: port_attributes_dict[port] for port in lports},
        link_up_timeout_sec=_remaining_wait(startup_deadline),
        health_baseline=baseline.health,
        lport_to_first_subport_mapping=lport_to_first_subport_mapping,
    )
    failures = [] if port_recovery["passed"] else [port_recovery["details"]]
    # 4. The shutdown and the startup of each reseat are the only transitions.
    failures += oir_helpers.verify_flap_count_increment(
        duthost, lports, baseline.flap_counts, expected_increment=2 * reseats)

    # 1. EEPROM content, from STATE_DB and from the module itself.
    failures += oir_helpers.verify_presence_clis(duthost, lports, present=True)
    failures += recovery.verify_transceiver_recovery(
        duthost,
        port_attributes_dict,
        lport_to_first_subport_mapping,
        _remaining_wait(startup_deadline),
        "after remote reseat",
        ports=lports,
        firmware_wait_sec=_remaining_wait(publication_deadline),
        # The reset re-initialized the module, so always re-read its EEPROM.
        live_i2c_confirm=True,
    )

    # 7. A reset is no removal: every transceiver table stays, TRANSCEIVER_STATUS_SW
    # returns to READY, and the DOM, status, VDM and flag tables are republished
    # once DOM monitoring is back.  This also proves step 7 took effect.
    failures += oir_helpers.verify_state_tables_present(
        duthost, lports, [], _remaining_wait(publication_deadline), baseline_tables=baseline.tables,
        event="the remote reseat")
    failures += remote_reseat_helpers.verify_tables_refreshed(
        duthost, update_times, _remaining_wait(publication_deadline))

    # 2. DOM values (VDM and PM presence is covered by the table checks above).
    dom_failures = oir_helpers.verify_dom_data_recovered(
        duthost, port_attributes_dict, lport_to_first_subport_mapping, lports, {},
        _remaining_wait(publication_deadline))
    if dom_failures:
        failures.append("DOM sensor data:\n  " + "\n  ".join(dom_failures))
    failures += dom_helpers.verify_dom_thresholds_after_operation(duthost, port_attributes_dict, modules)

    # 5. SI settings re-applied to the module and still published to the NPU.
    failures += _failed_details(check_optics_si_settings(
        duthost, lports, port_attributes_dict, lport_to_first_subport_mapping))
    failures += _failed_details(check_media_si_settings(duthost, lports, port_attributes_dict))

    # 6. LLDP neighbor information.
    failures += remote_reseat_helpers.verify_lldp_neighbors(duthost, scope.lldp_timeouts, scope.peers)

    # 7. Link peers.
    failures += remote_reseat_helpers.verify_peers(
        scope.remote_peers, baseline.peers, reseats,
        _remaining_wait(startup_deadline), _remaining_wait(publication_deadline))

    # 4. No flap for link_flap_monitor_timeout_sec from recovery on.
    failures += oir_helpers.verify_no_link_flap(
        duthost,
        port_attributes_dict,
        lports,
        sentinels=port_recovery["post_recovery_sentinels"],
        observation_start=port_recovery["post_recovery_started_at"],
        attribute_key=REMOTE_RESEAT_ATTRIBUTES_KEY,
    )
    # 8. Kernel errors.
    failures += oir_helpers.verify_no_kernel_errors(duthost, baseline.watermark)
    return failures


def _run_remote_reseats(duthost, port_attributes_dict, lport_to_first_subport_mapping,
                        reseat_pport_to_lports, reseat_link_peers, reseats, label):
    """Remotely reseat every module under test ``reseats`` times, then verify TC1.

    Baselines are taken once, before the first reseat, so the system health,
    the flap counts and the kernel log are checked across every reseat; only
    the peers' Rx signal flag counts are taken again before the last reseat,
    whose own outage they must record.
    """
    scope = _reseat_scope(duthost, port_attributes_dict, lport_to_first_subport_mapping,
                          reseat_pport_to_lports, reseat_link_peers)
    lports = scope.lports

    baseline_tables, setup_failures = oir_helpers.capture_state_tables(duthost, lports)
    peer_baseline, peer_errors = remote_reseat_helpers.capture_peer_baseline(scope.remote_peers)
    setup_failures += peer_errors
    if setup_failures:
        pytest.fail(f"{label} setup failures:\n  - " + "\n  - ".join(setup_failures))
    same_dut_peers = {
        lport: peer.port for lport, peer in scope.peers.items() if peer.device == duthost.hostname
    }
    other_flap_baseline = oir_helpers.get_other_ports_flap_counts(
        duthost, port_attributes_dict, lports, same_dut_peers)
    baseline = _ReseatBaseline(
        tables=baseline_tables,
        peers=peer_baseline,
        flap_counts=oir_helpers.get_flap_counts(duthost, lports),
        health=capture_baseline(duthost),
        watermark=oir_helpers.capture_kernel_error_watermark(
            duthost, port_attributes_dict, lports, attribute_key=REMOTE_RESEAT_ATTRIBUTES_KEY),
    )

    failures = []
    update_times = {}
    reseat_failed = False
    for reseat in range(1, reseats + 1):
        if reseats > 1 and reseat == reseats:
            # The peer flags must record the last reseat itself, not only an earlier one.
            peer_baseline, errors = remote_reseat_helpers.rebase_peer_rx_signal_flags(
                scope.remote_peers, baseline.peers)
            baseline = baseline._replace(peers=peer_baseline)
            failures += [f"peer baseline before reseat {reseat}/{reseats}: {error}" for error in errors]
        logger.info("Remote reseat %d/%d of %d module(s)", reseat, reseats, len(scope.modules))
        reseat_failures, update_times = remote_reseat_helpers.perform_remote_reseat(
            duthost, port_attributes_dict, lports, scope.modules, scope.waits,
            scope.lldp_timeouts, scope.timeout_sec)
        if reseat_failures:
            prefix = f"reseat {reseat}/{reseats}: " if reseats > 1 else ""
            failures += [prefix + failure for failure in reseat_failures]
            reseat_failed = True
            break

    if reseat_failed:
        # The reseat did not complete; still report what does not depend on it.
        health = verify_health(duthost, baseline.health)
        failures += [f"health: {failure}" for failure in health["failures"]]
        failures += oir_helpers.verify_no_kernel_errors(duthost, baseline.watermark)
    else:
        failures += _verify_reseat(
            duthost, port_attributes_dict, lport_to_first_subport_mapping,
            scope, baseline, update_times, reseats)
    failures += oir_helpers.verify_other_ports_no_flap(
        duthost, other_flap_baseline, cause="the ports under test were remotely reseated")
    return failures


def test_remote_reseat(
    duthost, port_attributes_dict, lport_to_first_subport_mapping, reseat_pport_to_lports, reseat_link_peers,
):
    """TC1: remotely reseat every module under test and verify its ports and link peers recover."""
    failures = _run_remote_reseats(
        duthost, port_attributes_dict, lport_to_first_subport_mapping,
        reseat_pport_to_lports, reseat_link_peers, reseats=1, label="Remote reseat (TC1)")

    if failures:
        pytest.fail("Remote reseat (TC1) failures:\n  - " + "\n  - ".join(failures))


def test_remote_reseat_stress(
    duthost, port_attributes_dict, lport_to_first_subport_mapping, reseat_pport_to_lports, reseat_link_peers,
):
    """TC2: remotely reseat every module under test ``remote_reseat_stress_iteration``
    times and verify the TC1 expected results after the last reseat."""
    first_port = next(iter(reseat_pport_to_lports.values()))[0]
    iterations = port_attributes_dict[first_port][REMOTE_RESEAT_ATTRIBUTES_KEY]["remote_reseat_stress_iteration"]
    if not isinstance(iterations, int) or iterations < 1:
        pytest.fail(
            "remote_reseat_stress_iteration must be a positive integer to exercise a stress "
            f"cycle, got {iterations!r}")

    failures = _run_remote_reseats(
        duthost, port_attributes_dict, lport_to_first_subport_mapping,
        reseat_pport_to_lports, reseat_link_peers, reseats=iterations, label="Remote reseat stress (TC2)")

    if failures:
        pytest.fail("Remote reseat stress (TC2) failures:\n  - " + "\n  - ".join(failures))
