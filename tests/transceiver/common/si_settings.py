"""Signal-integrity (SI) settings verification building blocks.

Standard Port Recovery and Verification sub-check 6 from
``docs/testplan/transceiver/system_test_plan.md``:

* Optics SI - the host-side SI controls the module runs with must match the
  per-lane ``optics_si_settings`` system attribute.  xcvrd stages the values of
  ``optics_si_settings.json`` into CMIS Staged Control Set 0 and the module
  copies them into the Active Control Set (page 11h) on DataPath init, so the
  active set is read back.
* Media SI - the NPU SerDes settings xcvrd published to APPL_DB ``PORT_TABLE``
  from ``media_settings.json`` must match the ``media_si_settings`` system
  attribute (a ``media_settings.json`` leaf: ``{param: {"lane<N>": value}}``),
  and STATE_DB ``PORT_TABLE`` must show the settings were synced to the NPU.

Both checks are opt-in per port: a port whose attribute is empty or absent is
not checked.  Results use the ``{port: {'passed': bool, 'details': str}}``
shape of the other building blocks in :mod:`tests.transceiver.common.verification`,
with one entry per checked port.
"""
import logging
from collections import defaultdict

from natsort import natsorted

from tests.transceiver.attribute_parser.attribute_keys import SYSTEM_ATTRIBUTES_KEY
from tests.transceiver.common import cli_helpers, db_helpers

logger = logging.getLogger(__name__)

OPTICS_SI_ATTRIBUTE = "optics_si_settings"
MEDIA_SI_ATTRIBUTE = "media_si_settings"

ACTIVE_CONTROL_SET_PAGE = "0x11"
_ACTIVE_CONTROL_SET_SECTION = "upper_page_11"
# CMIS Active Control Set host-lane SI controls as ``(first byte, bits per lane)``,
# lane 1 in the least significant bits.  The layout mirrors Staged Control Set 0
# (page 10h bytes 153-173, ``STAGED_CTRL0_TX_RX_CTRL_FIELD`` in sonic_xcvr) 61
# bytes higher; the keys are the ``optics_si_settings.json`` parameter names.
ACTIVE_CONTROL_SET_SI_LAYOUT = {
    "AdaptiveInputEqEnableTx": (214, 1),
    "AdaptiveInputEqRecalledTx": (215, 2),
    "FixedInputEqTargetTx": (217, 4),
    "CDREnableTx": (221, 1),
    "CDREnableRx": (222, 1),
    "OutputEqPreCursorTargetRx": (223, 4),
    "OutputEqPostCursorTargetRx": (227, 4),
    "OutputAmplitudeTargetRx": (231, 4),
}

NPU_SI_SETTINGS_SYNC_STATUS_KEY = "NPU_SI_SETTINGS_SYNC_STATUS"
# xcvrd moves the STATE_DB status from DEFAULT to NOTIFIED once it publishes the
# media settings to APPL_DB.  DONE, the NPU SI handshake's acknowledgement, is
# accepted but not required: upstream orchagent never writes it, so NOTIFIED is
# the last state those images reach.
NPU_SI_SETTINGS_SYNCED_VALUES = ("NPU_SI_SETTINGS_NOTIFIED", "NPU_SI_SETTINGS_DONE")
# xcvrd slices these media parameters by the gearbox line-side lane count.
GEARBOX_LINE_PARAM_MARKER = "gb_line"


def _si_attribute(port_attributes_dict, port, attribute):
    return port_attributes_dict.get(port, {}).get(SYSTEM_ATTRIBUTES_KEY, {}).get(attribute) or {}


def _parse_int(value):
    """Return ``value`` as an int (decimal or ``0x`` hex string), or ``None``."""
    if isinstance(value, int):
        return int(value)
    text = str(value).strip()
    for base in (0, 10):
        try:
            return int(text, base)
        except ValueError:
            continue
    return None


def _read_port_tables(duthost, db, sep, ports):
    """Return ``({port: PORT_TABLE entry}, {port: read error})``, one dump per namespace."""
    ports_by_namespace = defaultdict(list)
    for port in ports:
        ports_by_namespace[db_helpers.resolve_port_namespace(duthost, port)].append(port)

    entries, errors = {}, {}
    for namespace, namespace_ports in ports_by_namespace.items():
        table, err = db_helpers.get_db_table(duthost, db, "PORT_TABLE", namespace=namespace, sep=sep)
        for port in namespace_ports:
            if err:
                errors[port] = err
            else:
                entries[port] = table.get(port, {})
    return entries, errors


def port_host_lanes(port_entry):
    """Return the 1-based module host lanes of a port, or ``None`` without ``lanes``.

    ``port_entry`` is the port's APPL_DB ``PORT_TABLE`` entry: its ``lanes``
    count is the port's host lane count, and a breakout ``subport`` N owns the
    Nth group of that many module host lanes.
    """
    lanes = port_entry.get("lanes")
    if not lanes:
        return None
    lane_count = len(lanes.split(","))
    subport = _parse_int(port_entry.get("subport") or 0) or 0
    first_lane = lane_count * max(0, subport - 1) + 1
    return list(range(first_lane, first_lane + lane_count))


def decode_active_control_set_value(page_bytes, param, lane):
    """Return ``param``'s value for host ``lane`` from page 11h bytes, or ``None``."""
    first_byte, width = ACTIVE_CONTROL_SET_SI_LAYOUT[param]
    lanes_per_byte = 8 // width
    raw = page_bytes.get(first_byte + (lane - 1) // lanes_per_byte)
    if raw is None:
        return None
    return (raw >> ((lane - 1) % lanes_per_byte * width)) & ((1 << width) - 1)


def _optics_si_mismatches(expected, page_bytes, lanes):
    """Return mismatch strings for ``expected`` (``{param: {param<lane>: value}}``)."""
    mismatches = []
    compared = 0
    for param, lane_values in expected.items():
        if param not in ACTIVE_CONTROL_SET_SI_LAYOUT:
            mismatches.append(f"unsupported optics SI parameter {param!r}")
            continue
        if not isinstance(lane_values, dict):
            mismatches.append(f"{param} must map '{param}<lane>' to a value, got {lane_values!r}")
            continue
        for lane in lanes:
            key = f"{param}{lane}"
            if key not in lane_values:
                continue
            compared += 1
            expected_value = _parse_int(lane_values[key])
            actual_value = decode_active_control_set_value(page_bytes, param, lane)
            if expected_value is None:
                mismatches.append(f"{key} has unparsable expected value {lane_values[key]!r}")
            elif actual_value != expected_value:
                mismatches.append(f"{key}={actual_value} (expected {expected_value})")
    if not compared and not mismatches:
        mismatches.append(f"{OPTICS_SI_ATTRIBUTE} defines no value for host lane(s) {lanes}")
    return mismatches


def check_optics_si_settings(duthost, ports, port_attributes_dict, lport_to_first_subport_mapping):
    """Verify the CMIS Active Control Set SI values of every port with ``optics_si_settings``.

    Only the port's own host lanes are compared, and only the lanes the
    attribute pins.  The page 11h hexdump is read once per module.

    Returns:
        dict: ``{port: {'passed': bool, 'details': str}}`` for the checked ports.
    """
    targets = {
        port: _si_attribute(port_attributes_dict, port, OPTICS_SI_ATTRIBUTE) for port in ports
    }
    targets = {port: expected for port, expected in targets.items() if expected}
    if not targets:
        return {}

    port_entries, read_errors = _read_port_tables(duthost, "APPL_DB", ":", targets)
    pages_by_module = {}
    per_port = {}
    for port, expected in targets.items():
        module = lport_to_first_subport_mapping.get(port, port)
        if module not in pages_by_module:
            sections, err = cli_helpers.sfputil_show_eeprom_hexdump(
                duthost, module, page=ACTIVE_CONTROL_SET_PAGE)
            pages_by_module[module] = (
                (None, err) if err else (sections.get(_ACTIVE_CONTROL_SET_SECTION), None))
        page_bytes, err = pages_by_module[module]
        lanes = port_host_lanes(port_entries.get(port, {}))

        if port in read_errors:
            problems = [f"APPL_DB PORT_TABLE read failed: {read_errors[port]}"]
        elif lanes is None:
            problems = [f"APPL_DB PORT_TABLE:{port} has no 'lanes' field"]
        elif err:
            problems = [err]
        elif not page_bytes:
            problems = [f"no page 11h (Active Control Set) bytes in the EEPROM hexdump of {module}"]
        else:
            problems = _optics_si_mismatches(expected, page_bytes, lanes)

        if problems:
            details = f"{port}: optics SI settings mismatch - " + "; ".join(problems)
            logger.warning("Optics SI check FAILED: %s", details)
        else:
            details = f"{port}: optics SI settings match on host lane(s) {lanes}"
            logger.info("Optics SI check PASSED: %s", details)
        per_port[port] = {"passed": not problems, "details": details}
    return per_port


def expected_media_si_value(value, lane_count, subport):
    """Return the APPL_DB string xcvrd publishes for one ``media_settings.json`` parameter.

    Mirrors xcvrd's media settings parser: per-lane values are taken in natural
    ``lane<N>`` order, sliced to the subport's lanes (from lane 0 when the slice
    would overrun the values), and comma joined; a scalar is published as is.
    """
    if not isinstance(value, dict):
        return str(value)
    lane_values = [value[key] for key in natsorted(value)]
    first = (subport - 1) * lane_count if subport else 0
    if first + lane_count > len(lane_values):
        first = 0
    return ",".join(str(lane_value) for lane_value in lane_values[first:first + lane_count])


def media_si_values_match(actual, expected):
    """Return whether two comma-separated SI value strings are equal, value by value.

    Values compare numerically when both parse as integers, so ``0x1E`` matches
    ``0x1e`` and ``30``.
    """
    if actual is None:
        return False
    actual_values = [token.strip() for token in str(actual).split(",")]
    expected_values = [token.strip() for token in str(expected).split(",")]
    if len(actual_values) != len(expected_values):
        return False
    for actual_value, expected_value in zip(actual_values, expected_values):
        actual_int, expected_int = _parse_int(actual_value), _parse_int(expected_value)
        if actual_int is not None and expected_int is not None:
            if actual_int != expected_int:
                return False
        elif actual_value.lower() != expected_value.lower():
            return False
    return True


def _read_gearbox_line_lane_counts(duthost, ports):
    """Return ``{port: line-side lane count}`` for the gearbox ports, as xcvrd reads them."""
    line_lane_counts = {}
    for namespace in {db_helpers.resolve_port_namespace(duthost, port) for port in ports}:
        table, err = db_helpers.get_db_table(duthost, "APPL_DB", "_GEARBOX_TABLE", namespace=namespace, sep=":")
        if err:
            logger.warning("Cannot read APPL_DB _GEARBOX_TABLE: %s", err)
            continue
        for key, entry in table.items():
            if key.startswith("interface:") and entry.get("name") and entry.get("line_lanes"):
                line_lane_counts[entry["name"]] = len(entry["line_lanes"].split(","))
    return line_lane_counts


def check_media_si_settings(duthost, ports, port_attributes_dict):
    """Verify the published media SI settings of every port with ``media_si_settings``.

    Each parameter in the attribute must match the port's APPL_DB
    ``PORT_TABLE`` field, and STATE_DB ``PORT_TABLE`` must show the settings
    were synced to the NPU (see ``NPU_SI_SETTINGS_SYNCED_VALUES``).

    Returns:
        dict: ``{port: {'passed': bool, 'details': str}}`` for the checked ports.
    """
    targets = {
        port: _si_attribute(port_attributes_dict, port, MEDIA_SI_ATTRIBUTE) for port in ports
    }
    targets = {port: expected for port, expected in targets.items() if expected}
    if not targets:
        return {}

    appl_entries, appl_errors = _read_port_tables(duthost, "APPL_DB", ":", targets)
    state_entries, state_errors = _read_port_tables(duthost, "STATE_DB", "|", targets)
    gearbox_line_lane_counts = _read_gearbox_line_lane_counts(duthost, targets)
    per_port = {}
    for port, expected in targets.items():
        appl_entry = appl_entries.get(port, {})
        problems = []
        if port in appl_errors:
            problems.append(f"APPL_DB PORT_TABLE read failed: {appl_errors[port]}")
        elif not appl_entry.get("lanes"):
            problems.append(f"APPL_DB PORT_TABLE:{port} has no 'lanes' field")
        else:
            lane_count = len(appl_entry["lanes"].split(","))
            subport = _parse_int(appl_entry.get("subport") or 0) or 0
            for param, value in expected.items():
                param_lane_count = lane_count
                if GEARBOX_LINE_PARAM_MARKER in param and port in gearbox_line_lane_counts:
                    param_lane_count = gearbox_line_lane_counts[port]
                expected_value = expected_media_si_value(value, param_lane_count, subport)
                if not media_si_values_match(appl_entry.get(param), expected_value):
                    problems.append(f"{param}={appl_entry.get(param)!r} (expected {expected_value!r})")

        sync_status = None
        if port in state_errors:
            problems.append(f"STATE_DB PORT_TABLE read failed: {state_errors[port]}")
        else:
            sync_status = state_entries.get(port, {}).get(NPU_SI_SETTINGS_SYNC_STATUS_KEY)
            if sync_status not in NPU_SI_SETTINGS_SYNCED_VALUES:
                problems.append(
                    f"{NPU_SI_SETTINGS_SYNC_STATUS_KEY}={sync_status!r} "
                    f"(expected one of {', '.join(NPU_SI_SETTINGS_SYNCED_VALUES)})")

        if problems:
            details = f"{port}: media SI settings mismatch - " + "; ".join(problems)
            logger.warning("Media SI check FAILED: %s", details)
        else:
            details = (f"{port}: media SI settings match ({', '.join(natsorted(expected))}), "
                       f"{NPU_SI_SETTINGS_SYNC_STATUS_KEY}={sync_status}")
            logger.info("Media SI check PASSED: %s", details)
        per_port[port] = {"passed": not problems, "details": details}
    return per_port
