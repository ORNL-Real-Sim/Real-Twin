"""Regression tests for retaining SUMO identities during OpenDrive conversion."""

import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

from realtwin.func_lib._c_abstract_scenario import _abstractScenario as abstract
from realtwin.func_lib._c_abstract_scenario import rt_matchup_table_generation as matchup
from realtwin.func_lib._e_simulation import _sumo as simulation
from realtwin.func_lib._f_calibration.algo_sumo import util_cali_turn_inflow as calibration
from realtwin.rt_sumo import rt_sumo, xodr_sumo
from realtwin.util_lib import mapping_SUMO_OpenDrive_ID as mapping


@pytest.fixture
def native_network(tmp_path):
    """Build a native fixture with mixed IDs, multiple lanes, and a named TLS."""
    if shutil.which("netconvert") is None:
        pytest.skip("Native SUMO netconvert is unavailable")
    pytest.importorskip("sumolib")
    nodes = tmp_path / "nodes.nod.xml"
    edges = tmp_path / "edges.edg.xml"
    nodes.write_text(
        '<nodes><node id="west" x="0" y="0"/>'
        '<node id="signal_junction" x="100" y="0" type="traffic_light" tl="controller_custom"/>'
        '<node id="east" x="200" y="0"/>'
        '<node id="north" x="100" y="100"/></nodes>', encoding="utf-8"
    )
    edges.write_text(
        '<edges><edge id="positive_link" from="west" to="signal_junction" numLanes="2" speed="13.89"/>'
        '<edge id="-42" from="signal_junction" to="east" numLanes="2" speed="13.89"/>'
        '<edge id="north_in" from="north" to="signal_junction"/>'
        '<edge id="north_out" from="signal_junction" to="north"/></edges>', encoding="utf-8"
    )
    net_file = tmp_path / "source.net.xml"
    subprocess.run(
        ["netconvert", "-n", str(nodes), "-e", str(edges), "-o", str(net_file)],
        check=True, capture_output=True, text=True,
    )
    # Exporter order differs from this source order, so positional mapping fails.
    tree = ET.parse(net_file)
    root = tree.getroot()
    source_edges = root.findall("edge")
    for edge in source_edges:
        root.remove(edge)
    for index, edge in enumerate(reversed(source_edges)):
        root.insert(index, edge)
    tree.write(net_file, encoding="utf-8", xml_declaration=True)
    return net_file


def test_native_round_trip_retains_ids_and_source(native_network, tmp_path):
    original_bytes = native_network.read_bytes()
    original = ET.fromstring(original_bytes)
    xodr_file = tmp_path / "nested/export.xodr"
    restored_file = tmp_path / "separate/restored.net.xml"

    odr_map = xodr_sumo.xodr_from_net_xml(native_network, xodr_file)
    xodr_bytes = xodr_file.read_bytes()
    net = xodr_sumo.xodr_to_net_xml(xodr_file, restored_file)

    assert native_network.read_bytes() == original_bytes
    assert xodr_file.read_bytes() == xodr_bytes
    assert odr_map.xodr_file == str(xodr_file)
    assert net.getEdge("positive_link").getLane(0).getID() == "positive_link_0"
    xodr = ET.parse(xodr_file).getroot()
    assert {road.get("id") for road in xodr.findall("road") if road.get("junction") == "-1"} == {
        edge.get("id") for edge in original.findall("edge") if edge.get("function") != "internal"
    }
    assert {junction.get("id") for junction in xodr.findall("junction")} == {"north", "signal_junction"}
    road_ids = {road.get("id") for road in xodr.findall("road")}
    for connection in xodr.findall("./junction/connection"):
        assert connection.get("incomingRoad") in road_ids
        assert connection.get("connectingRoad") in road_ids

    restored = ET.parse(restored_file).getroot()
    external = lambda root: {
        edge.get("id"): (edge.get("from"), edge.get("to"),
                         [(lane.get("id"), lane.get("index")) for lane in edge.findall("lane")])
        for edge in root.findall("edge") if edge.get("function") != "internal"
    }
    assert external(restored) == external(original)
    for edge in restored.findall("edge"):
        if edge.get("function") != "internal":
            for lane in edge.findall("lane"):
                origin = lane.find("param[@key='origId']").get("value")
                assert origin.rsplit("_", 1)[0] == edge.get("id")
    assert {junction.get("id") for junction in restored.findall("junction")} == {
        junction.get("id") for junction in original.findall("junction")
    }
    assert [ET.canonicalize(ET.tostring(logic), strip_text=True) for logic in restored.findall("tlLogic")] == [
        ET.canonicalize(ET.tostring(logic), strip_text=True) for logic in original.findall("tlLogic")
    ]
    attrs = ("from", "to", "fromLane", "toLane", "tl", "linkIndex")
    controlled = lambda root: {tuple(conn.get(attr) for attr in attrs)
                               for conn in root.findall("connection") if conn.get("tl")}
    assert controlled(restored) == controlled(original)
    edge_ids = {edge.get("id") for edge in restored.findall("edge")}
    lane_ids = {lane.get("id") for lane in restored.findall("./edge/lane")}
    for junction in restored.findall("junction"):
        for attr in ("incLanes", "intLanes"):
            assert set(junction.get(attr, "").split()) <= lane_ids
    for connection in restored.findall("connection"):
        assert connection.get("from") in edge_ids
        assert connection.get("to") in edge_ids
        assert set(connection.get("via", "").split()) <= lane_ids
    # Re-read the final XML with the native tool to catch broken references.
    subprocess.run(["netconvert", "-s", str(restored_file), "-o", str(tmp_path / "checked.net.xml")],
                   check=True, capture_output=True, text=True)


@pytest.mark.parametrize("supplied", [False, True])
def test_generate_inputs_exports_without_rewriting_source(native_network, tmp_path, monkeypatch, supplied):
    config = {"input_dir": str(tmp_path / "input"), "output_dir": str(tmp_path / "output"),
              "demo_data": False, "Network": {"NetworkName": "test"},
              "SUMO": {"is_control_data": False}}
    source_bytes = native_network.read_bytes()
    monkeypatch.setattr(rt_sumo, "load_input_configs", Mock(return_value=config))
    monkeypatch.setattr(rt_sumo, "format_junction_bearing", Mock(return_value=pd.DataFrame()))
    monkeypatch.setattr(rt_sumo, "generate_matchup_table", Mock())
    create_network = Mock(side_effect=lambda scenario: shutil.copy(
        native_network, Path(config["output_dir"]) / "OpenDrive/test.net.xml"))
    monkeypatch.setattr(abstract.AbstractScenario, "create_SUMO_network", lambda scenario: create_network(scenario))
    twin = rt_sumo.RealTwinSUMO("config.yaml", input_confirm=False)

    twin.generate_inputs(incl_sumo_net=str(native_network) if supplied else "")

    assert native_network.read_bytes() == source_bytes
    assert (Path(config["output_dir"]) / "OpenDrive/test.net.xml").read_bytes() == source_bytes
    paths = twin.abstract_scenario.Network.OpenDriveNetwork.OpenDrive_network
    assert paths == [str(Path(config["output_dir"]) / "OpenDrive/test.xodr").replace("\\", "/"), None]
    assert ET.parse(paths[0]).getroot().tag == "OpenDRIVE"
    assert create_network.call_count == int(not supplied)


@pytest.mark.parametrize("road, expected", [("42", "42"), ("-42", "-42"), ("link_with_underscore", "link_with_underscore")])
def test_resolve_edge_prefers_exact_ids(road, expected):
    assert mapping.resolve_sumo_edge_id(road, {"42", "-42", "link_with_underscore"}) == expected


def test_resolve_edge_supports_legacy_table():
    assert mapping.resolve_sumo_edge_id("42", {"-42"}) == "-42"
    with pytest.raises(ValueError, match="not in the SUMO network"):
        mapping.resolve_sumo_edge_id("missing", {"-42"})


def test_export_mapping_uses_metadata_and_reserves_road_ids(tmp_path):
    net_file = tmp_path / "source.net.xml"
    xodr_file = tmp_path / "export.xodr"
    net_file.write_text('<net><edge id="2" from="start" to="junction"/><edge id="other" from="junction" to="end"/>'
                        '<junction id="junction"/></net>', encoding="utf-8")
    xodr_file.write_text('<OpenDRIVE><road id="2" name=":junction_0" junction="8"/>'
                         '<road id="0" junction="-1"><userData code="sumoId" value="other"/></road>'
                         '<road id="1" junction="-1"><userData code="sumoId" value="2"/></road>'
                         '<junction id="8" name="junction"><connection id="0" incomingRoad="1" connectingRoad="2"/></junction>'
                         '</OpenDRIVE>', encoding="utf-8")
    xodr_sumo._restore_opendrive_ids(xodr_file, net_file)
    root = ET.parse(xodr_file).getroot()
    assert [road.get("id") for road in root.findall("road")] == [":junction_0", "other", "2"]
    assert root.find("./junction/connection").get("connectingRoad") == ":junction_0"
    assert root.find("./junction/connection").get("incomingRoad") == "2"
    assert root.find("road").get("junction") == "junction"


def test_export_rejects_missing_edge_metadata(tmp_path):
    net_file = tmp_path / "source.net.xml"
    xodr_file = tmp_path / "export.xodr"
    net_file.write_text('<net><edge id="42"/></net>', encoding="utf-8")
    xodr_file.write_text('<OpenDRIVE><road id="0" junction="-1"/></OpenDRIVE>', encoding="utf-8")
    with pytest.raises(ValueError, match="No source SUMO edge mapping"):
        xodr_sumo._restore_opendrive_ids(xodr_file, net_file)


def test_existing_tls_id_is_not_reassigned(native_network, monkeypatch):
    original_bytes = native_network.read_bytes()
    table = pd.DataFrame([{"JunctionID_OpenDrive": "signal_junction", "Turn_Synchro": "NBT"}])
    for column in ("File_GridSmart", "Date_GridSmart", "IntersectionName_GridSmart", "File_Synchro",
                   "IntersectionID_Synchro", "Need calibration?"):
        table[column] = ""
    monkeypatch.setattr(mapping.pd, "read_excel", Mock(return_value=table))
    run = Mock(side_effect=AssertionError("Existing controller should not be rebuilt"))
    monkeypatch.setattr(mapping.subprocess, "run", run)
    assert mapping.parse_SUMO_TLS_ID("unused.xlsx", str(native_network))
    run.assert_not_called()
    assert native_network.read_bytes() == original_bytes


def test_signal_connection_lookup_accepts_exact_and_legacy_ids(tmp_path, monkeypatch):
    net_file = tmp_path / "source.net.xml"
    net_file.write_text('<net><edge id="incoming_link"/><edge id="-42"/>'
                        '<connection from="incoming_link" to="-42" linkIndex="8"/></net>', encoding="utf-8")
    table = pd.DataFrame([{"JunctionID_OpenDrive": "junction", "FromRoadID_OpenDrive": "incoming_link",
                           "ToRoadID_OpenDrive": "42", "Turn_Synchro": "NBT"}])
    for column in ("File_GridSmart", "Date_GridSmart", "IntersectionName_GridSmart", "File_Synchro",
                   "IntersectionID_Synchro", "Need calibration?"):
        table[column] = ""
    monkeypatch.setattr(simulation.pd, "read_excel", Mock(return_value=table))
    assert simulation.update_net_before_signal(str(net_file), "unused.xlsx")
    connection = ET.parse(net_file).getroot().find("connection")
    assert connection.get("from") == "incoming_link"
    assert connection.get("to") == "-42"
    assert connection.get("linkIndex") == "0"


def test_inflow_follows_edge_ids_with_underscores(native_network):
    table = pd.DataFrame([{"FromRoadID_OpenDrive": "positive_link", "Need calibration?": "N"}])
    turns = pd.DataFrame([{"Time": "08:00", "IntersectionName": "intersection", "NBT": 12}])
    ids = pd.DataFrame([{"IntersectionName": "intersection", "Turn": "NBT",
                         "OpenDriveFromID": "positive_link", "OpenDriveToID": "-42"}])
    inflows, _, _ = calibration.generate_inflow(str(native_network), table, turns, ids)
    assert inflows["OpenDriveFromID"].tolist() == ["positive_link"]
    assert inflows["Count"].tolist() == [12]


def test_turn_summary_uses_network_ids(native_network):
    table = pd.DataFrame([{"IntersectionName_GridSmart": "intersection", "Turn_GridSmart": "NBT",
                           "FromRoadID_OpenDrive": "positive_link", "ToRoadID_OpenDrive": "42",
                           "Need calibration?": "N", "Bearing": "0", "Numbering": "0",
                           "JunctionID_OpenDrive": "signal_junction", "Turn": "thru"}])
    turns = pd.DataFrame([{"IntersectionName": "intersection", "IntervalStart": 28800,
                           "IntervalEnd": 29700, "NBT": 12}])
    _, ratios, summary, _, _ = calibration.generate_turn_summary(turns, table, 0, path_net=str(native_network))
    assert ratios["OpenDriveFromID"].tolist() == ["positive_link"]
    assert ratios["OpenDriveToID"].tolist() == ["-42"]
    assert summary["entrance_sumo"].tolist() == ["positive_link"]
    assert summary["exit_sumo"].tolist() == ["-42"]


def test_failed_netconvert_reports_stderr(monkeypatch):
    monkeypatch.setattr(xodr_sumo.subprocess, "run", Mock(return_value=SimpleNamespace(
        returncode=1, stdout="", stderr="bad OpenDrive input")))
    with pytest.raises(RuntimeError, match="bad OpenDrive input"):
        xodr_sumo._run_netconvert(["netconvert"])


def test_generic_opendrive_keeps_generated_direction_ids(native_network, tmp_path):
    xodr_file = tmp_path / "native.xodr"
    net_file = tmp_path / "generic.net.xml"
    subprocess.run(["netconvert", "-s", str(native_network), "--opendrive-output", str(xodr_file)],
                   check=True, capture_output=True, text=True)
    xodr_sumo.xodr_to_net_xml(xodr_file, net_file)
    road_ids = {road.get("id") for road in ET.parse(xodr_file).getroot().findall("road")
                if road.get("junction") == "-1"}
    edge_ids = {edge.get("id") for edge in ET.parse(net_file).getroot().findall("edge")
                if edge.get("function") != "internal"}
    assert edge_ids == {f"-{road_id}" for road_id in road_ids}


@pytest.mark.parametrize("extra_controlled_connection", [False, True])
def test_signal_restore_rejects_changed_lane_connections(extra_controlled_connection):
    imported = ET.fromstring('<net><connection from="a" to="b" fromLane="0" toLane="0"/></net>')
    source = ET.fromstring('<OpenDRIVE><userData code="realtwin.sumo_signals">'
                          '<sumoConnection from="a" to="b" fromLane="1" toLane="0" tl="original"/>'
                          '</userData></OpenDRIVE>')
    if extra_controlled_connection:
        imported.find("connection").set("fromLane", "1")
        ET.SubElement(imported, "connection", {"from": "a", "to": "b", "fromLane": "0",
                                               "toLane": "0", "tl": "generated"})
    with pytest.raises(ValueError, match="controlled lane connections changed"):
        xodr_sumo._restore_sumo_signals(imported, source)


def test_split_road_keeps_distinct_segment_ids(tmp_path):
    xodr_file = tmp_path / "split.xodr"
    net_file = tmp_path / "split.net.xml"
    xodr_file.write_text('<OpenDRIVE><road id="1"><userData><vector key="pyopendrive.original_link_id" '
                         'value="original"/></userData></road></OpenDRIVE>', encoding="utf-8")
    net_file.write_text('<net><edge id="-1#0"><lane id="-1#0_0" index="0"><param key="origId" '
                       'value="1_-1"/></lane></edge><edge id="-1#1"><lane id="-1#1_0" index="0">'
                       '<param key="origId" value="1_-1"/></lane></edge></net>', encoding="utf-8")
    xodr_sumo._annotate_sumo_net_ids(net_file, xodr_file)
    root = ET.parse(net_file).getroot()
    assert [edge.get("id") for edge in root.findall("edge")] == ["-1#0", "-1#1"]
    assert [edge.find("param").get("value") for edge in root.findall("edge")] == ["original", "original"]


def test_matchup_table_keeps_negative_ids(tmp_path, monkeypatch):
    bearings = pd.DataFrame([{
        "Junction ID": "junction", "Approach Edge": "-42", "Degree": 0,
        "Runway Bearing": 0, "Second Last Point (x, y)": (0, 0),
        "Last Point (x, y)": (0, 1), "Second Last Point (lat, lon)": (0, 0),
        "Last Point (lat, lon)": (0, 1),
    }])
    monkeypatch.setattr(matchup, "generate_junction_bearing", Mock(return_value=bearings))
    monkeypatch.setattr(matchup, "get_net_connections", Mock(return_value=pd.DataFrame([
        {"FromEdge": "-42", "ToEdge": "positive_link", "Direction": "thru"}
    ])))
    table = matchup.format_junction_bearing("unused.net.xml")
    assert table["FromRoadID_OpenDrive"].tolist() == ["-42"]
    assert table["ToRoadID_OpenDrive"].tolist() == ["positive_link"]


@pytest.mark.parametrize("function", [xodr_sumo.xodr_from_net_xml, xodr_sumo.xodr_to_net_xml])
def test_conversion_rejects_overwriting_input(tmp_path, monkeypatch, function):
    source = tmp_path / "source.xml"
    source.write_text("<network/>", encoding="utf-8")
    monkeypatch.setattr(xodr_sumo, "_require_sumolib", Mock())
    with pytest.raises(ValueError, match="must differ"):
        function(source, source)
    assert source.read_text(encoding="utf-8") == "<network/>"


@pytest.mark.parametrize("function", [xodr_sumo.xodr_from_net_xml, xodr_sumo.xodr_to_net_xml])
def test_conversion_requires_original_names(tmp_path, monkeypatch, function):
    source = tmp_path / "source.xml"
    source.write_text("<network/>", encoding="utf-8")
    monkeypatch.setattr(xodr_sumo, "_require_sumolib", Mock())
    monkeypatch.setattr(xodr_sumo, "_netconvert_binary", Mock(return_value="netconvert"))
    with pytest.raises(ValueError, match="original-names is required"):
        function(source, tmp_path / "output.xml", netconvert_options={"--output.original-names": False})


@pytest.mark.parametrize("calibration_routes", [False, True])
def test_native_routing_uses_original_edge_ids(native_network, tmp_path, monkeypatch, calibration_routes):
    if shutil.which("jtrrouter") is None:
        pytest.skip("Native SUMO jtrrouter is unavailable")
    turn_ratios = pd.DataFrame([{"IntersectionName": "intersection", "IntervalStart": 0,
                                 "IntervalEnd": 60, "OpenDriveFromID": "positive_link",
                                 "OpenDriveToID": "-42", "TurnRatio": 1}])
    inflows = pd.DataFrame([{"IntervalStart": 0, "IntervalEnd": 60,
                             "OpenDriveFromID": "positive_link", "Count": 3}])
    route_dir = tmp_path / "routes"
    route_dir.mkdir()
    route_file = route_dir / "test.rou.xml"
    if calibration_routes:
        calibration.run_jtrrouter_to_create_rou_xml("test", str(native_network), turn_ratios, inflows,
                                                   str(route_file), 0, 60)
    else:
        monkeypatch.setattr(simulation, "read_MatchupTable", Mock(return_value=pd.DataFrame()))
        monkeypatch.setattr(simulation, "generate_turn_demand_cali", Mock(return_value=(pd.DataFrame(), pd.DataFrame())))
        monkeypatch.setattr(simulation, "generate_inflow", Mock(return_value=(inflows, [], 0)))
        monkeypatch.setattr(simulation, "generate_turn_summary", Mock(return_value=(pd.DataFrame(), turn_ratios, pd.DataFrame(), 0, 0)))
        prep = simulation.SUMOPrep()
        prep.Network = str(native_network)
        prep.SUMOPath = str(route_dir)
        scenario = SimpleNamespace(Supply=SimpleNamespace(NetworkName="test"),
                                   input_config={"input_dir": str(tmp_path / "input")})
        prep.importDemand(scenario, 0, 60, [101])
    assert ET.parse(route_dir / "test.flow.xml").getroot().find("flow").get("from") == "positive_link"
    relation = ET.parse(route_dir / "test.turn.xml").getroot().find("./interval/edgeRelation")
    assert relation.get("from") == "positive_link"
    assert relation.get("to") == "-42"
    routes = ET.parse(route_file).getroot().findall("./vehicle/route")
    assert len(routes) == 3
    assert all(route.get("edges") == "positive_link -42" for route in routes)
