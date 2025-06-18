##############################################################################
# Copyright (c) 2024, Oak Ridge National Laboratory                          #
# All rights reserved.                                                       #
#                                                                            #
# This file is part of RealTwin and is distributed under a GPL               #
# license. For the licensing terms see the LICENSE file in the top-level     #
# directory.                                                                 #
#                                                                            #
# Contributors: ORNL Real-Twin Team                                          #
# Contact: realtwin@ornl.gov                                                 #
##############################################################################
import xml.etree.ElementTree as ET
from xml.dom.minidom import parseString
import shutil
import os
from pathlib import Path
import pyufunc as pf
import copy

from realtwin.func_lib._f_calibration.algo_sumo.cali_turn_inflow import TurnInflowCali
from realtwin.func_lib._f_calibration.algo_sumo.cali_behavior import BehaviorCali
from realtwin.func_lib._f_calibration.algo_sumo.util_cali_turn_inflow import (read_MatchupTable,
                                                                              generate_turn_demand_cali,
                                                                              generate_inflow,
                                                                              generate_turn_summary)
from realtwin.func_lib._f_calibration.algo_sumo.util_cali_behavior import auto_select_two_routes


# for the beta version
def cali_sumo(*, sel_algo: dict = None, input_config: dict = None, verbose: bool = True, **kwargs) -> bool:
    """Run SUMO calibration based on the selected algorithm and input configuration.

    Args:
        sel_algo (dict): the dictionary of selected algorithm for turn_inflow and behavior. Defaults to None.
        input_config (dict): the dictionary contain configurations from input yaml file. Defaults to None.
        verbose (bool): print out processing message. Defaults to True.

    Raises:
        ValueError: if algo_config is not a dict with two levels with keys of 'ga', 'sa', and 'ts'
        ValueError: if sel_algo is not a dict with keys of 'turn_inflow' and 'behavior'

    Returns:
        bool: True if calibration is successful, False otherwise.
    """

    # Test-driven Development: check selected algorithm from input
    if sel_algo is None:  # use default algorithm if not provided
        sel_algo = {"turn_inflow": "ga", "behavior": "ga"}

    if not isinstance(sel_algo, dict):
        print("  :Error:parameter sel_algo must be a dict with"
              " keys of 'turn_inflow' and 'behavior', using"
              " genetic algorithm as default values.")
        sel_algo = {"turn_inflow": "ga", "behavior": "ga"}

    # Prepare scenario_config and algo_config from input_config
    scenario_config_turn_inflow = prepare_scenario_config_turn_inflow(input_config)

    # Prepare Algorithm configure: e.g. {"ga": {}, "sa": {}, "ts": {}}
    algo_config_turn_inflow = input_config["Calibration"]["turn_inflow"]
    algo_config_turn_inflow["ga_config"] = input_config["Calibration"]["ga_config"]
    algo_config_turn_inflow["sa_config"] = input_config["Calibration"]["sa_config"]
    algo_config_turn_inflow["ts_config"] = input_config["Calibration"]["ts_config"]

    if "update_turn_flow_algo" in kwargs:
        algo_config_turn_inflow["ga_config"] = algo_config_turn_inflow["ga_config"].update(
            kwargs["update_turn_flow_algo"].get("ga_config", {}))
        algo_config_turn_inflow["sa_config"] = algo_config_turn_inflow["sa_config"].update(
            kwargs["update_turn_flow_algo"].get("sa_config", {}))
        algo_config_turn_inflow["ts_config"] = algo_config_turn_inflow["ts_config"].update(
            kwargs["update_turn_flow_algo"].get("ts_config", {}))

    algo_config_behavior = input_config["Calibration"]["behavior"]
    algo_config_behavior["ga_config"] = input_config["Calibration"]["ga_config"]
    algo_config_behavior["sa_config"] = input_config["Calibration"]["sa_config"]
    algo_config_behavior["ts_config"] = input_config["Calibration"]["ts_config"]

    if "update_behavior_algo" in kwargs:
        algo_config_behavior["ga_config"] = algo_config_behavior["ga_config"].update(
            kwargs["update_behavior_algo"].get("ga_config", {}))
        algo_config_behavior["sa_config"] = algo_config_behavior["sa_config"].update(
            kwargs["update_behavior_algo"].get("sa_config", {}))
        algo_config_behavior["ts_config"] = algo_config_behavior["ts_config"].update(
            kwargs["update_behavior_algo"].get("ts_config", {}))

    # run calibration based on the selected algorithm: optimize turn and inflow
    print("\n  :Optimize Turn and Inflow...")
    turn_inflow = TurnInflowCali(scenario_config_turn_inflow, algo_config_turn_inflow, verbose=verbose)

    match sel_algo["turn_inflow"]:
        case "ga":
            g_best, model = turn_inflow.run_GA()
            path_model_result = "turn_inflow_ga_result"
        case "sa":
            g_best, model = turn_inflow.run_SA()
            path_model_result = "turn_inflow_sa_result"
        case "ts":
            g_best, model = turn_inflow.run_TS()
            path_model_result = "turn_inflow_ts_result"
        case _:
            print(f"  :Error: unsupported algorithm {sel_algo['turn_inflow']}, using genetic algorithm as default.")
            g_best, model = turn_inflow.run_GA()
            path_model_result = "turn_inflow_ga_result"

    turn_inflow.run_vis(path_model_result, model)
    # clean up the temporary files generated during turn and inflow calibration
    turn_inflow._clean_up()

    print("\n  :Optimize Behavior parameters based on the optimized turn and inflow...")
    scenario_config_behavior = prepare_scenario_config_behavior(input_config)
    if "sel_behavior_routes" in kwargs:
        scenario_config_behavior["sel_behavior_routes"] = kwargs["sel_behavior_routes"]
    else:
        # automatically select two routes from network
        dir_behavior = scenario_config_behavior["dir_behavior"]
        network_name = input_config.get("Network").get("NetworkName")
        path_net = Path(dir_behavior) / f"{network_name}.net.xml"
        path_route = Path(dir_behavior) / f"{network_name}.rou.xml"
        path_report = Path(dir_behavior) / "selected_routes_travel_time_map.html"
        google_api = ""
        routes_list, time_list, edge_list = auto_select_two_routes(path_route, path_net,
                                                                   api_key=google_api, path_report=path_report)
        print(f"  : selected routes: {routes_list}")
        print(f"  : selected travel time: {time_list}")
        print(f"  : selected edge list: {edge_list}")
        sel_route_dict = {}
        route_id = 1
        for route_name, travel_time, edge_id_list in zip(routes_list, time_list, edge_list):
            sel_route_dict[f"route_{route_id}"] = {"time": travel_time,
                                                   "edge_list": edge_id_list,
                                                   "route_list": route_name}
            route_id += 1
        scenario_config_behavior["sel_behavior_routes"] = sel_route_dict
    print(f"  \n:Selected behavior routes: {scenario_config_behavior['sel_behavior_routes']}\n")
    behavior = BehaviorCali(scenario_config_behavior, algo_config_behavior, verbose=verbose)

    match sel_algo["behavior"]:
        case "ga":
            g_best, model = behavior.run_GA()
            path_model_result = "behavior_ga_result"
        case "sa":
            g_best, model = behavior.run_SA()
            path_model_result = "behavior_sa_result"
        case "ts":
            g_best, model = behavior.run_TS()
            path_model_result = "behavior_ts_result"
        case _:
            print(f"  :Error: unsupported algorithm {sel_algo['behavior']}, using genetic algorithm as default.")
            g_best, model = behavior.run_GA()
            path_model_result = "behavior_ga_result"

    behavior.run_vis(path_model_result, model)
    return True


def prepare_scenario_config_turn_inflow(input_config: dict) -> dict:
    """Prepare scenario_config from input_config"""

    scenario_config_dict = input_config.get("Calibration").get("scenario_config")

    # # add input_dir to scenario_config from generated SUMO dir(scenario generation)
    generated_sumo_dir = pf.path2linux(Path(input_config["output_dir"]) / "SUMO")
    # generated_sumo_dir = pf.path2linux(Path(__file__).parents[3] / "datasets/input_dir_dummy/")
    # print(f"  :use dummy input: {generated_sumo_dir} for calibration in beta version")

    # create turn_inflow directory under generated_sumo_dir
    turn_inflow_dir = pf.path2linux(Path(generated_sumo_dir) / "turn_inflow")
    os.makedirs(turn_inflow_dir, exist_ok=True)
    turn_inflow_route_dir = pf.path2linux(Path(generated_sumo_dir) / "turn_inflow" / "route")
    os.makedirs(turn_inflow_route_dir, exist_ok=True)

    # add input_dir as turn_inflow_dir
    scenario_config_dict["dir_turn_inflow"] = turn_inflow_dir

    # copy net.xml to turn_inflow directory
    network_name = input_config.get("Network").get("NetworkName")
    path_net_sumo = pf.path2linux(Path(generated_sumo_dir) / f"{network_name}.net.xml")
    shutil.copy(path_net_sumo, turn_inflow_dir)
    # shutil.copy(path_net_sumo, turn_inflow_route_dir)

    # create Edge.add.xml in turn_inflow directory
    path_edge_add = pf.path2linux(Path(turn_inflow_dir) / "Edge.add.xml")
    generate_edge_add_xml(path_edge_add)

    # create .cfg file in turn_inflow directory
    path_sumocfg = pf.path2linux(Path(turn_inflow_dir) / f"{network_name}.sumocfg")
    seed = scenario_config_dict.get("calibration_seed", 812)
    sim_start_time = scenario_config_dict.get("sim_start_time", 3600 * 8)
    sim_end_time = scenario_config_dict.get("sim_end_time", 3600 * 10)
    calibration_time_step = scenario_config_dict.get("calibration_time_step", 1)
    generate_sumocfg_xml(path_sumocfg, network_name, seed, sim_start_time, sim_end_time, calibration_time_step)

    # create turn and inflow and summary df
    path_matchup_table = pf.path2linux(Path(input_config["input_dir"]) / "MatchupTable.xlsx")
    traffic_dir = pf.path2linux(Path(input_config["input_dir"]) / "Traffic")
    path_net_turn_inflow = pf.path2linux(Path(turn_inflow_dir) / f"{network_name}.net.xml")
    MatchupTable_UserInput = read_MatchupTable(path_matchup_table=path_matchup_table)
    TurnDf, IDRef = generate_turn_demand_cali(path_matchup_table=path_matchup_table, traffic_dir=traffic_dir)

    InflowDf_Calibration, InflowEdgeToCalibrate, N_InflowVariable = generate_inflow(path_net_turn_inflow,
                                                                                    MatchupTable_UserInput,
                                                                                    TurnDf,
                                                                                    IDRef)

    (TurnToCalibrate, TurnDf_Calibration,
     RealSummary_Calibration,
     N_Variable, N_TurnVariable) = generate_turn_summary(TurnDf,
                                                         MatchupTable_UserInput,
                                                         N_InflowVariable)

    scenario_config_dict["TurnToCalibrate"] = TurnToCalibrate
    scenario_config_dict["TurnDf_Calibration"] = TurnDf_Calibration
    scenario_config_dict["InflowDf_Calibration"] = InflowDf_Calibration
    scenario_config_dict["InflowEdgeToCalibrate"] = InflowEdgeToCalibrate
    scenario_config_dict["RealSummary_Calibration"] = RealSummary_Calibration
    scenario_config_dict["N_InflowVariable"] = N_InflowVariable
    scenario_config_dict["N_Variable"] = N_Variable
    scenario_config_dict["N_TurnVariable"] = N_TurnVariable

    # add network name to scenario_config and sim_name
    scenario_config_dict["network_name"] = network_name
    scenario_config_dict["sim_name"] = f"{network_name}.sumocfg"
    scenario_config_dict["path_net"] = pf.path2linux(Path(turn_inflow_dir) / f"{network_name}.net.xml")

    return scenario_config_dict


def prepare_scenario_config_behavior(input_config: dict) -> dict:

    scenario_config_behavior = input_config.get("Calibration").get("scenario_config")
    network_name = input_config.get("Network").get("NetworkName")

    # # add input_dir to scenario_config from generated SUMO dir(scenario generation)
    generated_sumo_dir = pf.path2linux(Path(input_config["output_dir"]) / "SUMO")

    # create behavior directory under generated_sumo_dir
    behavior_dir = pf.path2linux(Path(generated_sumo_dir) / "behavior")
    os.makedirs(behavior_dir, exist_ok=True)
    scenario_config_behavior["dir_behavior"] = behavior_dir

    # copy files from turn_inflow directory to behavior directory
    turn_inflow_dir = pf.path2linux(Path(generated_sumo_dir) / "turn_inflow")
    file_sim = Path(turn_inflow_dir) / f"{network_name}.sumocfg"
    file_net = Path(turn_inflow_dir) / f"{network_name}.net.xml"
    file_edge_add = Path(turn_inflow_dir) / "Edge.add.xml"
    file_route = Path(turn_inflow_dir) / f"{network_name}.rou.xml"
    file_turn = Path(turn_inflow_dir) / f"{network_name}.turn.xml"
    file_inflow = Path(turn_inflow_dir) / f"{network_name}.flow.xml"
    file_EdgeData = Path(turn_inflow_dir) / "EdgeData.xml"
    shutil.copy(file_sim, behavior_dir)
    shutil.copy(file_net, behavior_dir)
    shutil.copy(file_edge_add, behavior_dir)
    shutil.copy(file_route, behavior_dir)
    shutil.copy(file_turn, behavior_dir)
    shutil.copy(file_inflow, behavior_dir)
    shutil.copy(file_EdgeData, behavior_dir)

    scenario_config_behavior["path_turn"] = f"{network_name}.turn.xml"
    scenario_config_behavior["path_inflow"] = f"{network_name}.flow.xml"
    return scenario_config_behavior


def generate_edge_add_xml(path_edge_add: str) -> bool:
    """Generate Edge.add.xml file in the input directory.

    Args:
        input_config (dict): the dictionary contain configurations from input yaml file.
    """
    # create Edge.add.xml in turn_inflow directory
    additional = ET.Element("additional")
    edgeData = ET.SubElement(additional, "edgeData")
    edgeData.set("id", "1")
    edgeData.set("file", "EdgeData.xml")
    tree = ET.ElementTree(additional)
    tree.write(path_edge_add, encoding="utf-8", xml_declaration=True)
    return True


def generate_sumocfg_xml(path_sumocfg: str, network_name: str, seed: int,
                         sim_start_time: int, sim_end_time: int, calibration_time_step: int) -> bool:
    # create .cfg file in turn_inflow directory
    # Create XML root
    root = ET.Element('configuration')
    root.set('xmlns:xsi', 'http://www.w3.org/2001/XMLSchema-instance')
    root.set('xsi:noNamespaceSchemaLocation', 'http://sumo.dlr.de/xsd/duarouterConfiguration.xsd')

    # Random seed
    random = ET.SubElement(root, 'random')
    ET.SubElement(random, 'seed', {'value': str(seed)})

    # Input files
    input_elem = ET.SubElement(root, 'input')
    ET.SubElement(input_elem, 'net-file', {'value': f'{network_name}.net.xml'})
    ET.SubElement(input_elem, 'route-files', {'value': f'{network_name}.rou.xml'})
    ET.SubElement(input_elem, 'additional-files', {'value': 'Edge.add.xml'})

    # Output (empty section placeholder)
    ET.SubElement(root, 'output')

    # Time setup
    time = ET.SubElement(root, 'time')
    ET.SubElement(time, 'begin', {'value': str(sim_start_time)})
    ET.SubElement(time, 'end', {'value': str(sim_end_time)})
    ET.SubElement(time, 'step-length', {'value': str(calibration_time_step)})

    # GUI options
    gui_only = ET.SubElement(root, 'gui_only')
    ET.SubElement(gui_only, 'start', {'value': 't'})

    # Report options
    report = ET.SubElement(root, 'report')
    ET.SubElement(report, 'no-warnings', {'value': 'true'})
    ET.SubElement(report, 'no-step-log', {'value': 'true'})

    # Pretty print
    rough_string = ET.tostring(root, 'utf-8')
    reparsed = parseString(rough_string)
    xml_string = reparsed.toprettyxml(indent="    ")

    # Write to file
    with open(path_sumocfg, 'w') as file:
        file.write(xml_string)
    return True
