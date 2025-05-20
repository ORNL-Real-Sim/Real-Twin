'''
##############################################################
# Created Date: Thursday, March 13th 2025
# Contact Info: luoxiangyong01@gmail.com
# Author/Copyright: Mr. Xiangyong Luo
##############################################################
'''


'''
##############################################################
# Created Date: Wednesday, February 26th 2025
# Contact Info: luoxiangyong01@gmail.com
# Author/Copyright: Mr. Xiangyong Luo
##############################################################
'''
import xml.etree.ElementTree as ET
from xml.dom.minidom import parseString
import shutil
import os
from pathlib import Path
import pyufunc as pf
import copy

from realtwin.func_lib._f_calibration.algo_sumo_.calib_turn_inflow import TurnInflowCalib
from realtwin.func_lib._f_calibration.algo_sumo_.calib_behavior import BehaviorCalib
from realtwin.func_lib._f_calibration.algo_sumo_.util_cali_turn_inflow import (read_MatchupTable,
                                                                               generate_turn_demand_cali,
                                                                               generate_inflow,
                                                                               generate_turn_summary)


# for the beta version
def cali_sumo(*, sel_algo: dict = None, input_config: dict = None, verbose: bool = True) -> bool:
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
    scenario_config_turn_inflow = prepare_scenario_config(input_config)

    # Prepare Algorithm configure: e.g. {"ga": {}, "sa": {}, "ts": {}}
    algo_config_turn_inflow = input_config["Calibration"]["turn_inflow"]
    algo_config_turn_inflow["ga_config"] = input_config["Calibration"]["ga_config"]
    algo_config_turn_inflow["sa_config"] = input_config["Calibration"]["sa_config"]
    algo_config_turn_inflow["ts_config"] = input_config["Calibration"]["ts_config"]

    algo_config_behavior = input_config["Calibration"]["behavior"]
    algo_config_behavior["ga_config"] = input_config["Calibration"]["ga_config"]
    algo_config_behavior["sa_config"] = input_config["Calibration"]["sa_config"]
    algo_config_behavior["ts_config"] = input_config["Calibration"]["ts_config"]

    # run calibration based on the selected algorithm: optimize turn and inflow
    print("\n  :Optimize Turn and Inflow...")
    turn_inflow = TurnInflowCalib(scenario_config_turn_inflow, algo_config_turn_inflow, verbose=verbose)

    match sel_algo["turn_inflow"]:
        case "ga":
            g_best, model = turn_inflow.run_GA()
            path_model_result = pf.path2linux(Path(algo_config_turn_inflow["input_dir"]) / "turn_inflow_ga_result")
            # path_model_result = "turn_inflow_ga_result"

        case "sa":
            g_best, model = turn_inflow.run_SA()
            path_model_result = pf.path2linux(Path(algo_config_turn_inflow["input_dir"]) / "turn_inflow_sa_result")
            # path_model_result = "turn_inflow_sa_result"
        case "ts":
            g_best, model = turn_inflow.run_TS()
            path_model_result = pf.path2linux(Path(algo_config_turn_inflow["input_dir"]) / "turn_inflow_ts_result")
            # path_model_result = "turn_inflow_ts_result"
        case _:
            print(f"  :Error: unsupported algorithm {sel_algo['turn_inflow']}, using genetic algorithm as default.")
            g_best, model = turn_inflow.run_GA()
            path_model_result = pf.path2linux(Path(algo_config_turn_inflow["input_dir"]) / "ga_turn_inflow_result")
            # path_model_result = "ga_turn_inflow_result"

    turn_inflow.run_vis(path_model_result, model)

    # run calibration based on the selected algorithm: optimize behavior
    # update path_turn and path_flow to generated xml files
    scenario_config_behavior = copy.deepcopy(scenario_config_turn_inflow)
    scenario_config_behavior["path_turn"] = f"{scenario_config_behavior.get("network_name")}.turn.xml"
    scenario_config_behavior["path_inflow"] = f"{scenario_config_behavior.get("network_name")}.flow.xml"
    scenario_config_behavior["EB_tt"] = algo_config_behavior.get("EB_tt")
    scenario_config_behavior["WB_tt"] = algo_config_behavior.get("WB_tt")
    scenario_config_behavior["EB_edge_list"] = algo_config_behavior.get("EB_edge_list")
    scenario_config_behavior["WB_edge_list"] = algo_config_behavior.get("WB_edge_list")

    print("\n  :Optimize Behavior parameters based on the optimized turn and inflow...")
    behavior = BehaviorCalib(scenario_config_behavior, algo_config_behavior, verbose=verbose)

    match sel_algo["behavior"]:
        case "ga":
            g_best, model = behavior.run_GA()
            path_model_result = pf.path2linux(Path(scenario_config_behavior["input_dir"]) / "behavior_ga_result")
            # path_model_result = "behavior_ga_result"
        case "sa":
            g_best, model = behavior.run_SA()
            path_model_result = pf.path2linux(Path(scenario_config_behavior["input_dir"]) / "behavior_sa_result")
            # path_model_result = "behavior_sa_result"
        case "ts":
            g_best, model = behavior.run_TS()
            path_model_result = pf.path2linux(Path(scenario_config_behavior["input_dir"]) / "behavior_ts_result")
            # path_model_result = "behavior_ts_result"
        case _:
            print(f"  :Error: unsupported algorithm {sel_algo['behavior']}, using genetic algorithm as default.")
            g_best, model = behavior.run_GA()
            path_model_result = pf.path2linux(Path(scenario_config_behavior["input_dir"]) / "ga_behavior_result")
            # path_model_result = "ga_behavior_result"

    behavior.run_vis(path_model_result, model)
    return True


def prepare_scenario_config(input_config: dict) -> dict:
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
    scenario_config_dict["input_dir"] = turn_inflow_dir

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
    seed = scenario_config_dict.get("calibration_seed")
    sim_start_time = scenario_config_dict.get("sim_start_time")
    sim_end_time = scenario_config_dict.get("sim_end_time")
    calibration_time_step = scenario_config_dict.get("calibration_time_step")
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

    # TODO Do not copy generated files to generated_sumo_dir in beta version
#     # check whether required files exist in the input dir
#     required_files = {key: value for key, value in scenario_config_dict.items() if key.startswith("path_")}
#
#     if not pf.check_files_in_dir(required_files.values(), input_config.get("input_dir")):
#         return None
#
#     # copy required files to generated_sumo_dir
#     for key, file in required_files.items():
#         shutil.copy(Path(input_config["input_dir"]) / file, generated_sumo_dir)

    return scenario_config_dict


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
