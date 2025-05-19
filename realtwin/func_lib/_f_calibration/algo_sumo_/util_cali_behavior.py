'''
##############################################################
# Created Date: Tuesday, February 25th 2025
# Contact Info: luoxiangyong01@gmail.com
# Author/Copyright: Mr. Xiangyong Luo
##############################################################
'''
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
import subprocess
import random
import pandas as pd
import numpy as np
import pyufunc as pf

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
    sys.path = list(set(sys.path))  # remove duplicates
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")
import traci


def update_turn_flow_from_solution(path_turn: str,
                                   path_inflow: str,
                                   initial_solution: np.array,
                                   cali_interval: int,
                                   demand_interval: int) -> tuple:
    """assign the new turn ratios and inflow counts to the given dataframes

    Args:
        df_turn (pd.DataFrame): the turn dataframe from turn.xlsx
        df_inflow (pd.DataFrame): the inflow dataframe from inflow.xlsx
        initial_solution (np.array): the initial solution from the genetic algorithm
        cali_interval (int): the calibration interval
        demand_interval (int): the demand interval

    Returns:
        tuple: the updated turn and inflow dataframes
    """

    # Much improved in terms of speed and readability. 1 min for 2 - 5 size and generations

    # create the copy of turn and inflow dataframes for internal operations
    TurnDf = pd.read_excel(path_turn)
    InflowDf = pd.read_excel(path_inflow)

    # --- Update TurnRatios ---
    # Instead of many .loc calls, set a MultiIndex on the two key columns.
    TurnDf.set_index(['OpenDriveFromID', 'OpenDriveToID'], inplace=True)
    TurnDf.sort_index(inplace=True)

    # Create a mapping from (from, to) to the new TurnRatio.
    # Notice that for each pair one of the assignments is the value,
    # and the other is 1 minus that value.
    turn_mapping = {
        # Between Amin Dr. and  I-75 SB Off Ramp
        (290, 298): initial_solution[0],
        (290, 299): 1 - initial_solution[0],
        (331, 297): initial_solution[1],
        (331, 298): 1 - initial_solution[1],
        (293, 299): initial_solution[2],
        (293, 297): 1 - initial_solution[2],

        # Between Napier Rd. and Lifestyle Way1
        (315, 321): initial_solution[3],
        (315, 323): 1 - initial_solution[3],
        (320, 3221): initial_solution[4],
        (320, 321): 1 - initial_solution[4],
        (302, 323): initial_solution[5],
        (302, 3221): 1 - initial_solution[5],

        # Between Napier Rd. and Lifestyle Way2
        (281, 315): initial_solution[6],
        (281, 314): 1 - initial_solution[6],
        (316, 313): initial_solution[7],
        (316, 315): 1 - initial_solution[7],
        (322, 314): initial_solution[8],
        (322, 313): 1 - initial_solution[8],

        # Between  Lifestyle Way and Gunbarrel Road
        (330, 327): initial_solution[9],
        (330, 328): 1 - initial_solution[9],
        (307, 329): initial_solution[10],
        (307, 327): 1 - initial_solution[10],
        (284, 328): initial_solution[11],
        (284, 329): 1 - initial_solution[11],
    }

    # Loop over the (small) mapping dictionary.
    # Because the DataFrame index is a MultiIndex, each lookup is fast.
    for key, value in turn_mapping.items():
        if key in TurnDf.index:
            TurnDf.loc[key, 'TurnRatio'] = value

    # Reset the index so that the returned DataFrame has the original format.
    TurnDf.reset_index(inplace=True)

    # --- Update Inflow Counts ---
    # First ensure that Count is a float.
    InflowDf['Count'] = InflowDf['Count'].astype(float)

    # For inflows, the key is just OpenDriveFromID. Set it as the index.
    InflowDf.set_index('OpenDriveFromID', inplace=True)

    #  considering the calibration interval and demand interval
    inflow_mapping = {
        331: initial_solution[12] / cali_interval * demand_interval,
        320: initial_solution[13] / cali_interval * demand_interval,
        316: initial_solution[14] / cali_interval * demand_interval,
        330: initial_solution[15] / cali_interval * demand_interval,
    }

    # Loop over the inflow mapping and assign new values.
    for key, value in inflow_mapping.items():
        if key in InflowDf.index:
            InflowDf.loc[key, 'Count'] = value

    InflowDf.reset_index(inplace=True)
    return (TurnDf, InflowDf)


def create_rou_turn_flow_xml(network_name: str, sim_start_time: float, sim_end_time: float,
                             df_turn: pd.DataFrame,
                             df_inflow: pd.DataFrame,
                             input_dir: str) -> bool:
    """Using SUMO jtrrouter to generate the demand file for the given network

    Args:
        network_name (str): the name of the network,
        sim_start_time (float): start time of the simulation
        sim_end_time (float): end time of the simulation
        df_turn (pd.DataFrame): the turn dataframe
        df_inflow (pd.DataFrame): the inflow dataframe
        ical (str): the iteration number for the calibration
        input_dir (str): the path to the input directory
        output_dir (str): the path to the output directory
        remove_old_files (bool): whether to remove temporary files in time. Defaults to True.

    Returns:
        bool: True if the demand file is generated successfully
    """

    # Process the turn dataframe
    TurnDf = df_turn.copy()
    TurnDf['IntervalStart'] = TurnDf['IntervalStart'].astype(float)
    TurnDf['IntervalEnd'] = TurnDf['IntervalEnd'].astype(float)

    # Filter for simulation time only once
    mask = (TurnDf['IntervalStart'] >= sim_start_time) & (
        TurnDf['IntervalEnd'] <= sim_end_time)
    TurnDf = TurnDf.loc[mask]

    # Build the XML for turns
    turns = ET.Element('turns')

    # Group by interval so we don't re-filter on each iteration
    for (start, end), group in TurnDf.groupby(['IntervalStart', 'IntervalEnd']):
        # Create an interval element with begin and end attributes
        interval_el = ET.SubElement(
            turns, 'interval', begin=str(start), end=str(end))

        # Iterate over rows quickly using itertuples
        for row in group.itertuples(index=False):
            ET.SubElement(
                interval_el,
                'edgeRelation',
                **{'from': str(-int(row.OpenDriveFromID)),
                   'to': str(-int(row.OpenDriveToID)),
                   'probability': str(row.TurnRatio)})

    # Write the turn XML file - the hard coded path and will be deleted after the calibration
    turn_xml_path = f'{pf.path2linux(input_dir)}/{network_name}.turn.xml'
    ET.ElementTree(turns).write(
        turn_xml_path, encoding='utf-8', xml_declaration=True)

    # Process the inflow dataframe
    InflowDf = df_inflow.copy()
    InflowDf['IntervalStart'] = InflowDf['IntervalStart'].astype(float)
    InflowDf['IntervalEnd'] = InflowDf['IntervalEnd'].astype(float)
    mask = (InflowDf['IntervalStart'] >= sim_start_time) & (
        InflowDf['IntervalEnd'] <= sim_end_time)
    InflowDf = InflowDf.loc[mask]

    routes = ET.Element('routes')
    ET.SubElement(routes, 'vType', id='car', type='passenger')

    # Enumerate flows (starting at 1)
    for flow_id, row in enumerate(InflowDf.itertuples(index=False), start=1):
        ET.SubElement(
            routes,
            'flow',
            id=str(flow_id),
            begin=str(row.IntervalStart),
            end=str(row.IntervalEnd),
            **{'from': str(-int(row.OpenDriveFromID))},
            number=str(int(row.Count)),
            type='car'
        )

    # Write the inflow XML file
    inflow_xml_path = f'{pf.path2linux(input_dir)}/{network_name}.flow.xml'
    ET.ElementTree(routes).write(inflow_xml_path,
                                 encoding='utf-8', xml_declaration=True)

    # Generate the route file using jtrrouter
    # Build the command (adjust if necessary for your platform)
    path_net = pf.path2linux(os.path.join(input_dir, f"{network_name}.net.xml"))
    path_rou = pf.path2linux(os.path.join(input_dir, f"{network_name}.rou.xml"))

    cmd = (
        f'cmd /c "jtrrouter -r {inflow_xml_path} -t {turn_xml_path} '
        f'-n {path_net} --accept-all-destinations '
        f'--remove-loops True --randomize-flows -o {path_rou}"'
    )
    process = subprocess.Popen(cmd, shell=True)
    process.wait()

    return True


def run_SUMO_create_EdgeData(sim_name: str, sim_end_time: float) -> bool:
    """run SUMO simulation using traci module

    Args:
        sim_name (str): the name of the simulation, it should be the .sumocfg file
        sim_end_time (float): the end time of the simulation

    Returns:
        bool: True if the simulation is successful
    """

    traci.start(["sumo", "-c", sim_name])
    while traci.simulation.getTime() < sim_end_time:
        traci.simulationStep()
    traci.close()
    return True


def get_travel_time_from_EdgeData_xml(path_EdgeData: str, edge_ids: list) -> float:
    """
    Calculate total travel time along a route composed of multiple edges.

    Parameters:
    edge_output_file (str): Path to the edge output file generated by SUMO.
    edge_ids (list): List of edge IDs that make up the route.

    Returns:
    float: Total travel time along the route.
    """
    total_travel_time = 0.0
    tree = ET.parse(path_EdgeData)
    root = tree.getroot()
    # print (root.attrib)
    for edge_id in edge_ids:
        p = root.findall('interval')
        for parent in p:
            for child in parent:
                if child.get('id') == edge_id:
                    travel_time = child.get('traveltime')
                    if travel_time is not None:
                        total_travel_time += float(travel_time)
    return total_travel_time


def update_flow_xml_from_solution(path_flow: str, solution: list | np.ndarray) -> bool:
    """Update the flow XML file with new car-following parameters."""

    min_gap, accel, decel, sigma, tau, emergencyDecel = solution

    # Load the XML file
    tree = ET.parse(path_flow)
    root = tree.getroot()

    # Find the tag
    parent = root.find('vType')

    if parent is not None:
        # print(parent.tag, parent.attrib)  # Prints child tag name and text
        parent.set('minGap', str(min_gap))
        parent.set('accel', str(accel))  # Add a new attribute
        parent.set('decel', str(decel))
        parent.set('sigma', str(sigma))
        parent.set('tau', str(tau))
        parent.set('emergencyDecel', str(emergencyDecel))
    else:
        print("Parent tag not found")
    tree.write(path_flow)
    return True


def run_jtrrouter_to_create_rou_xml(network_name: str, path_net: str, path_flow: str, path_turn: str, path_rou: str, verbose: bool = False) -> None:
    """Runs jtrrouter to generate a route file from flow and network files in SUMO.

    Args:
        network_name (str): The name of the network.
        path_net (str): The path to the network file.
        path_flow (str): The path to the flow file.
        path_turn (str): The path to the turn file.
        path_rou (str): The path to the output route file.
        verbose (bool): If True, print additional information. Defaults to False.
    """

    # Define the jtrrouter command with all necessary arguments
    cmd = [
        "jtrrouter",
        "-n", path_net,
        "-r", path_flow,
        "-t", path_turn,
        "-o", path_rou,
        "--accept-all-destinations",
        "--remove-loops True",
        # "--seed","101",
        "--ignore-errors",  # Continue on errors; remove if not desired
    ]

    # Execute the command
    try:
        subprocess.run(cmd, capture_output=True, text=True)
        if verbose:
            print(f"  :Route file generated successfully: {path_rou}")
    except subprocess.CalledProcessError as e:
        print(f"  :An error occurred while running jtrrouter: {e}")


def result_analysis_on_EdgeData(path_summary: str,
                                path_EdgeData: str,
                                calibration_target: dict,
                                sim_start_time: float,
                                sim_end_time: float) -> tuple:
    """Analyze the result of the simulation and return the flag, mean GEH, and GEH percent

    Args:
        path_summary (str or pd.DataFrame): the summary dataframe from summary.xlsx in input dir
        path_EdgeData (str): the path to the EdgeData.xml file in the input dir
        calibration_target (dict): the calibration target from the scenario config, it should contain GEH and GEHPercent
        sim_start_time (float): the start time of the simulation
        sim_end_time (float): the end time of the simulation

    Returns:
        tuple: (flag, mean GEH, geh percent)
    """
    # Load and parse the new XML file
    # mapping of sumo id with GridSmart Intersection from user input

    # 1. Filter and group the summary data
    df_summary = pd.read_excel(path_summary)
    df_filtered = df_summary.loc[df_summary["realcount"].notna()].copy()
    approach_summary = df_filtered.groupby(['IntersectionName',
                                            'entrance_sumo',
                                            'Bound'], as_index=False)['realcount'].sum()

    # 2. Parse the XML file using a list comprehension with immediate type conversion
    tree = ET.parse(path_EdgeData)
    root = tree.getroot()
    edge_data = [
        {
            'id': int(edge.get('id')) if edge.get('id') else None,
            'travel_time': float(edge.get('traveltime')) if edge.get('traveltime') else None,
            'arrived': int(edge.get('arrived')) if edge.get('arrived') else None,
            'departed': int(edge.get('departed')) if edge.get('departed') else None,
            'left': int(edge.get('left')) if edge.get('left') else None,
            'density': float(edge.get('density')) if edge.get('density') else None,
            'speed': float(edge.get('speed')) if edge.get('speed') else None
        }
        for interval in root.findall('.//interval')
        for edge in interval.findall('edge')
    ]
    edge_df = pd.DataFrame(edge_data)

    # 3. Ensure matching key types and merge data
    approach_summary['entrance_sumo'] = approach_summary['entrance_sumo'].astype(int)
    edge_df['id'] = edge_df['id'].astype(int)
    merged = approach_summary.merge(edge_df,
                                    left_on='entrance_sumo',
                                    right_on='id',
                                    how='inner')
    merged.rename(columns={'left': 'count'}, inplace=True)
    merged.drop(columns=['id'], inplace=True)

    # 4. Calculate flows (vehicles per hour)
    duration = sim_end_time - sim_start_time
    merged['flow'] = merged['count'] / duration * 3600
    merged['realflow'] = merged['realcount'] / duration * 3600

    # 5. Compute GEH and summary statistics
    merged['GEH'] = np.sqrt(2
                            * ((merged['count'] - merged['realcount']) ** 2)
                            / (merged['count'] + merged['realcount']))
    mean_geh = merged['GEH'].mean()
    geh_percent = (merged['GEH'] < calibration_target['GEH']).mean()

    flag = 1
    if geh_percent < calibration_target['GEHPercent']:
        flag = 0

    # 6. Compute absolute differences and relative differences once
    diff_abs = (merged['realflow'] - merged['flow']).abs()
    relative_diff = (
        (merged['realflow'] - merged['flow']) / merged['realflow']).abs()

    # 7. Vectorized condition checks for different volume ranges

    # within 100 vph for volumes < 700
    cond_low = (merged['realflow'] < 700) & (diff_abs > 100)

    # within 15%  for volumes  700-2700
    cond_mid = (merged['realflow'].between(700, 2700)) & (relative_diff > 0.15)

    # within 400 vph for volumes > 2700
    cond_high = (merged['realflow'] > 2700) & (diff_abs > 400)

    # If any of the conditions are met, set flag to 0
    if cond_low.any() or cond_mid.any() or cond_high.any():
        flag = 0

    return (flag, mean_geh, geh_percent)


def fitness_func(solution: list | np.ndarray, scenario_config: dict = None, error_func: str = "rmse") -> float:
    """ Evaluate the fitness of a given solution for SUMO calibration."""
    # print(f"  :solution: {solution}")
    # Set up SUMO command with car-following parameters
    if error_func not in ["rmse", "mae"]:
        raise ValueError("error_func must be either 'rmse' or 'mae'")

    if solution[5] >= 9.3:  # emergencyDecel
        solution[5] = 9.3
    if solution[5] < solution[2]:  # emergencyDecel < deceleration
        solution[5] = solution[2] + random.randrange(1, 5)
    # print("after emergencydecel update", solution)

    # get path from scenario_config
    network_name = scenario_config.get("network_name")
    sim_input_dir = Path(scenario_config.get("input_dir"))
    path_net = pf.path2linux(sim_input_dir / f"{network_name}.net.xml")
    path_flow = pf.path2linux(sim_input_dir / f"{network_name}.flow.xml")
    path_turn = pf.path2linux(sim_input_dir / f"{network_name}.turn.xml")
    path_rou = pf.path2linux(sim_input_dir / f"{network_name}.rou.xml")
    path_EdgeData = pf.path2linux(sim_input_dir / "EdgeData.xml")
    EB_tt = scenario_config.get("EB_tt")
    WB_tt = scenario_config.get("WB_tt")
    EB_edge_list = scenario_config.get("EB_edge_list")
    WB_edge_list = scenario_config.get("WB_edge_list")

    sim_name = scenario_config.get("sim_name")
    sim_end_time = scenario_config.get("sim_end_time")

    update_flow_xml_from_solution(path_flow, solution)

    run_jtrrouter_to_create_rou_xml(network_name, path_net, path_flow, path_turn, path_rou)

    # change the working directory to the input directory for SUMO
    os.chdir(sim_input_dir)
    # Define the command to run SUMO
    sumo_command = f"sumo -c \"{sim_name}\""
    sumoProcess = subprocess.Popen(sumo_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    sumoProcess.wait()

    # Read output file or TraCI to evaluate the fitness
    # Example: calculate average travel time, lower is better
    # Logic to read and calculate travel time from SUMO output
    travel_time_EB = get_travel_time_from_EdgeData_xml(path_EdgeData, EB_edge_list)
    travel_time_WB = get_travel_time_from_EdgeData_xml(path_EdgeData, WB_edge_list)

    if error_func == "rmse":
        fitness_err = np.sqrt(0.5 * ((EB_tt - travel_time_EB)**2 + (WB_tt - travel_time_WB)**2))
    elif error_func == "mae":
        fitness_err = ((abs(EB_tt - travel_time_EB) + abs(WB_tt - travel_time_WB)) / 2)
    else:
        raise ValueError("error_func must be either 'rmse' or 'mae'")

    # Calculate GEH from updated results
    path_summary = pf.path2linux(sim_input_dir / "summary.xlsx")
    calibration_target = scenario_config.get("calibration_target")
    sim_start_time = scenario_config.get("sim_start_time")
    sim_end_time = scenario_config.get("sim_end_time")
    _, mean_geh, geh_percent = result_analysis_on_EdgeData(path_summary,
                                                           path_EdgeData,
                                                           calibration_target,
                                                           sim_start_time,
                                                           sim_end_time)
    print(f"  :GEH: Mean Percentage: {mean_geh:.6f}, {geh_percent:.6f}, Travel time error: {fitness_err:.6f}")

    return fitness_err
