##############################################################################
# Copyright (c) 2024, Oak Ridge National Laboratory                          #
# All rights reserved.                                                       #
#                                                                            #
# This file is part of RealTwin and is distributed under a __TBD__           #
# license. For the licensing terms see the LICENSE file in the top-level     #
# directory.                                                                 #
#                                                                            #
# Contributors: ORNL Real-Twin Team                                          #
# Contact: realtwin@ornl.gov                                                 #
##############################################################################

import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
import shutil
import subprocess
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


def update_turn_flow_from_solution(initial_solution: np.array,
                                   TurnDf: pd.DataFrame,
                                   TurnToCalibrate: pd.DataFrame,
                                   InflowDf: pd.DataFrame,
                                   InflowEdgeToCalibrate: pd.DataFrame,
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

    i = 0  # index into initial_solution

    # Loop through each group of (JunctionID_OpenDrive, Numbering)
    grouped = TurnToCalibrate.groupby(["JunctionID_OpenDrive", "Numbering"])
    for _, group in grouped:
        assigned_turns = []

        for _, row in group.iterrows():
            x = row["OpenDriveFromID"]
            y = row["OpenDriveToID"]
            if row["Calibration variable?"] == 1:
                # Assign the given initial_solution[i]
                TurnDf.loc[
                    (TurnDf["OpenDriveFromID"] == x) & (TurnDf["OpenDriveToID"] == y),
                    "TurnRatio"
                ] = initial_solution[i]
                assigned_turns.append((x, y, initial_solution[i]))
                i += 1

        # After all Calibration variable?==1 turns in the group, assign 1 - sum to the rest
        for _, row in group.iterrows():
            if row["Calibration variable?"] == 0:
                x = row["OpenDriveFromID"]
                y = row["OpenDriveToID"]
                total_assigned = sum(val for a, b, val in assigned_turns if a == x)
                TurnDf.loc[
                    (TurnDf["OpenDriveFromID"] == x) & (TurnDf["OpenDriveToID"] == y),
                    "TurnRatio"
                ] = 1 - total_assigned

    # Inflow calibration assignments based on InflowEdgeToCalibrate
    for edge in InflowEdgeToCalibrate:
        InflowDf.loc[InflowDf['OpenDriveFromID'] == edge, 'Count'] = initial_solution[i] / cali_interval * demand_interval
        i += 1

    return (TurnDf, InflowDf)


def run_jtrrouter_to_create_rou_xml(network_name: str, path_net: str,
                                    TurnDf: pd.DataFrame, InflowDf: pd.DataFrame,
                                    path_rou: str, sim_start_time: float,
                                    sim_end_time: float, verbose: bool = False) -> None:
    """Runs jtrrouter to generate a route file from flow and network files in SUMO.

    Args:
        network_name (str): The name of the network.
        path_net (str): The path to the network file.
        path_flow (str): The path to the flow file.
        path_turn (str): The path to the turn file.
        path_rou (str): The path to the output route file.
        verbose (bool): If True, print additional information. Defaults to False.
    """

    dir_rou = Path(path_rou).parent
    dir_net = Path(path_net).parent
    path_turn = pf.path2linux(dir_rou / f"{network_name}.turn.xml")
    path_flow = pf.path2linux(dir_rou / f"{network_name}.flow.xml")

    TurnDf['IntervalStart'] = TurnDf['IntervalStart'].astype(float)
    TurnDf['IntervalEnd'] = TurnDf['IntervalEnd'].astype(float)
    TurnDf = TurnDf[(TurnDf['IntervalStart'] >= sim_start_time) & (TurnDf['IntervalEnd'] <= sim_end_time)]
    turns = ET.Element('turns')
    # Create the 'interval' element
    IntervalSet = TurnDf[['IntervalStart', 'IntervalEnd']].drop_duplicates().reset_index(drop=True)
    for _, IntervalData in IntervalSet.iterrows():
        Interval = ET.SubElement(turns, 'interval')
        Interval.set('begin', str(IntervalData['IntervalStart']))
        Interval.set('end', str(IntervalData['IntervalEnd']))

        TurnDfSubset = TurnDf[(TurnDf['IntervalStart'] == IntervalData['IntervalStart']) & (
            TurnDf['IntervalEnd'] == IntervalData['IntervalEnd'])]
        TurnDictSubset = TurnDfSubset.to_dict(orient='records')
        for TurnData in TurnDictSubset:
            edge_relation = ET.SubElement(Interval, 'edgeRelation')
            edge_relation.set('from', str(-int(TurnData['OpenDriveFromID'])))
            edge_relation.set('to', str(-int(TurnData['OpenDriveToID'])))
            edge_relation.set('probability', str(TurnData['TurnRatio']))
    # <edgeRelation from="" probability="" to=""/>
    TreeTurn = ET.ElementTree(turns)
    # Write the XML to the file
    TreeTurn.write(path_turn, encoding='utf-8', xml_declaration=True)

    # Create the .flow.xml
    InflowDf['IntervalStart'] = InflowDf['IntervalStart'].astype(float)
    InflowDf['IntervalEnd'] = InflowDf['IntervalEnd'].astype(float)
    InflowDf = InflowDf[(InflowDf['IntervalStart'] >= sim_start_time) & (InflowDf['IntervalEnd'] <= sim_end_time)]
    routes = ET.Element('routes')
    vtype = ET.SubElement(routes, 'vType')
    vtype.set('id', 'car')
    vtype.set('type', 'passenger')
    InflowDict = InflowDf.to_dict(orient='records')
    FlowID = 0
    for InflowData in InflowDict:
        FlowID += 1
        flow = ET.SubElement(routes, 'flow')
        flow.set('id', str(FlowID))
        flow.set('begin', str(InflowData['IntervalStart']))
        flow.set('end', str(InflowData['IntervalEnd']))
        flow.set('from', str(-int(InflowData['OpenDriveFromID'])))
        flow.set('number', str(int(InflowData['Count'])))
        # flow.set('number', str(int(int(InflowData['Count'])/CablibrationInterval*DemandInterval)))
        flow.set('type', 'car')

    # <flow begin="0.0" end="3600.0" from="" id="" number="" type="car"/>
    TreeInflow = ET.ElementTree(routes)
    # TreeInflow.write('MyNetwork/SUMO/{}.flow.xml'.format(NetworkName), encoding='utf-8', xml_declaration=True)
    TreeInflow.write(path_flow, encoding='utf-8', xml_declaration=True)

    # Define the jtrrouter command with all necessary arguments
    # cmd = [
    #     "jtrrouter",
    #     "-n", path_net,
    #     "-r", path_flow,
    #     "-t", path_turn,
    #     "-o", path_rou,
    #     "--accept-all-destinations",
    #     "--remove-loops True",
    #     "--randomize-flows",
    #     # "--seed","101",
    #     # "--ignore-errors",  # Continue on errors; remove if not desired
    # ]
    cmd = f'cmd /c "jtrrouter -r {path_flow} -t {path_turn} -n {path_net} --accept-all-destinations --remove-loops True --randomize-flows -o {path_rou}"'

    # Execute the command
    try:
        # subprocess.run(cmd, capture_output=True, text=True)
        process = subprocess.Popen(cmd, shell=True)
        process.wait()
        if verbose:
            print(f"  :Route file generated successfully: {path_rou}")

        shutil.copy(path_rou, dir_net)
    except subprocess.CalledProcessError as e:
        print(f"  :An error occurred while running jtrrouter: {e}")


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


def result_analysis_on_EdgeData(Summary_data: pd.DataFrame,
                                path_EdgeData: str,
                                calibration_target: dict,
                                sim_start_time: float,
                                sim_end_time: float) -> tuple:
    """Analyze the result of the simulation and return the flag, mean GEH, and GEH percent

    Args:
        Summary_data (pd.DataFrame): the summary dataframe from summary.xlsx in input dir
        path_EdgeData (str): the path to the EdgeData.xml file in the input dir
        calibration_target (dict): the calibration target from the scenario config, it should contain GEH and GEHPercent
        sim_start_time (float): the start time of the simulation
        sim_end_time (float): the end time of the simulation

    Returns:
        tuple: (flag, mean GEH, geh percent)
    """
    RealSummary = Summary_data[Summary_data["realcount"].notna()]

    ApproachSummary = RealSummary.groupby(['IntersectionName',
                                           'entrance_sumo',
                                           'Bound']).agg({'realcount': 'sum'}).reset_index()
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
    EdgeData = pd.DataFrame(edge_data)

    EdgeData = EdgeData.astype({'id': int,
                                'travel_time': float,
                                'arrived': int,
                                'departed': int,
                                "left": int,
                                'density': float,
                                'speed': float})
    ApproachSummary['entrance_sumo'] = ApproachSummary['entrance_sumo'].astype(int)
    EdgeData['id'] = EdgeData['id'].astype(int)
    ApproachSummary = pd.merge(ApproachSummary, EdgeData, left_on='entrance_sumo', right_on='id')
    ApproachSummary.rename(columns={'left': 'count'}, inplace=True)
    ApproachSummary.drop(columns=['id'], inplace=True)
    ApproachSummary['flow'] = ApproachSummary['count'] / (sim_end_time - sim_start_time) * 3600
    ApproachSummary['realflow'] = ApproachSummary['realcount'] / (sim_end_time - sim_start_time) * 3600

    ApproachSummary['GEH'] = np.sqrt(2 * (ApproachSummary['count'] - ApproachSummary['realcount'])
                                     ** 2 / (ApproachSummary['count'] + ApproachSummary['realcount']))
    MeanGEH = ApproachSummary['GEH'].mean()
    GEHPercent = (ApproachSummary['GEH'] < calibration_target['GEH']).mean()
    flag = 1
    # more than 85% of links have a GEH <= 5
    if GEHPercent < calibration_target['GEHPercent']:
        flag = 0

    BadVolume = ApproachSummary[ApproachSummary['count'] < 0]
    # within 100 vph for volumes < 700
    df1 = ApproachSummary[ApproachSummary['realflow'] < 700]
    if sum((df1['realflow'] - df1['flow']).abs() > 100) > 0:
        flag = 0
        BadVolume = pd.concat([BadVolume, (df1[(df1['realflow'] - df1['flow']).abs() > 100])])

    # within 15%  for volumes  700-2700
    df2 = ApproachSummary[(ApproachSummary['realflow'] >= 700) & (ApproachSummary['realflow'] <= 2700)]
    if sum(((df2['realflow'] - df2['flow']) / -df2['realflow']).abs() > 0.15) > 0:
        flag = 0
        BadVolume = pd.concat([BadVolume, (df2[((df2['realflow'] - df2['flow']) / -df2['realflow']).abs() > 0.15])])

    # within 400 vph for volumes > 2700
    df3 = ApproachSummary[ApproachSummary['realflow'] > 2700]
    if sum((df3['realflow'] - df3['flow']).abs() > 400) > 0:
        flag = 0
        BadVolume = pd.concat([BadVolume, (df3[(df3['realflow'] - df3['flow']).abs() > 400])])

    return (flag, MeanGEH, GEHPercent)


def time_to_seconds(time_str):
    """ Convert a time string in the format 'HH:MM' to seconds."""
    hour, minute = [int(x) for x in time_str.split(':')]
    return (hour * 3600) + (minute * 60)


def read_MatchupTable(path_matchup_table: str) -> pd.DataFrame:
    """ Load the matchup table from user input

    Args:
        path_matchup_table (str): the path to user updated matchup table

    Returns:
        pd.DataFrame: the loaded matchup table dataframe.
    """

    MatchupTable_UserInput = pd.read_excel(path_matchup_table, skiprows=1, dtype=str)

    # Forward fill missing values in merged columns
    merged_columns = ["JunctionID_OpenDrive",
                      "IntersectionName_GridSmart", "File_Synchro", "Need calibration?"]
    MatchupTable_UserInput[merged_columns] = MatchupTable_UserInput[merged_columns].ffill()
    return MatchupTable_UserInput


def generate_turn_demand_cali(*, path_matchup_table: str | pd.DataFrame,
                              traffic_dir: str) -> list[pd.DataFrame]:
    """ Generate turn demand from user input lookup table and Synchro UTDF files.

    Args:
        path_matchup_table (str): Path to the matchup table with user input.
        traffic_dir (str): Directory where demand files are located.
            Defaults to "Traffic".

    See Also:
        traffic_dir: check sample demand files in datasets/Traffic directory

    Example:
        >>> path_matchup_table = "./MatchupTable_OpenDrive_with user input.xlsx"
        >>> TurnDf, IDRef = generate_turn_demand(path_matchup_table, output_dir="./Output", traffic_dir="Traffic")

    Returns:
        list[pd.DataFrame]: A list containing two DataFrames:
            - TurnDf: DataFrame with turn demand data.
            - IDRef: DataFrame with reference IDs for OpenDrive turns (demand lookup table).
    """

    if isinstance(path_matchup_table, str):
        # Load the MatchupTable_OpenDrive_withsignal.xlsx file, skipping the first row for correct headers
        MatchupTable_UserInput = pd.read_excel(path_matchup_table, skiprows=1, dtype=str)

        # Forward fill missing values in merged columns
        merged_columns = ["JunctionID_OpenDrive", "IntersectionName_GridSmart", "File_Synchro", "Need calibration?"]
        MatchupTable_UserInput[merged_columns] = MatchupTable_UserInput[merged_columns].ffill()
    elif isinstance(path_matchup_table, pd.DataFrame):
        MatchupTable_UserInput = path_matchup_table
    else:
        raise Exception("  : Invalid path_matchup_table, it should be in string path or pandas DataFrame.")

    turn_values = ["NBR", "NBT", "NBL", "NBU", "EBR", "EBT", "EBL", "EBU",
                   "SBR", "SBT", "SBL", "SBU", "WBR", "WBT", "WBL", "WBU"]
    TurnDf_list = []
    IDRef_list = []

    # Process each unique JunctionID_OpenDrive where File_GridSmart has input
    for junction_id in MatchupTable_UserInput["JunctionID_OpenDrive"].unique():
        subset = MatchupTable_UserInput[MatchupTable_UserInput["JunctionID_OpenDrive"] == junction_id]

        # Get the file path from File_GridSmart if available
        file_name = subset["File_GridSmart"].dropna().iloc[0] if not subset["File_GridSmart"].isna().all() else None

        if file_name:
            # Retrieve the IntersectionName_GridSmart
            intersection_name = subset["IntersectionName_GridSmart"].dropna().iloc[0] if not subset["IntersectionName_GridSmart"].isna().all() else "Unknown"
            # Create df_lookup with predefined Turn values
            df_lookup = pd.DataFrame({"Turn": turn_values})
            # Assign IntersectionName
            df_lookup["IntersectionName"] = intersection_name
            # Initialize OpenDriveFromID and OpenDriveToID as empty strings
            df_lookup["OpenDriveFromID"] = ""
            df_lookup["OpenDriveToID"] = ""
            # Map values from subset based on Turn_GridSmart
            for idx, row in df_lookup.iterrows():
                turn = row["Turn"]
                match = subset[subset["Turn_GridSmart"] == turn]
                if not match.empty:
                    df_lookup.at[idx, "OpenDriveFromID"] = match["FromRoadID_OpenDrive"].values[0] if not match["FromRoadID_OpenDrive"].isna().all() else ""
                    df_lookup.at[idx, "OpenDriveToID"] = match["ToRoadID_OpenDrive"].values[0] if not match["ToRoadID_OpenDrive"].isna().all() else ""
            # Append to the list
            IDRef_list.append(df_lookup)

        # Get the file path from File_GridSmart if available
        file_name = subset["File_GridSmart"].dropna().iloc[0] if not subset["File_GridSmart"].isna().all() else None

        if file_name:
            gs_file_path = os.path.join(traffic_dir, file_name)
            # gs_file_path = f"GridSmart/{file_name}"

            # Check if the file exists before processing
            if os.path.exists(gs_file_path):
                # Load the Excel file
                df = pd.read_excel(gs_file_path, header=None)  # Read without predefined headers

                # Find the first row that contains a time value in the first column
                time_row_index = df[df[0].astype(str).str.match(r'^\d{1,2}:\d{2}$', na=False)].index.min()

                # Determine the starting row for data extraction
                start_row = time_row_index - 2 if pd.notna(time_row_index) else None

                if start_row is not None:
                    # Read the file again with proper headers starting from the determined row
                    df_data = pd.read_excel(gs_file_path, header=[start_row, start_row + 1])

                    # Fill merged cells in the first row
                    df_data.columns = df_data.columns.to_frame().fillna(method="ffill").agg("".join, axis=1)

                    # Remove spaces from column names
                    df_data.columns = [col.replace(" ", "") for col in df_data.columns]

                    # Rename the first column to "Time"
                    df_data.rename(columns={df_data.columns[0]: "Time"}, inplace=True)

                    # Drop fully empty columns
                    df_data.dropna(axis=1, how='all', inplace=True)

                    # Drop data of "Total"
                    df_data = df_data[df_data["Time"] != "Total"]

                    # Convert data columns to numeric
                    for col in df_data.columns[1:]:  # Excluding 'Time' column
                        df_data[col] = pd.to_numeric(df_data[col], errors='coerce').fillna(0).astype(int)

                    # Remove 'Unassigned' columns
                    df_data = df_data.loc[:, ~df_data.columns.str.contains(r'Unassigned', na=False)]

                    # Standardize column names for directions
                    df_data.columns = [col.replace("Northbound", "NB")
                                       .replace("Southbound", "SB")
                                       .replace("Westbound", "WB")
                                       .replace("Eastbound", "EB") for col in df_data.columns]

                    # Define expected columns and ensure all are present
                    expected_columns = ["IntersectionName", "Time", "NBR", "NBT", "NBL", "NBU",
                                        "EBR", "EBT", "EBL", "EBU", "SBR", "SBT", "SBL", "SBU",
                                        "WBR", "WBT", "WBL", "WBU"]
                    df_data = df_data.reindex(columns=expected_columns, fill_value="")

                    # Fill IntersectionName using IntersectionName_GridSmart from MatchupTable_UserInput
                    intersection_name = subset["IntersectionName_GridSmart"].dropna().iloc[0] if not subset["IntersectionName_GridSmart"].isna().all() else "Unknown"
                    df_data["IntersectionName"] = intersection_name

                    # Append processed data to list
                    TurnDf_list.append(df_data)

    TurnDf = pd.concat(TurnDf_list, ignore_index=True) if TurnDf_list else pd.DataFrame()

    IDRef = pd.concat(IDRef_list, ignore_index=True) if IDRef_list else pd.DataFrame()
    IDRef = IDRef[["IntersectionName", "Turn", "OpenDriveFromID", "OpenDriveToID"]]

    # replace "" to numpy.nan
    # TurnDf = TurnDf.replace("", np.nan)
    # IDRef = IDRef.replace("", np.nan)
    IDRef = IDRef.dropna(subset=['OpenDriveFromID', 'OpenDriveToID'])

    # drop '' in column OpenDriveFromID and OpenDriveToID
    IDRef = IDRef[IDRef["OpenDriveFromID"].astype(str) != ""]
    IDRef = IDRef[IDRef["OpenDriveToID"].astype(str) != ""]

    return [TurnDf, IDRef]


def generate_inflow(path_net: str,
                    MatchupTable_UserInput: pd.DataFrame,
                    TurnDf: pd.DataFrame,
                    IDRef: pd.DataFrame,
                    sim_begin: int = 28800,
                    sim_end: int = 32400,):
    """ Generate inflow data for calibration."""

    # Apply the conversion function to the 'Time' column and create a new 'Seconds' column
    TurnDf['IntervalStart'] = TurnDf['Time'].apply(time_to_seconds)
    TurnDf['IntervalEnd'] = TurnDf['IntervalStart'] + 15 * 60
    TurnDf = TurnDf.drop('Time', axis=1)

    # Reshape the DataFrame to the long format
    TurnDfTemp = TurnDf.melt(id_vars=['IntersectionName', 'IntervalStart', 'IntervalEnd'],
                             var_name='Turn',
                             value_name='Count')
    # Sort the DataFrame by IntersectionName and Turn columns
    TurnDfTemp.sort_values(['IntersectionName', 'IntervalStart', 'IntervalEnd'], inplace=True)
    # Reset the index
    TurnDfTemp.reset_index(drop=True, inplace=True)
    Count = TurnDfTemp.copy()

    IDRef = IDRef.dropna(subset=['OpenDriveFromID', 'OpenDriveToID'])
    IDRef = IDRef[(IDRef['OpenDriveFromID'] != '') & (IDRef['OpenDriveToID'] != '')]
    IDRef = IDRef.astype({'OpenDriveFromID': int, 'OpenDriveToID': int})
    IDRef = IDRef.astype(str)

    MergedDf1 = pd.merge(Count, IDRef, on=['IntersectionName', 'Turn'], how='left')
    Count['OpenDriveFromID'] = MergedDf1['OpenDriveFromID']
    Count['OpenDriveToID'] = MergedDf1['OpenDriveToID']
    Count['Count'] = Count['Count'].replace('', 0)  # Replace empty strings with 0
    Count['Count'] = Count['Count'].fillna(0)       # Fill NaNs with 0
    Count['Count'] = pd.to_numeric(Count['Count'], errors='coerce').fillna(0).astype(int)  # Ensure it's int
    Count = Count.groupby(['IntervalStart', 'IntervalEnd', 'IntersectionName', 'OpenDriveFromID'],
                          as_index=False)['Count'].sum()
    Count = Count.dropna(subset=['OpenDriveFromID'])
    Count = Count.astype({'Count': int})

    tree = ET.parse(path_net)
    root = tree.getroot()
    # DeadEndJunction = set()
    # for junction in root.findall('junction'):
    #     if junction.attrib.get('type') == 'dead_end':
    #         DeadEndJunction.add(junction.attrib['id'])

    conn_from_map = {}
    for conn in root.findall("connection"):
        from_edge = conn.attrib.get("from")
        direction = conn.attrib.get("dir")
        if from_edge:
            if from_edge not in conn_from_map:
                conn_from_map[from_edge] = []
            conn_from_map[from_edge].append(direction)

    DeadEndJunction = set()
    for junction in root.findall("junction"):
        jid = junction.attrib['id']
        jtype = junction.attrib.get('type')
        incLanes = junction.attrib.get('incLanes', '')
        if jtype == 'dead_end':
            DeadEndJunction.add(jid)
            continue

        if jtype == 'internal':
            continue

        if incLanes == '':
            DeadEndJunction.add(jid)
            continue

        lane_ids = incLanes.strip().split()
        edge_ids = set(lane.split('_')[0] for lane in lane_ids)
        match = True
        for edge_id in edge_ids:
            if edge_id in conn_from_map:
                if any(d != 't' for d in conn_from_map[edge_id]):
                    match = False
                    break
        if match:
            DeadEndJunction.add(jid)

    InflowRoad = []
    SingleRoad = []
    for edge in root.findall('edge'):
        edge_from = edge.attrib.get('from')
        edge_to = edge.attrib.get('to')
        edge_id = edge.attrib.get('id')
        if edge_from in DeadEndJunction:
            InflowRoad.append(edge_id.lstrip('-'))
        if edge_from in DeadEndJunction and edge_to in DeadEndJunction:
            SingleRoad.append(edge_id.lstrip('-'))

    FromRoadID_Sumo = ["-" + str(x) for x in MatchupTable_UserInput["FromRoadID_OpenDrive"].dropna().unique()]
    Lookup_InflowEdge = pd.DataFrame(columns=["FromRoadID_Sumo", "InflowID_Sumo"])
    edges = {edge.get("id"): edge for edge in root.findall("edge")}
    junctions = {junc.get("id"): junc for junc in root.findall("junction")}
    for from_id in FromRoadID_Sumo:
        current_id = from_id
        visited = set()
        while current_id in edges and current_id not in visited:
            visited.add(current_id)
            edge = edges[current_id]
            from_junction_id = edge.get("from")
            junction = junctions.get(from_junction_id)
            if junction is None:
                break
            if junction.get("type") == "dead_end":
                Lookup_InflowEdge.loc[len(Lookup_InflowEdge)] = [
                    from_id, current_id]
                break
            incLanes = junction.get("incLanes")
            if not incLanes:
                break
            lane_ids = [lane.split("_")[0] for lane in incLanes.split()]
            lane_ids = list(set(lane_ids))  # Unique
            lane_ids = [x for x in lane_ids if x]  # Remove empty strings
            if len(lane_ids) != 1:
                break
            next_id = lane_ids[0]
            if next_id not in edges:
                break
            current_id = next_id

    Lookup_InflowEdge["FromRoadID_Sumo_stripped"] = Lookup_InflowEdge["FromRoadID_Sumo"].str.lstrip("-")
    Count = Count.merge(
        Lookup_InflowEdge,
        left_on="OpenDriveFromID",
        right_on="FromRoadID_Sumo_stripped",
        how="left"
    )
    Count["InflowID_Sumo_stripped"] = Count["InflowID_Sumo"].str.lstrip("-")
    InflowCount = Count[Count["InflowID_Sumo_stripped"].isin(InflowRoad)].copy()
    InflowCount.loc[InflowCount["InflowID_Sumo_stripped"].notna(),
                    "OpenDriveFromID"] = InflowCount["InflowID_Sumo_stripped"]
    InflowCount.drop(columns=["InflowID_Sumo",
                              "InflowID_Sumo_stripped",
                              "FromRoadID_Sumo",
                              "FromRoadID_Sumo_stripped"],
                     inplace=True)
    Count.drop(columns=["InflowID_Sumo",
                        "InflowID_Sumo_stripped",
                        "FromRoadID_Sumo",
                        "FromRoadID_Sumo_stripped"],
               inplace=True)

    # create InflowDf_Calibration for turn and inflow purpose
    InflowDf_Calibration = InflowCount[(InflowCount["IntervalStart"] >= sim_begin) &
                                       (InflowCount["IntervalEnd"] <= sim_end)].copy()

    InflowDf_Calibration["OpenDriveFromID"] = InflowDf_Calibration["OpenDriveFromID"].astype(str)
    InflowDf_Calibration["Count"] = InflowDf_Calibration["Count"].astype(int)
    intervals = InflowDf_Calibration[["IntervalStart", "IntervalEnd"]].drop_duplicates()

    # Inflow edges to calibrate
    MissingInflow = [
        road for road in InflowRoad
        if road not in Count['OpenDriveFromID'].unique() and road not in SingleRoad
    ]
    filtered_missing_inflow = MatchupTable_UserInput[
        (MatchupTable_UserInput["FromRoadID_OpenDrive"].isin(MissingInflow)) &
        (MatchupTable_UserInput["Need calibration?"] == "Y")
    ]
    InflowEdgeToCalibrate = filtered_missing_inflow['FromRoadID_OpenDrive'].dropna().unique().tolist()
    N_InflowVariable = len(InflowEdgeToCalibrate)

    calibration_rows = []
    for _, row in intervals.iterrows():
        for edge in InflowEdgeToCalibrate:
            calibration_rows.append({
                "IntervalStart": row["IntervalStart"],
                "IntervalEnd": row["IntervalEnd"],
                "IntersectionName": f"Calibration for Edge {edge}",
                "OpenDriveFromID": edge,
                "Count": 0})
    calibration_df = pd.DataFrame(calibration_rows)
    InflowDf_Calibration = pd.concat([InflowDf_Calibration, calibration_df], ignore_index=True)
    InflowDf_Calibration.sort_values(by=["IntervalStart", "IntersectionName", "OpenDriveFromID"], inplace=True)
    InflowDf_Calibration.reset_index(drop=True, inplace=True)

    return InflowDf_Calibration, InflowEdgeToCalibrate, N_InflowVariable


def generate_turn_summary(TurnDf: pd.DataFrame, MatchupTable_UserInput: pd.DataFrame, N_InflowVariable: int, sim_begin: int = 28800, sim_end: int = 32400, allow_u_turn: bool = False):
    """ Generate turn summary for calibration."""

    # create TurnDf_Calibration for turn and inflow purpose
    TurnDfTemp = TurnDf.melt(id_vars=['IntersectionName', 'IntervalStart', 'IntervalEnd'],
                             var_name='Turn',
                             value_name='Count')
    TurnDfTemp.sort_values(['IntersectionName', 'IntervalStart', 'IntervalEnd'], inplace=True)
    TurnDfTemp.reset_index(drop=True, inplace=True)
    TurnDfTemp['Bound'] = TurnDfTemp['Turn'].str[0]
    TurnDfTemp['Direction'] = TurnDfTemp['Turn'].str[-1]
    TurnDfTemp['Count'] = TurnDfTemp['Count'].replace('', 0)  # Replace empty strings with 0
    TurnDfTemp['Count'] = TurnDfTemp['Count'].fillna(0)       # Fill NaNs with 0
    TurnDfTemp['Count'] = pd.to_numeric(TurnDfTemp['Count'], errors='coerce').fillna(
        0).astype(int)  # Ensure integer type
    FlowTemp = TurnDfTemp.groupby(['IntervalStart',
                                   'IntervalEnd',
                                   'IntersectionName',
                                   'Bound'],
                                  as_index=False)['Count'].sum()
    Turn = pd.merge(TurnDfTemp, FlowTemp, on=['IntervalStart',
                                              'IntervalEnd',
                                              'IntersectionName',
                                              'Bound'],
                    how='left')
    Turn['realcount'] = Turn['Count_x']
    Turn['TurnRatio'] = Turn['Count_x'] / Turn['Count_y']
    Turn = Turn.drop(['Count_x', 'Count_y'], axis=1)
    Turn.reset_index(drop=True, inplace=True)

    Turn = Turn.merge(
        MatchupTable_UserInput[["IntersectionName_GridSmart",
                                "Turn_GridSmart",
                                "Bearing",
                                "Numbering",
                                "FromRoadID_OpenDrive",
                                "ToRoadID_OpenDrive"]],
        how="left",
        left_on=["IntersectionName", "Turn"],
        right_on=["IntersectionName_GridSmart", "Turn_GridSmart"])

    Turn.drop(columns=["IntersectionName_GridSmart", "Turn_GridSmart"], inplace=True)
    Turn_Calibration = Turn[(Turn["IntervalStart"] >= sim_begin) & (Turn["IntervalEnd"] <= sim_end)].copy()
    Turn_Calibration.dropna(subset=["FromRoadID_OpenDrive", "ToRoadID_OpenDrive", "TurnRatio"], inplace=True)

    TurnDf_Calibration = Turn_Calibration.copy()
    TurnDf_Calibration.rename(columns={"FromRoadID_OpenDrive": "OpenDriveFromID",
                                       "ToRoadID_OpenDrive": "OpenDriveToID"}, inplace=True)

    # Turning ratios to calibrate
    MatchupTable_UserInput["merge_key"] = (
        MatchupTable_UserInput["IntersectionName_GridSmart"].astype(str) + "||" +
        MatchupTable_UserInput["Turn_GridSmart"].astype(str))
    Turn["merge_key"] = Turn["IntersectionName"].astype(str) + "||" + Turn["Turn"].astype(str)
    TurnToCalibrate = MatchupTable_UserInput[
        (~MatchupTable_UserInput["merge_key"].isin(Turn["merge_key"])) &
        (MatchupTable_UserInput["Need calibration?"] == "Y")].copy()
    MatchupTable_UserInput.drop(columns=["merge_key"], inplace=True)
    Turn.drop(columns=["merge_key"], inplace=True)
    TurnToCalibrate = TurnToCalibrate[['JunctionID_OpenDrive',
                                       'Bearing',
                                       'Numbering',
                                       'FromRoadID_OpenDrive',
                                       'ToRoadID_OpenDrive',
                                       'Turn']]
    TurnToCalibrate.rename(columns={
        "FromRoadID_OpenDrive": "OpenDriveFromID",
        "ToRoadID_OpenDrive": "OpenDriveToID"}, inplace=True)

    intervals = TurnDf_Calibration[[
        "IntervalStart", "IntervalEnd"]].drop_duplicates()
    intervals["key"] = 1
    TurnToCalibrate["key"] = 1
    new_rows = pd.merge(intervals, TurnToCalibrate, on="key").drop(columns=["key"])
    TurnDf_Calibration = pd.concat([TurnDf_Calibration, new_rows], ignore_index=True)
    TurnDf_Calibration.sort_values(by=["IntervalStart",
                                       "IntersectionName",
                                       "JunctionID_OpenDrive"],
                                   inplace=True)
    TurnDf_Calibration.reset_index(drop=True, inplace=True)
    TurnToCalibrate.drop(columns=["key"], inplace=True)
    TurnDf_Calibration["TurnRatio"] = TurnDf_Calibration["TurnRatio"].fillna(0)

    # If U-turns are not allowed, drop them
    if not allow_u_turn:
        TurnToCalibrate = TurnToCalibrate[TurnToCalibrate["Turn"] != "Uturn"].copy()

    TurnToCalibrate.sort_values(by=["JunctionID_OpenDrive", "Numbering", "Turn"], inplace=True)
    TurnToCalibrate.reset_index(drop=True, inplace=True)

    TurnToCalibrate["Calibration variable?"] = 1
    last_indices = TurnToCalibrate.groupby(["JunctionID_OpenDrive", "Numbering"]).tail(1).index
    TurnToCalibrate.loc[last_indices, "Calibration variable?"] = 0

    N_TurnVariable = (TurnToCalibrate["Calibration variable?"] == 1).sum()
    N_Variable = N_TurnVariable + N_InflowVariable

    # create RealSummary_Calibration for turn and inflow purpose
    RealSummary_Calibration = Turn_Calibration.copy()
    RealSummary_Calibration["entrance_sumo"] = "-" + RealSummary_Calibration["FromRoadID_OpenDrive"].astype(str)
    RealSummary_Calibration["exit_sumo"] = "-" + RealSummary_Calibration["ToRoadID_OpenDrive"].astype(str)

    return TurnToCalibrate, TurnDf_Calibration, RealSummary_Calibration, N_Variable, N_TurnVariable


if __name__ == "__main__":
    pass
#     path_matchup = "./MatchupTable.xlsx"
#     path_net = "./chatt.net.xml"
#
#     MatchupTable_UserInput = read_MatchupTable(path_matchup_table=path_matchup)
#     TurnDf, IDRef = generate_turn_demand_cali(path_matchup_table=path_matchup, traffic_dir="./RealTwinDemand/")
#
#     InflowDf_Calibration, InflowEdgeToCalibrate, N_InflowVariable = generate_inflow(path_net,
#                                                                                     MatchupTable_UserInput,
#                                                                                     TurnDf,
#                                                                                     IDRef)
#
#     (TurnToCalibrate, TurnDf_Calibration,
#      RealSummary_Calibration, N_Variable, N_TurnVariable) = generate_turn_summary(TurnDf,
#                                                                                   MatchupTable_UserInput,
#                                                                                   N_InflowVariable)
