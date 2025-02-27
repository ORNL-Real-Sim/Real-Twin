'''
##############################################################
# Created Date: Thursday, February 20th 2025
# Contact Info: luoxiangyong01@gmail.com
# Author/Copyright: Mr. Xiangyong Luo
##############################################################
'''

import os
import sys
import xml.etree.ElementTree as ET
import shutil
import subprocess
import pandas as pd
import numpy as np
import pyufunc as pf

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import traci


def update_turn_flow_from_solution(df_turn: pd.DataFrame,
                                   df_inflow: pd.DataFrame,
                                   initial_solution: np.array,
                                   cali_interval: int,
                                   demand_interval: int) -> tuple:
    """assign the new turn ratios and inflow counts to the given dataframes

    Args:
        df_turn (pd.DataFrame): the turn dataframe from turn.xlsx
        df_inflow (pd.DataFrame): the inflow dataframe from inflow.xlsx
        initial_solution (np.array): the initial solution from the genetic algorithm

    Returns:
        tuple: the updated turn and inflow dataframes
    """

    # Much improved in terms of speed and readability. 1 min for 2 - 5 size and generations

    # create the copy of turn and inflow dataframes for internal operations
    TurnDf = df_turn.copy()
    InflowDf = df_inflow.copy()

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
                             df_inflow: pd.DataFrame, ical: str,
                             input_dir: str,
                             output_dir: str,
                             remove_old_files: bool = True) -> bool:
    """Using SUMO jtrrouter to generate the demand file for the given network

    Args:
        network_name (str): the name of the network,
        sim_start_time (float): start time of the simulation
        sim_end_time (float): end time of the simulation
        df_turn (pd.DataFrame): the turn dataframe
        df_inflow (pd.DataFrame): the inflow dataframe
        ical (str): the iteration number for the calibration
        input_dir (str): the path to the input directory
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
    temp_route = pf.path2linux(os.path.join(output_dir, 'temp_route/'))
    os.makedirs(temp_route, exist_ok=True)

    turn_xml_path = f'{temp_route}/{network_name}{ical}.turn.xml'
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
    inflow_xml_path = f'{temp_route}/{network_name}{ical}.flow.xml'
    ET.ElementTree(routes).write(inflow_xml_path,
                                 encoding='utf-8', xml_declaration=True)

    # Generate the route file using jtrrouter
    # Build the command (adjust if necessary for your platform)
    path_net = pf.path2linux(os.path.join(
        input_dir, f"{network_name}.net.xml"))
    path_rou = pf.path2linux(os.path.join(
        input_dir, f"{network_name}.rou.xml"))
    path_temp_rou = pf.path2linux(os.path.join(
        temp_route, f"{network_name}{ical}.rou.xml"))

    cmd = (
        f'cmd /c "jtrrouter -r {inflow_xml_path} -t {turn_xml_path} '
        f'-n {path_net} --accept-all-destinations '
        f'--remove-loops True --randomize-flows -o {path_temp_rou}"'
    )
    process = subprocess.Popen(cmd, shell=True)
    process.wait()

    # Copy the generated route file to the desired location
    shutil.copy(path_temp_rou, path_rou)

    # Clean up temporary files
    if remove_old_files:
        os.remove(turn_xml_path)
        os.remove(inflow_xml_path)
        os.remove(path_temp_rou)

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


def result_analysis_on_EdgeData(path_summary: str | pd.DataFrame,
                                path_EdgeData: str,
                                calibration_target: dict,
                                sim_start_time: float,
                                sim_end_time: float) -> tuple:
    """Analyze the result of the simulation and return the flag, mean GEH, and GEH percent

    Args:
        df_summary (pd.DataFrame): the summary dataframe from summary.xlsx in input dir
        calibration_target (dict): the calibration target from the scenario config, it should contain GEH and GEHPercent
        sim_start_time (float): the start time of the simulation
        sim_end_time (float): the end time of the simulation
        path_edge (str): the path to the EdgeData.xml file in the input dir

    Returns:
        tuple: (flag, mean GEH, geh percent)
    """
    # Load and parse the new XML file
    # mapping of sumo id with GridSmart Intersection from user input

    # 1. Filter and group the summary data
    if isinstance(path_summary, str):
        df_summary = pd.read_excel(path_summary)
    elif isinstance(path_summary, pd.DataFrame):
        df_summary = path_summary
    else:
        raise ValueError("path_summary must be either a str or a pd.DataFrame")

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
    approach_summary['entrance_sumo'] = approach_summary['entrance_sumo'].astype(
        int)
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


# old functions


def run_SUMO(sim_name: str, sim_end_time: float) -> bool:
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


def generate_demand(network_name: str, sim_start_time: float, sim_end_time: float,
                    df_turn: pd.DataFrame, df_inflow: pd.DataFrame, ical,
                    output_dir: str = "None"):

    # create the copy of turn and inflow dataframes for internal operations
    TurnDf = df_turn.copy()
    InflowDf = df_inflow.copy()
    TurnDf['IntervalStart'] = TurnDf['IntervalStart'].astype(float)
    TurnDf['IntervalEnd'] = TurnDf['IntervalEnd'].astype(float)
    TurnDf = TurnDf[(TurnDf['IntervalStart'] >= sim_start_time)
                    & (TurnDf['IntervalEnd'] <= sim_end_time)]
    turns = ET.Element('turns')
    # Create the 'interval' element
    IntervalSet = TurnDf[['IntervalStart', 'IntervalEnd']
                         ].drop_duplicates().reset_index(drop=True)
    for index, IntervalData in IntervalSet.iterrows():
        Interval = ET.SubElement(turns, 'interval')
        Interval.set('begin', str(IntervalData['IntervalStart']))
        Interval.set('end', str(IntervalData['IntervalEnd']))
        TurnDfSubset = TurnDf[(TurnDf['IntervalStart'] == IntervalData['IntervalStart'])
                              & (TurnDf['IntervalEnd'] == IntervalData['IntervalEnd'])]
        TurnDictSubset = TurnDfSubset.to_dict(orient='records')
        for TurnData in TurnDictSubset:
            edge_relation = ET.SubElement(Interval, 'edgeRelation')
            edge_relation.set('from', str(-int(TurnData['OpenDriveFromID'])))
            edge_relation.set('to', str(-int(TurnData['OpenDriveToID'])))
            edge_relation.set('probability', str(TurnData['TurnRatio']))
    # <edgeRelation from="" probability="" to=""/>
    TreeTurn = ET.ElementTree(turns)
    # Write the XML to the file
    TreeTurn.write('route/{}{}.turn.xml'.format(network_name,
                   ical), encoding='utf-8', xml_declaration=True)
    # Create the .flow.xml
    InflowDf['IntervalStart'] = InflowDf['IntervalStart'].astype(float)
    InflowDf['IntervalEnd'] = InflowDf['IntervalEnd'].astype(float)
    InflowDf = InflowDf[(InflowDf['IntervalStart'] >= sim_start_time)
                        & (InflowDf['IntervalEnd'] <= sim_end_time)]
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
        flow.set('type', 'car')
    # <flow begin="0.0" end="3600.0" from="" id="" number="" type="car"/>
    TreeInflow = ET.ElementTree(routes)
    # TreeInflow.write('MyNetwork/SUMO/{}.flow.xml'.format(NetworkName), encoding='utf-8', xml_declaration=True)
    TreeInflow.write(f'route/{network_name}{ical}.flow.xml',
                     encoding='utf-8', xml_declaration=True)
    # Create the .rou.xml
    cmd = f'cmd /c "jtrrouter -r route/{network_name}{ical}.flow.xml -t route/{network_name}{ical}.turn.xml -n {network_name}.net.xml --accept-all-destinations --remove-loops True --randomize-flows -o route/{network_name}{ical}.rou.xml"'
    process = subprocess.Popen(cmd, shell=True)
    process.wait()
    shutil.copy("route/{}{}.rou.xml".format(network_name, ical),
                "{}.rou.xml".format(network_name))


def assign_new_turn(df_turn: pd.DataFrame, df_inflow: pd.DataFrame, initial_solution: np.array) -> tuple:

    TurnDf = df_turn.copy()
    InflowDf = df_inflow.copy()

    # Between Amin Dr. and  I-75 SB Off Ramp
    TurnDf.loc[(TurnDf['OpenDriveFromID'] == 290) & (TurnDf['OpenDriveToID'] == 298),
               'TurnRatio'] = initial_solution[0]
    TurnDf.loc[(TurnDf['OpenDriveFromID'] == 290) & (TurnDf['OpenDriveToID'] == 299),
               'TurnRatio'] = 1 - initial_solution[0]
    TurnDf.loc[(TurnDf['OpenDriveFromID'] == 331) & (TurnDf['OpenDriveToID'] == 297),
               'TurnRatio'] = initial_solution[1]
    TurnDf.loc[(TurnDf['OpenDriveFromID'] == 331) & (TurnDf['OpenDriveToID'] == 298),
               'TurnRatio'] = 1 - initial_solution[1]
    TurnDf.loc[(TurnDf['OpenDriveFromID'] == 293) & (TurnDf['OpenDriveToID'] == 299),
               'TurnRatio'] = initial_solution[2]
    TurnDf.loc[(TurnDf['OpenDriveFromID'] == 293) & (TurnDf['OpenDriveToID'] == 297),
               'TurnRatio'] = 1 - initial_solution[2]
    # Between Napier Rd. and Lifestyle Way1
    TurnDf.loc[(TurnDf['OpenDriveFromID'] == 315) & (TurnDf['OpenDriveToID'] == 321),
               'TurnRatio'] = initial_solution[3]
    TurnDf.loc[(TurnDf['OpenDriveFromID'] == 315) & (TurnDf['OpenDriveToID'] == 323),
               'TurnRatio'] = 1 - initial_solution[3]
    TurnDf.loc[(TurnDf['OpenDriveFromID'] == 320) & (TurnDf['OpenDriveToID'] == 3221),
               'TurnRatio'] = initial_solution[4]
    TurnDf.loc[(TurnDf['OpenDriveFromID'] == 320) & (TurnDf['OpenDriveToID'] == 321),
               'TurnRatio'] = 1 - initial_solution[4]
    TurnDf.loc[(TurnDf['OpenDriveFromID'] == 302) & (TurnDf['OpenDriveToID'] == 323),
               'TurnRatio'] = initial_solution[5]
    TurnDf.loc[(TurnDf['OpenDriveFromID'] == 302) & (TurnDf['OpenDriveToID'] == 3221),
               'TurnRatio'] = 1 - initial_solution[5]
    # Between Napier Rd. and Lifestyle Way2
    TurnDf.loc[(TurnDf['OpenDriveFromID'] == 281) & (TurnDf['OpenDriveToID'] == 315),
               'TurnRatio'] = initial_solution[6]
    TurnDf.loc[(TurnDf['OpenDriveFromID'] == 281) & (TurnDf['OpenDriveToID'] == 314),
               'TurnRatio'] = 1 - initial_solution[6]
    TurnDf.loc[(TurnDf['OpenDriveFromID'] == 316) & (TurnDf['OpenDriveToID'] == 313),
               'TurnRatio'] = initial_solution[7]
    TurnDf.loc[(TurnDf['OpenDriveFromID'] == 316) & (TurnDf['OpenDriveToID'] == 315),
               'TurnRatio'] = 1 - initial_solution[7]
    TurnDf.loc[(TurnDf['OpenDriveFromID'] == 322) & (TurnDf['OpenDriveToID'] == 314),
               'TurnRatio'] = initial_solution[8]
    TurnDf.loc[(TurnDf['OpenDriveFromID'] == 322) & (TurnDf['OpenDriveToID'] == 313),
               'TurnRatio'] = 1 - initial_solution[8]
    # Between  Lifestyle Way and Gunbarrel Road
    TurnDf.loc[(TurnDf['OpenDriveFromID'] == 330) & (TurnDf['OpenDriveToID'] == 327),
               'TurnRatio'] = initial_solution[9]
    TurnDf.loc[(TurnDf['OpenDriveFromID'] == 330) & (TurnDf['OpenDriveToID'] == 328),
               'TurnRatio'] = 1 - initial_solution[9]
    TurnDf.loc[(TurnDf['OpenDriveFromID'] == 307) & (TurnDf['OpenDriveToID'] == 329),
               'TurnRatio'] = initial_solution[10]
    TurnDf.loc[(TurnDf['OpenDriveFromID'] == 307) & (TurnDf['OpenDriveToID'] == 327),
               'TurnRatio'] = 1 - initial_solution[10]
    TurnDf.loc[(TurnDf['OpenDriveFromID'] == 284) & (TurnDf['OpenDriveToID'] == 328),
               'TurnRatio'] = initial_solution[11]
    TurnDf.loc[(TurnDf['OpenDriveFromID'] == 284) & (TurnDf['OpenDriveToID'] == 329),
               'TurnRatio'] = 1 - initial_solution[11]
    # Between Amin Dr. and  I-75 SB Off Ramp
    InflowDf['Count'] = InflowDf['Count'].astype(float)
    InflowDf.loc[(InflowDf['OpenDriveFromID'] == 331),
                 'Count'] = initial_solution[12]
    # Between Napier Rd. and Lifestyle Way1
    InflowDf.loc[(InflowDf['OpenDriveFromID'] == 320),
                 'Count'] = initial_solution[13]
    # Between Napier Rd. and Lifestyle Way2
    InflowDf.loc[(InflowDf['OpenDriveFromID'] == 316),
                 'Count'] = initial_solution[14]
    # Between  Lifestyle Way and Gunbarrel Road
    InflowDf.loc[(InflowDf['OpenDriveFromID'] == 330),
                 'Count'] = initial_solution[15]

    return (TurnDf, InflowDf)


def result_analysis(df_summary: str | pd.DataFrame,
                    calibration_target: dict,
                    sim_start_time: float,
                    sim_end_time: float) -> tuple:
    # Load and parse the new XML file
    # mapping of sumo id with GridSmart Intersection from user input

    RealSummary = df_summary.copy()
    RealSummary = RealSummary[RealSummary["realcount"].notna()]
    ApproachSummary = RealSummary.groupby(['IntersectionName',
                                           'entrance_sumo',
                                           'Bound']).agg({'realcount': 'sum'}).reset_index()
    tree = ET.parse("EdgeData.xml")
    root = tree.getroot()
    edge_data = []
    for interval in root.findall('.//interval'):
        for edge in interval.findall('edge'):
            edge_id = edge.get('id')
            travel_time = edge.get('traveltime')
            density = edge.get('density')
            speed = edge.get('speed')
            arrived = edge.get('arrived')
            departed = edge.get('departed')
            left = edge.get('left')
            edge_data.append({'id': edge_id,
                              'travel_time': travel_time,
                              'arrived': arrived,
                              'departed': departed,
                              "left": left,
                              'density': density,
                              'speed': speed})
    EdgeData = pd.DataFrame(edge_data)
    EdgeData = EdgeData.astype({'id': int,
                                'travel_time': float,
                                'arrived': int,
                                'departed': int,
                                "left": int,
                                'density': float,
                                'speed': float})
    ApproachSummary['entrance_sumo'] = ApproachSummary['entrance_sumo'].astype(
        int)
    EdgeData['id'] = EdgeData['id'].astype(int)
    ApproachSummary = pd.merge(
        ApproachSummary, EdgeData, left_on='entrance_sumo', right_on='id')
    ApproachSummary.rename(columns={'left': 'count'}, inplace=True)
    ApproachSummary.drop(columns=['id'], inplace=True)
    ApproachSummary['flow'] = ApproachSummary['count'] / \
        (sim_end_time - sim_start_time) * 3600
    ApproachSummary['realflow'] = ApproachSummary['realcount'] / \
        (sim_end_time - sim_start_time) * 3600
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
        BadVolume = pd.concat(
            [BadVolume, (df1[(df1['realflow']-df1['flow']).abs() > 100])])
    # within 15%  for volumes  700-2700
    df2 = ApproachSummary[(ApproachSummary['realflow'] >= 700) & (
        ApproachSummary['realflow'] <= 2700)]
    if sum(((df2['realflow'] - df2['flow']) / -df2['realflow']).abs() > 0.15) > 0:
        flag = 0
        BadVolume = pd.concat(
            [BadVolume, (df2[((df2['realflow'] - df2['flow']) / -df2['realflow']).abs() > 0.15])])
    # within 400 vph for volumes > 2700
    df3 = ApproachSummary[ApproachSummary['realflow'] > 2700]
    if sum((df3['realflow'] - df3['flow']).abs() > 400) > 0:
        flag = 0
        BadVolume = pd.concat(
            [BadVolume, (df3[(df3['realflow'] - df3['flow']).abs() > 400])])
    return (flag, MeanGEH, GEHPercent)
