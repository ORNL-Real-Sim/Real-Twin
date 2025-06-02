##############################################################################
# Copyright (c) 2024, Oak Ridge National Laboratory                          #
# All rights reserved.                                                       #
#                                                                            #
# This file is part of RealTwin and is distributed under a MIT               #
# license. For the licensing terms see the LICENSE file in the top-level     #
# directory.                                                                 #
#                                                                            #
# Contributors: ORNL Real-Twin Team                                          #
# Contact: realtwin@ornl.gov                                                 #
##############################################################################
import os
import sys
import xml.etree.ElementTree as ET
import subprocess
import warnings
import pandas as pd
import numpy as np
import folium
import sumolib
from itertools import combinations
import requests

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
    sys.path = list(set(sys.path))  # remove duplicates
else:
    warnings.warn("Environment variable 'SUMO_HOME' is not set. "
                  "please declare environment variable 'SUMO_HOME'")
    # sys.exit("please declare environment variable 'SUMO_HOME'")
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
    # cmd = [
    #     "jtrrouter",
    #     "-n", path_net,
    #     "-r", path_flow,
    #     "-t", path_turn,
    #     "-o", path_rou,
    #     "--accept-all-destinations",
    #     "--remove-loops True",
    #     # "--seed","101",
    #     "--ignore-errors",  # Continue on errors; remove if not desired
    # ]
    cmd = f'cmd /c "jtrrouter -r {path_flow} -t {path_turn} -n {path_net} --accept-all-destinations --remove-loops True --randomize-flows -o {path_rou}"'

    # Execute the command
    try:
        # subprocess.run(cmd, capture_output=True, text=True)
        process = subprocess.Popen(cmd, shell=True)
        process.wait()
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


def compute_route_summary(rou_file: str, net_file: str) -> pd.DataFrame:
    """
    Parses the SUMO .rou.xml and .net.xml files, counts each unique route's frequency,
    computes its total length, and returns a DataFrame sorted by descending frequency.
    """
    # 1) load network
    net = sumolib.net.readNet(net_file)

    # 2) parse the .rou.xml
    tree = ET.parse(rou_file)
    root = tree.getroot()

    # 3) accumulate counts and lengths
    route_counts = {}
    route_lengths = {}
    for route_elem in root.findall(".//route"):
        edges = tuple(route_elem.attrib["edges"].split())
        # freq
        route_counts[edges] = route_counts.get(edges, 0) + 1
        # length
        if edges not in route_lengths:
            route_lengths[edges] = sum(
                net.getEdge(e).getLength()
                for e in edges
                if net.getEdge(e) is not None
            )

    # 4) build DataFrame
    df = pd.DataFrame([
        {
            "route": " ".join(edges),
            "frequency": freq,
            "length_meters": route_lengths[edges]
        }
        for edges, freq in route_counts.items()
    ])

    # 5) sort and return
    return df.sort_values("frequency", ascending=False).reset_index(drop=True)


def filter_mid_routes(df_sorted: pd.DataFrame) -> pd.DataFrame:
    """Filter 80-90th percentile by length"""

    length_85 = df_sorted["length_meters"].quantile(0.85)
    length_95 = df_sorted["length_meters"].quantile(0.95)
    return df_sorted[(df_sorted["length_meters"] >= length_85)
                     & (df_sorted["length_meters"] < length_95)].reset_index(drop=True)


def select_two_distinct(mid_routes: pd.DataFrame) -> list[list[str]]:
    """Select two most distinct via Jaccard """

    def jaccard_dist(a: str, b: str) -> float:
        sa, sb = set(a.split()), set(b.split())
        return 1 - len(sa & sb) / len(sa | sb)

    pairs = list(combinations(range(len(mid_routes)), 2))
    scores = [
        (i, j, jaccard_dist(mid_routes.loc[i, "route"], mid_routes.loc[j, "route"]))
        for i, j in pairs
    ]
    i, j, _ = max(scores, key=lambda x: x[2])
    return [
        mid_routes.loc[i, "route"].split(),
        mid_routes.loc[j, "route"].split()
    ]


def get_route_coords(net, edge_list: list[str]) -> list[tuple[float, float]]:
    """Extract WGS84 coords for a route
    net: a sumolib Net object
    edge_list: list of edge IDs (strings)
    """
    pts = []
    for eid in edge_list:
        edge = net.getEdge(eid)
        for x, y in edge.getShape():
            lon, lat = net.convertXY2LonLat(x, y)
            pts.append((lat, lon))
    return pts


def estimate_travel_time(route_coords: list[tuple[float, float]], api_key: str = "") -> int | None:
    """Estimate travel time with waypoint downsampling"""
    if len(route_coords) < 2:
        return None

    if not api_key:
        api_key = "AIzaSyBH5lSmawDWrJpPw9cd4h6wgN04lGNt3_A"

    def downsample(C: list, max_pts=23) -> list:
        if len(C) <= max_pts:
            return C
        step = max(1, len(C) // max_pts)
        return C[::step][:max_pts]

    origin = f"{route_coords[0][0]},{route_coords[0][1]}"
    destination = f"{route_coords[-1][0]},{route_coords[-1][1]}"
    mids = downsample(route_coords[1:-1], 23)
    waypts = "|".join(f"{lat},{lon}" for lat, lon in mids)

    resp = requests.get(
        "https://maps.googleapis.com/maps/api/directions/json",
        params={
            "origin": origin,
            "destination": destination,
            "waypoints": waypts,
            "departure_time": "now",
            "key": api_key
        }
    )
    j = resp.json()
    if j.get("status") == "OK":
        return sum(leg["duration"]["value"] for leg in j["routes"][0]["legs"])
    else:
        print("API error:", j.get("status"), j.get("error_message"))
        return None


def plot_and_report(route_coords_list: list[list[tuple[float, float]]],
                    travel_times: list[int | None],
                    selected_routes: list[list[str]],
                    out_html: str = "routes_travel_time_map.html"):

    """ Plot routes, print edges & times, and add legend"""
    # Print edge lists and travel times
    for idx, edges in enumerate(selected_routes, start=1):
        print(f"Route {idx} edges: {edges}")
        t = travel_times[idx - 1]
        if t is not None:
            print(f"Route {idx} travel time: {t} sec ({t / 60:.1f} min)")
        else:
            print(f"Route {idx} travel time: not available")

    # Folium map
    all_pts = [pt for coords in route_coords_list for pt in coords]
    center = (
        sum(lat for lat, lon in all_pts) / len(all_pts),
        sum(lon for lat, lon in all_pts) / len(all_pts)
    )
    m = folium.Map(location=center, zoom_start=14)
    colors = ["red", "blue"]

    for idx, coords in enumerate(route_coords_list, start=1):
        tooltip = f"Route {idx}"
        if travel_times[idx - 1] is not None:
            tooltip += f": {travel_times[idx - 1]} sec"
        folium.PolyLine(coords, color=colors[idx - 1], weight=5, opacity=0.8,
                        tooltip=tooltip).add_to(m)

    # Legend
    legend_html = """
     <div style="
       position: fixed;
       bottom: 50px; left: 50px; width: 150px; height: 80px;
       background-color: white; border:2px solid grey; z-index:9999;
       font-size:14px; line-height:18px;
       ">
       &nbsp;<b>Route Legend</b><br>
       &nbsp;<span style="color:red;">&#8212;&#8212;</span>&nbsp;Route 1<br>
       &nbsp;<span style="color:blue;">&#8212;&#8212;</span>&nbsp;Route 2
     </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    m.save(out_html)
    print(f"Map saved → {out_html}")


def auto_select_two_routes(path_rou: str, path_net: str, api_key: str = "",
                           path_report: str = "routes_travel_time_map.html") -> tuple:
    """Automatically select two distinct routes from the given route file and network file.
    Using Google API to estimate travel time.

    Args:
        path_rou (str): sumo .rou.xml file
        path_net (str): sumo .net.xml file
        path_output (_type_): save the map to this path
        api_key (str, optional): Google Api key. Defaults to "".

    Returns:
        tuple: time and edge id list for the two routes
    """

    df_sorted = compute_route_summary(path_rou, path_net)
    mid_df = filter_mid_routes(df_sorted)
    routes = select_two_distinct(mid_df)

    net = sumolib.net.readNet(path_net)
    coords_list = [get_route_coords(net, route) for route in routes]
    time_list = [estimate_travel_time(coords, api_key) for coords in coords_list]
    plot_and_report(coords_list, time_list, routes, path_report)
    return (routes, time_list, coords_list)
