import os
import sys
import time
import subprocess
import random
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET

def setup_sumo_environment():
    """Set up SUMO environment and return file paths and parameters."""
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("please declare environment variable 'SUMO_HOME'")

    filePath = "C:\\Users\\ets\\Desktop\\GithubProjects\\New_Bayesian\\transport-optimizer\\transport\\OnlineTests\\SimulationModels\\Sumo\\AfterTurnInflow\\"
    fPath_orig = "C:\\Users\\ets\\Dropbox (ORNL)\\VSI_Projects\\Projects Folder\\CC_RealTwin\\Modifiled Calibration 04.16\\SUMO\\Turn and Inflow\\"
    EB_tt = 180
    WB_tt = 180
    EB_edge_list = ["-312","-293", "-297", "-288",  "-2881", "-286", "-302", "-3221", "-322", "-313", "-284", "-2841", "-328", "-304"]
    WB_edge_list = ["-2801","-280", "-307","-327", "-3271", "-281", "-315", "-3151", "-321", "-300", "-2851", "-285", "-290", "-298", "-295"]
    return filePath, fPath_orig, EB_tt, WB_tt, EB_edge_list, WB_edge_list

def get_travel_time(edge_output_file, edge_ids):
    total_travel_time = 0.0
    tree = ET.parse(edge_output_file)
    root = tree.getroot()
    for interval in root.findall("interval"):
        interval_id = interval.get('id')
        if interval_id == '1': 
            for edge in interval.findall("edge"):
                edge_id = edge.get("id")
                if edge_id in edge_ids:
                    travel_time = float(edge.get("traveltime", 0))
                    if travel_time is not None:
                        total_travel_time += float(travel_time)
    return total_travel_time

def change_CF_parameters_in_SUMO(filePath, solution):
    min_gap = solution[0]
    accel = solution[1]
    decel = solution[2]
    sigma = solution[3]
    tau = solution[4]
    emergencyDecel = solution[5]
    tree = ET.parse(filePath+'chatt.flow.xml')
    root = tree.getroot()
    parent = root.find('vType')
    if parent is not None:
        parent.set('minGap', str(min_gap))
        parent.set('accel', str(accel))
        parent.set('decel', str(decel))
        parent.set('sigma', str(sigma))
        parent.set('tau', str(tau))
        parent.set('emergencyDecel', str(emergencyDecel))
    else:
        print("Parent tag not found")
    tree.write(filePath+'chatt.flow.xml')
    return 

def run_jtrrouter(filePath, net_file, flow_file, turn_file, output_file):
    cmd = [
        "jtrrouter",
        "-n", filePath+net_file,
        "-r", filePath+flow_file,
        "-t", filePath+turn_file,
        "-o", filePath+output_file,
        "--accept-all-destinations",
        "--remove-loops",
        "--seed","101",
        "--ignore-errors",
    ]
    try:
        subprocess.run(cmd,  capture_output=True, text=True)
        print(f"Route file generated successfully: {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running jtrrouter: {e}")

def fitness_func2(solution, filePath, EB_tt, WB_tt, EB_edge_list, WB_edge_list):
    print("Evaluating solution:", solution)
    change_CF_parameters_in_SUMO(filePath, solution)
    net_file = "chatt.net.xml"
    flow_file = "chatt.flow.xml"
    turn_file = "chatt.turn.xml"
    output_file = "chatt.rou.xml"
    run_jtrrouter(filePath, net_file, flow_file, turn_file, output_file)
    sumo_cfg_file = filePath+"chatt.sumocfg"
    sumo_command = f"sumo -c \"{sumo_cfg_file}\""
    sumoProcess = subprocess.Popen(sumo_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    sumoProcess.wait()
    edge_data_file= filePath+"EdgeData.xml"
    while not os.path.exists(edge_data_file):
        time.sleep(1)
    travel_time_EB = get_travel_time(edge_data_file, EB_edge_list)
    travel_time_WB = get_travel_time(edge_data_file, WB_edge_list)
    print("Travel times:", travel_time_EB, travel_time_WB)
    # fitness_mae = -((abs(EB_tt -travel_time_EB)+abs(WB_tt-travel_time_WB))/2)
    fitness_mae = ((abs(EB_tt -travel_time_EB)+abs(WB_tt-travel_time_WB))/2)
    
    print("Fitness (MAE):", fitness_mae)
    return fitness_mae

def resultAnalysis(filepath, CalibrationTarget,SimulationEndTime,SimulationStartTime):
    RealSummary = pd.read_excel(filepath+'summary.xlsx')
    RealSummary  = RealSummary [RealSummary ["realcount"].notna()]
    ApproachSummary = RealSummary.groupby(['IntersectionName','entrance_sumo','Bound']).agg({'realcount': 'sum'}).reset_index()
    tree = ET.parse(filepath+"EdgeData.xml")
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
            edge_data.append({'id': edge_id, 'travel_time': travel_time,'arrived': arrived, 'departed': departed, "left": left, 'density': density, 'speed': speed})
    EdgeData = pd.DataFrame(edge_data)
    EdgeData = EdgeData.astype({'id': int, 'travel_time': float,'arrived': int, 'departed': int, "left": int, 'density': float, 'speed': float})
    ApproachSummary ['entrance_sumo'] = ApproachSummary ['entrance_sumo'].astype(int)
    EdgeData['id'] = EdgeData['id'].astype(int)
    ApproachSummary = pd.merge(ApproachSummary, EdgeData, left_on='entrance_sumo', right_on='id')
    ApproachSummary.rename(columns={'left': 'count'}, inplace=True)
    ApproachSummary.drop(columns=['id'], inplace=True)
    ApproachSummary['flow'] = ApproachSummary['count']/(SimulationEndTime-SimulationStartTime)*3600
    ApproachSummary['realflow'] = ApproachSummary['realcount']/(SimulationEndTime-SimulationStartTime)*3600
    ApproachSummary['GEH'] = np.sqrt(2 * (ApproachSummary['count'] - ApproachSummary['realcount'])**2 / (ApproachSummary['count'] + ApproachSummary['realcount']))
    MeanGEH = ApproachSummary['GEH'].mean()
    GEHPercent = (ApproachSummary['GEH'] < CalibrationTarget['GEH']).mean()
    flag = 1
    if GEHPercent < CalibrationTarget['GEHPercent']:
        flag = 0
    BadVolume = ApproachSummary[ApproachSummary['count']<0]
    df1 = ApproachSummary[ApproachSummary['realflow']<700]
    if  sum((df1['realflow']-df1['flow']).abs()>100)>0:
        flag = 0
        BadVolume = pd.concat([BadVolume,(df1[(df1['realflow']-df1['flow']).abs()>100])])  
    df2 = ApproachSummary[(ApproachSummary['realflow']>=700) & (ApproachSummary['realflow']<=2700)]
    if  sum(((df2['realflow']-df2['flow'])/-df2['realflow']).abs()>0.15)>0:
        flag = 0
        BadVolume = pd.concat([BadVolume,(df2[((df2['realflow']-df2['flow'])/-df2['realflow']).abs()>0.15])])
    df3 = ApproachSummary[ApproachSummary['realflow']>2700]  
    if sum((df3['realflow']-df3['flow']).abs()>400)>0:
        flag = 0
        BadVolume = pd.concat([BadVolume,(df3[(df3['realflow']-df3['flow']).abs()>400])])
    return flag,MeanGEH,GEHPercent

def main():
    filePath, fPath_orig, EB_tt, WB_tt, EB_edge_list, WB_edge_list = setup_sumo_environment()
    SimulationStartTime = 28800
    SimulationEndTime = 32400
    CalibrationTarget = {'GEH':5,'GEHPercent':0.85}
    initial_parameters = [2.5, 2.6, 4.5, 0.5, 1.0,  9.0]
    print("Running initial fitness evaluation...")
    fitness_original = fitness_func2(initial_parameters, filePath, EB_tt, WB_tt, EB_edge_list, WB_edge_list)
    GEHOriginalData = resultAnalysis(filePath, CalibrationTarget,SimulationEndTime,SimulationStartTime)
    MeanGEH_original = GEHOriginalData[1]
    GEHPercent_original = GEHOriginalData[2]
    print("MeanGEH_original", MeanGEH_original)
    print ("GEHPercent_original", GEHPercent_original)
    # You can add more logic here to run optimization, compare models, etc.

if __name__ == "__main__":
    main()