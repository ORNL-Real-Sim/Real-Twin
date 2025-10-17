import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
import shutil
import subprocess
import os, sys
import time

# Set the SUMO data directory
SUMO_DATA_DIR = r"C:\Users\xh8\Downloads\transport\OnlineTests\SimulationModels\Sumo\BeforeTurn"

def setup_sumo_environment():
    """Set up SUMO environment and return key parameters."""
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("please declare environment variable 'SUMO_HOME'")
    NetworkName = "chatt"
    SimName = os.path.join(SUMO_DATA_DIR, "chatt.sumocfg")
    SimulationStartTime = 28800
    SimulationEndTime = 32400
    CalibrationTarget = {'GEH': 5, 'GEHPercent': 0.85}
    CablibrationInterval = 60
    DemandInterval = 15
    ubc = 200
    return NetworkName, SimName, SimulationStartTime, SimulationEndTime, CalibrationTarget, CablibrationInterval, DemandInterval, ubc

def runSumo(SimName, SimulationEndTime):
    import traci
    traci.start(["sumo", "-c", SimName])
    while traci.simulation.getTime() < SimulationEndTime:
        traci.simulationStep()
    traci.close()

def genDemand(NetworkName, SimulationStartTime, SimulationEndTime, TurnDf, InflowDf, ical):
    TurnDf['IntervalStart'] = TurnDf['IntervalStart'].astype(float)
    TurnDf['IntervalEnd'] = TurnDf['IntervalEnd'].astype(float)
    TurnDf = TurnDf[(TurnDf['IntervalStart'] >= SimulationStartTime) & (TurnDf['IntervalEnd'] <= SimulationEndTime)]
    turns = ET.Element('turns')
    IntervalSet = TurnDf[['IntervalStart', 'IntervalEnd']].drop_duplicates().reset_index(drop=True)
    for index, IntervalData in IntervalSet.iterrows():
        Interval = ET.SubElement(turns, 'interval')
        Interval.set('begin', str(IntervalData['IntervalStart']))
        Interval.set('end', str(IntervalData['IntervalEnd']))
        TurnDfSubset = TurnDf[(TurnDf['IntervalStart'] == IntervalData['IntervalStart']) & (TurnDf['IntervalEnd'] == IntervalData['IntervalEnd'])]
        TurnDictSubset = TurnDfSubset.to_dict(orient='records')
        for TurnData in TurnDictSubset:
            edge_relation = ET.SubElement(Interval, 'edgeRelation')
            edge_relation.set('from', str(-int(TurnData['OpenDriveFromID'])))
            edge_relation.set('to', str(-int(TurnData['OpenDriveToID'])))
            edge_relation.set('probability', str(TurnData['TurnRatio']))
    TreeTurn = ET.ElementTree(turns)
    turn_path = os.path.join(SUMO_DATA_DIR, f'route/{NetworkName}{ical}.turn.xml')
    TreeTurn.write(turn_path, encoding='utf-8', xml_declaration=True)

    InflowDf['IntervalStart'] = InflowDf['IntervalStart'].astype(float)
    InflowDf['IntervalEnd'] = InflowDf['IntervalEnd'].astype(float)
    InflowDf = InflowDf[(InflowDf['IntervalStart'] >= SimulationStartTime) & (InflowDf['IntervalEnd'] <= SimulationEndTime)]
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
    TreeInflow = ET.ElementTree(routes)
    flow_path = os.path.join(SUMO_DATA_DIR, f'route/{NetworkName}{ical}.flow.xml')
    TreeInflow.write(flow_path, encoding='utf-8', xml_declaration=True)

    cmd = f'cmd /c "jtrrouter -r {flow_path} -t {turn_path} -n {os.path.join(SUMO_DATA_DIR, NetworkName)}.net.xml --accept-all-destinations --remove-loops True --randomize-flows -o {os.path.join(SUMO_DATA_DIR, f"route/{NetworkName}{ical}.rou.xml")}"'
    process = subprocess.Popen(cmd, shell=True)
    process.wait()
    shutil.copy(
        os.path.join(SUMO_DATA_DIR, f"route/{NetworkName}{ical}.rou.xml"),
        os.path.join(SUMO_DATA_DIR, f"{NetworkName}.rou.xml")
    )

def assignNewTurn(TurnDf, InflowDf, initial_solution, CablibrationInterval, DemandInterval):
    # Assign turn ratios
    TurnDf = TurnDf.copy()
    InflowDf = InflowDf.copy()
    InflowDf['Count'] = InflowDf['Count'].astype(float)
    # Between Amin Dr. and  I-75 SB Off Ramp
    TurnDf.loc[(TurnDf['OpenDriveFromID'] == 290) & (TurnDf['OpenDriveToID'] == 298), 'TurnRatio'] = initial_solution[0]
    TurnDf.loc[(TurnDf['OpenDriveFromID'] == 290) & (TurnDf['OpenDriveToID'] == 299), 'TurnRatio'] = 1-initial_solution[0]
    TurnDf.loc[(TurnDf['OpenDriveFromID'] == 293) & (TurnDf['OpenDriveToID'] == 299), 'TurnRatio'] = initial_solution[1]
    TurnDf.loc[(TurnDf['OpenDriveFromID'] == 293) & (TurnDf['OpenDriveToID'] == 297), 'TurnRatio'] = 1-initial_solution[1]
    TurnDf.loc[(TurnDf['OpenDriveFromID'] == 3151) & (TurnDf['OpenDriveToID'] == 321), 'TurnRatio'] = initial_solution[2]
    TurnDf.loc[(TurnDf['OpenDriveFromID'] == 3151) & (TurnDf['OpenDriveToID'] == 323), 'TurnRatio'] = 1-initial_solution[2]
    TurnDf.loc[(TurnDf['OpenDriveFromID'] == 320) & (TurnDf['OpenDriveToID'] == 3221), 'TurnRatio'] = initial_solution[3]
    TurnDf.loc[(TurnDf['OpenDriveFromID'] == 320) & (TurnDf['OpenDriveToID'] == 321), 'TurnRatio'] = 1-initial_solution[3]
    TurnDf.loc[(TurnDf['OpenDriveFromID'] == 302) & (TurnDf['OpenDriveToID'] == 323), 'TurnRatio'] = initial_solution[4]
    TurnDf.loc[(TurnDf['OpenDriveFromID'] == 302) & (TurnDf['OpenDriveToID'] == 3221), 'TurnRatio'] = 1-initial_solution[4]
    TurnDf.loc[(TurnDf['OpenDriveFromID'] == 316) & (TurnDf['OpenDriveToID'] == 313), 'TurnRatio'] = initial_solution[5]
    TurnDf.loc[(TurnDf['OpenDriveFromID'] == 316) & (TurnDf['OpenDriveToID'] == 315), 'TurnRatio'] = 1-initial_solution[5]
    TurnDf.loc[(TurnDf['OpenDriveFromID'] == 322) & (TurnDf['OpenDriveToID'] == 314), 'TurnRatio'] = initial_solution[6]
    TurnDf.loc[(TurnDf['OpenDriveFromID'] == 322) & (TurnDf['OpenDriveToID'] == 313), 'TurnRatio'] = 1-initial_solution[6]
    TurnDf.loc[(TurnDf['OpenDriveFromID'] == 330) & (TurnDf['OpenDriveToID'] == 327), 'TurnRatio'] = initial_solution[7]
    TurnDf.loc[(TurnDf['OpenDriveFromID'] == 330) & (TurnDf['OpenDriveToID'] == 328), 'TurnRatio'] = 1-initial_solution[7]
    TurnDf.loc[(TurnDf['OpenDriveFromID'] == 307) & (TurnDf['OpenDriveToID'] == 329), 'TurnRatio'] = initial_solution[8]
    TurnDf.loc[(TurnDf['OpenDriveFromID'] == 307) & (TurnDf['OpenDriveToID'] == 327), 'TurnRatio'] = 1-initial_solution[8]
    TurnDf.loc[(TurnDf['OpenDriveFromID'] == 2841) & (TurnDf['OpenDriveToID'] == 328), 'TurnRatio'] = initial_solution[9]
    TurnDf.loc[(TurnDf['OpenDriveFromID'] == 2841) & (TurnDf['OpenDriveToID'] == 329), 'TurnRatio'] = 1-initial_solution[9]
    # Inflow assignments
    InflowDf.loc[(InflowDf['OpenDriveFromID'] == 331), 'Count'] = initial_solution[10]/CablibrationInterval*DemandInterval
    InflowDf.loc[(InflowDf['OpenDriveFromID'] == 320), 'Count'] = initial_solution[11]/CablibrationInterval*DemandInterval
    InflowDf.loc[(InflowDf['OpenDriveFromID'] == 316), 'Count'] = initial_solution[12]/CablibrationInterval*DemandInterval
    InflowDf.loc[(InflowDf['OpenDriveFromID'] == 330), 'Count'] = initial_solution[13]/CablibrationInterval*DemandInterval
    return TurnDf, InflowDf

def resultAnalysis(CalibrationTarget, SimulationEndTime, SimulationStartTime):
    RealSummary = pd.read_excel(os.path.join(SUMO_DATA_DIR, 'summary.xlsx'))
    RealSummary  = RealSummary [RealSummary ["realcount"].notna()]
    ApproachSummary = RealSummary.groupby(['IntersectionName','entrance_sumo','Bound']).agg({'realcount': 'sum'}).reset_index()
    tree = ET.parse(os.path.join(SUMO_DATA_DIR, "EdgeData.xml"))
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
    return flag, MeanGEH, GEHPercent

def objective_function(solution, TurnDf, InflowDf, NetworkName, SimulationStartTime, SimulationEndTime, CablibrationInterval, DemandInterval, SimName, CalibrationTarget):
    TurnDf_updated, InflowDf_updated = assignNewTurn(TurnDf, InflowDf, solution, CablibrationInterval, DemandInterval)
    genDemand(NetworkName, SimulationStartTime, SimulationEndTime, TurnDf_updated, InflowDf_updated, "temp")
    runSumo(SimName, SimulationEndTime)
    _, value, _ = resultAnalysis(CalibrationTarget, SimulationEndTime, SimulationStartTime)
    return value

# def main():
#     """
#     Run SUMO evaluation for a solution provided by an external optimizer,
#     such as onemodel_findpoints_online_modAS_SUMO_turn.py.
#     This function expects the solution to be passed via a file named 'sumo_candidate_solution.txt',
#     where the file contains 14 comma-separated values (10 turn ratios, 4 inflows).
#     """
#     # Setup environment and parameters
#     NetworkName, SimName, SimulationStartTime, SimulationEndTime, CalibrationTarget, CablibrationInterval, DemandInterval, ubc = setup_sumo_environment()
#     # Load data
#     TurnDf = pd.read_excel(os.path.join(SUMO_DATA_DIR, 'turn.xlsx'))
#     InflowDf = pd.read_excel(os.path.join(SUMO_DATA_DIR, 'inflow.xlsx'))

#     # Read solution from file
#     solution_file = 'sumo_candidate_solution.txt'
#     if not os.path.exists(solution_file):
#         print(f"Solution file '{solution_file}' not found. Please provide a solution vector in this file.")
#         return

#     with open(solution_file, 'r') as f:
#         line = f.readline().strip()
#         solution = [float(x) for x in line.split(',')]

#     if len(solution) != 14:
#         print(f"Solution must have 14 values (10 turn ratios, 4 inflows). Got {len(solution)} values.")
#         return

#     print("Evaluating solution from file:", solution)
#     value = objective_function(
#         solution, TurnDf, InflowDf, NetworkName, SimulationStartTime, SimulationEndTime,
#         CablibrationInterval, DemandInterval, SimName, CalibrationTarget
#     )
#     print(f"Objective value (Mean GEH): {value}")
def main():
    """
    Run SUMO evaluation for a solution provided by an external optimizer,
    such as onemodel_findpoints_online_modAS_SUMO_turn.py.
    This function expects the solution to be passed via a file named 'sumo_candidate_solution.txt',
    where the file contains 14 comma-separated values (10 turn ratios, 4 inflows).
    """
    # Setup environment and parameters
    NetworkName, SimName, SimulationStartTime, SimulationEndTime, CalibrationTarget, CablibrationInterval, DemandInterval, ubc = setup_sumo_environment()
    # Load data
    TurnDf = pd.read_excel(os.path.join(SUMO_DATA_DIR, 'turn.xlsx'))
    InflowDf = pd.read_excel(os.path.join(SUMO_DATA_DIR, 'inflow.xlsx'))

    # Read solution from file
    solution_file = 'sumo_candidate_solution.txt'
    if not os.path.exists(solution_file):
        print(f"Solution file '{solution_file}' not found. Please provide a solution vector in this file.")
        return

    with open(solution_file, 'r') as f:
        line = f.readline().strip()
        solution = [float(x) for x in line.split(',')]

    if len(solution) != 14:
        print(f"Solution must have 14 values (10 turn ratios, 4 inflows). Got {len(solution)} values.")
        return

    print("Evaluating solution from file:", solution)
    value = objective_function(
        solution, TurnDf, InflowDf, NetworkName, SimulationStartTime, SimulationEndTime,
        CablibrationInterval, DemandInterval, SimName, CalibrationTarget
    )
    print(f"Objective value (Mean GEH): {value}")

    # --- Get and print final MeanGEH and GEHPercent ---
    flag, meanGEH, GEHPercent = resultAnalysis(CalibrationTarget, SimulationEndTime, SimulationStartTime)
    print(f"Final Mean GEH: {meanGEH}")
    print(f"GEH Percent (fraction < {CalibrationTarget['GEH']}): {GEHPercent*100:.2f}%")
    if flag:
        print("All traffic volume requirements met.")
    else:
        print("Not all traffic volume requirements met.")

if __name__ == "__main__":
    main()
