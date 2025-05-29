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
"""The module to handle the SUMO simulation for the real-twin developed by ORNL ARMS group."""

import os
import shutil
import warnings
from pathlib import Path
import xml.etree.ElementTree as ET
from xml.dom.minidom import parseString
import xml.dom.minidom as minidom
import pandas as pd
import io
import pyufunc as pf
pd.options.mode.chained_assignment = None
warnings.simplefilter(action='ignore', category=FutureWarning)

from realtwin.func_lib._f_calibration.algo_sumo.util_cali_turn_inflow import (read_MatchupTable,
                                                                               generate_turn_demand_cali,
                                                                               generate_inflow,
                                                                               generate_turn_summary)


class SUMOPrep:
    """The class to handle the SUMO simulation for the real-twin developed by ORNL ARMS group.
    """
    def __init__(self, **kwargs):
        self.Network = {}
        # self.NetworkWithElevation = {}
        self.Demand = set()
        self.Signal = {}

        # add kwargs to the class
        self.kwargs = kwargs

    def importNetwork(self, ConcreteScn):
        """The function to import the network from the OpenDrive file and convert it to SUMO network file."""

        # self.Network = ConcreteScn.Supply.Network
        # self.NetworkWithElevation = ConcreteScn.Supply.NetworkWithElevation
        NetworkName = ConcreteScn.Supply.NetworkName

        # get output path from the configuration dict
        path_output = ConcreteScn.input_config.get('output_dir')

        self.SUMOPath = pf.path2linux(os.path.join(path_output, 'SUMO'))
        if os.path.exists(self.SUMOPath):
            shutil.rmtree(self.SUMOPath)
        os.mkdir(self.SUMOPath)

        # Sumo combine the OpenDrive file to sumo network file
        path_open_drive = pf.path2linux(os.path.join(path_output, f'OpenDrive/{NetworkName}.xodr'))
        path_sumo_net = pf.path2linux(os.path.join(path_output, f'SUMO/{NetworkName}.net.xml'))

        os.system(f'cmd/c "netconvert --opendrive {path_open_drive}'
                  f' -o {path_sumo_net} --no-internal-links"')
        self.Network = path_sumo_net

        if ConcreteScn.input_config.get('incl_sumo_net'):
            shutil.copy(ConcreteScn.input_config["incl_sumo_net"], self.Network)

        # Load the XML file
        tree = ET.parse(self.Network)
        root = tree.getroot()
        # Find all junctions
        junctions = root.findall('junction')

        # Function to extract road ids from incLanes attribute
        def get_road_ids_from_incLanes(incLanes):
            lane_ids = incLanes.split()
            road_ids = set(lane_id.split("_")[0] for lane_id in lane_ids)
            return list(road_ids)

        # Find all junctions with only one road connecting
        junctions_single_road = [
            junction for junction in junctions
            if len(get_road_ids_from_incLanes(junction.get('incLanes'))) == 1]

        # Iterate over junctions with only one road connecting
        for junction in junctions_single_road:
            # Get the id of the only road connected to the junction
            road_id = get_road_ids_from_incLanes(junction.get('incLanes'))[0]

            # Find all connections where from=road_id and dir='t'
            connections_to_delete = root.findall(
                f".//connection[@from='{road_id}'][@dir='t']")
            # Delete these connections
            for connection in connections_to_delete:
                root.remove(connection)

        # Write the modified tree back to the file
        tree.write(self.Network)

    def importDemand(self, ConcreteScn, SimulationStartTime, SimulationEndTime, SeedSet):
        """The function to import the demand from the demand file and convert it to SUMO demand file."""

        NetworkName = ConcreteScn.Supply.NetworkName

        # create turn and inflow and summary df
        path_matchup_table = pf.path2linux(Path(ConcreteScn.input_config["input_dir"]) / "MatchupTable.xlsx")
        traffic_dir = pf.path2linux(Path(ConcreteScn.input_config["input_dir"]) / "Traffic")
        path_net_SUMO = pf.path2linux(Path(ConcreteScn.input_config["input_dir"]) / f"output/SUMO/{NetworkName}.net.xml")
        MatchupTable_UserInput = read_MatchupTable(path_matchup_table=path_matchup_table)
        TurnDf, IDRef = generate_turn_demand_cali(path_matchup_table=path_matchup_table, traffic_dir=traffic_dir)
        InflowDf_Calibration, InflowEdgeToCalibrate, N_InflowVariable = generate_inflow(path_net_SUMO,
                                                                                        MatchupTable_UserInput,
                                                                                        TurnDf,
                                                                                        IDRef)
        (TurnToCalibrate, TurnDf_Calibration,
        RealSummary_Calibration,
        N_Variable, N_TurnVariable) = generate_turn_summary(TurnDf,
                                                            MatchupTable_UserInput,
                                                            N_InflowVariable)
        InflowDf = InflowDf_Calibration.copy()
        TurnDf = TurnDf_Calibration.copy()
        TurnDf = TurnDf[TurnDf['IntersectionName'].notna()]

        # Create the .flow.xml
        # InflowDf = ConcreteScn.Demand.Inflow
        InflowDf['IntervalStart'] = InflowDf['IntervalStart'].astype(float)
        InflowDf['IntervalEnd'] = InflowDf['IntervalEnd'].astype(float)
        InflowDf = InflowDf[(InflowDf['IntervalStart'] >= SimulationStartTime)
                            & (InflowDf['IntervalEnd'] <= SimulationEndTime)]

        routes = ET.Element('routes')
        v_type = ET.SubElement(routes, 'vType')
        v_type.set('id', 'car')
        v_type.set('type', 'passenger')
        InflowDict = InflowDf.to_dict(orient='records')
        FlowID = 0
        for InflowData in InflowDict:
            FlowID += 1
            flow = ET.SubElement(routes, 'flow')
            flow.set('id', str(FlowID))
            flow.set('begin', str(InflowData['IntervalStart']))
            flow.set('end', str(InflowData['IntervalEnd']))
            flow.set('from', str(-int(InflowData['OpenDriveFromID'])))

            # may need to change
            flow.set('number', str(InflowData['Count']))
            flow.set('type', 'car')

        # <flow begin="0.0" end="3600.0" from="" id="" number="" type="car"/>
        TreeInflow = ET.ElementTree(routes)
        path_sumo_flow = pf.path2linux(os.path.join(self.SUMOPath, f'{NetworkName}.flow.xml'))
        TreeInflow.write(path_sumo_flow,
                         encoding='utf-8', xml_declaration=True)

        # Create the .turn.xml
        # TurnDf = ConcreteScn.Route.TurningRatio
        TurnDf['IntervalStart'] = TurnDf['IntervalStart'].astype(float)
        TurnDf['IntervalEnd'] = TurnDf['IntervalEnd'].astype(float)
        TurnDf = TurnDf[(TurnDf['IntervalStart'] >= SimulationStartTime) & (
            TurnDf['IntervalEnd'] <= SimulationEndTime)]
        turns = ET.Element('turns')
        # Create the 'interval' element
        IntervalSet = TurnDf[['IntervalStart', 'IntervalEnd']
                             ].drop_duplicates().reset_index(drop=True)
        for _, IntervalData in IntervalSet.iterrows():
            Interval = ET.SubElement(turns, 'interval')
            Interval.set('begin', str(IntervalData['IntervalStart']))
            Interval.set('end', str(IntervalData['IntervalEnd']))

            TurnDfSubset = TurnDf[(TurnDf['IntervalStart'] == IntervalData['IntervalStart'])
                                  & (TurnDf['IntervalEnd'] == IntervalData['IntervalEnd'])]
            TurnDictSubset = TurnDfSubset.to_dict(orient='records')
            for TurnData in TurnDictSubset:
                edge_relation = ET.SubElement(Interval, 'edgeRelation')
                edge_relation.set(
                    'from', str(-int(TurnData['OpenDriveFromID'])))

                # may need to change
                edge_relation.set('to', str(-int(TurnData['OpenDriveToID'])))

                # may need to change
                edge_relation.set('probability', str(TurnData['TurnRatio']))
        # <edgeRelation from="" probability="" to=""/>
        TreeTurn = ET.ElementTree(turns)
        # Write the XML to the file
        path_sumo_turn = pf.path2linux(os.path.join(self.SUMOPath, f'{NetworkName}.turn.xml'))
        TreeTurn.write(path_sumo_turn, encoding='utf-8', xml_declaration=True)

        for Seed in SeedSet:
            # Create the .rou.xml for each random seed
            path_sumo_demand = pf.path2linux(
                os.path.join(self.SUMOPath, f'{NetworkName}.rou.xml'))

            os.system(f'cmd/c "jtrrouter -r {path_sumo_flow}'
                      f' -t {path_sumo_turn}'
                      f' -n {self.Network} --accept-all-destinations'
                      f' --remove-loops True --randomize-flows --seed {Seed}'
                      f' -o {path_sumo_demand}"')

            # add element to the set object
            self.Demand.add(path_sumo_demand)

    def generateConfig(self, ConcreteScn, SimulationStartTime, SimulationEndTime, SeedSet, StepLength):
        """The function to generate the SUMO configuration file for the simulation."""

        def prettify(elem):
            """Return a pretty-printed XML string for the Element."""
            rough_string = ET.tostring(elem, 'utf-8')
            re_parsed = parseString(rough_string)
            return re_parsed.toprettyxml(indent="    ")
        # Python code to generate 10 XML files with different seeds using xml.etree.ElementTree as ET

        # Create the root element
        root = ET.Element('configuration')
        root.set('xmlns:xsi', 'http://www.w3.org/2001/XMLSchema-instance')
        root.set('xsi:noNamespaceSchemaLocation',
                 'http://sumo.dlr.de/xsd/duarouterConfiguration.xsd')

        NetworkName = ConcreteScn.Supply.NetworkName
        # Add other elements to the root
        for Seed in SeedSet:
            random = ET.SubElement(root, 'random')
            ET.SubElement(random, 'seed', {'value': f'{Seed}'})

            input_val = ET.SubElement(root, 'input')
            ET.SubElement(input_val, 'net-file',
                          {'value': f'{NetworkName}.net.xml'})
            ET.SubElement(input_val, 'route-files',
                          {'value': f'{NetworkName}.rou.xml'})

            ET.SubElement(root, 'output')

            time = ET.SubElement(root, 'time')
            ET.SubElement(time, 'begin', {
                          'value': f'{SimulationStartTime}'})
            ET.SubElement(
                time, 'end', {'value': f'{SimulationEndTime}'})
            ET.SubElement(time, 'step-length',
                          {'value': f'{StepLength}'})

            gui_only = ET.SubElement(root, 'gui_only')
            ET.SubElement(gui_only, 'start', {'value': 't'})

            report = ET.SubElement(root, 'report')
            ET.SubElement(report, 'no-warnings', {'value': 'true'})
            ET.SubElement(report, 'no-step-log', {'value': 'true'})

            # Update the seed value

            xml_string = prettify(root)
            # Write the XML string to a file

            path_sumo_cfg = pf.path2linux(os.path.join(self.SUMOPath, f'{NetworkName}.sumocfg'))
            with open(path_sumo_cfg, 'w', encoding="utf-8") as file:
                file.write(xml_string)

        # 10 XML files 'config_1.smocfg' to 'config_10.smocfg' are created with different seed values

    def importSignal(self, ConcreteScn):
        """The function to import the signal from the signal file and convert it to SUMO signal file."""

        input_dir = ConcreteScn.input_config.get('input_dir')
        path_output = ConcreteScn.input_config.get('output_dir')
        control_dir = pf.path2linux(Path(input_dir) / 'Control')
        path_MatchupTable = pf.path2linux(os.path.join(input_dir, 'MatchupTable.xlsx'))
        path_net = pf.path2linux(Path(path_output) / f'SUMO/{ConcreteScn.Supply.NetworkName}.net.xml')

        # check if the file exists
        if not os.path.exists(path_net):
            raise FileNotFoundError(f"File not found: {path_net}")
        if not os.path.exists(path_MatchupTable):
            raise FileNotFoundError(f"File not found: {path_MatchupTable}")
        if not path_MatchupTable.endswith(".xlsx"):
            raise ValueError(f"Invalid file format: {path_MatchupTable}. Expected .xlsx file.")
        if not os.path.exists(control_dir):
            raise FileNotFoundError(f"File not found: {control_dir}")

        FixedTime = self.kwargs.get('FixedTime', False)

        try:
            signal_flag = sumo_signal_import(path_net=path_net, path_MatchupTable=path_MatchupTable,
                                             FixedTime=FixedTime, control_dir=control_dir)
            if signal_flag:
                print(f"  :SUMO signal updated at: {path_net}")
        except Exception as e:
            raise Exception(f"Error in importing SUMO signal: {e}")


def process_signal_data(path_signal: str) -> dict:

    with open(path_signal, 'r') as file:
        SignalData = file.readlines()

    SignalDict = {}
    current_table = None
    current_table_data = []

    removeflag = 0
    for line in SignalData:
        line = line.strip()

        if removeflag == 1:
            removeflag = 0
            continue
        if line.startswith("["):
            removeflag = 1
            if current_table is None:
                current_table = line[1:-1].split(',')[0].rstrip(']')
            else:
                if current_table_data:
                    df = pd.read_csv(io.StringIO('\n'.join(current_table_data)), dtype=str)
                    SignalDict[current_table] = df

                current_table = line[1:-1].split(',')[0].rstrip(']')
                current_table_data = []
        else:
            current_table_data.append(line)

    if current_table_data:
        SignalDict[current_table] = pd.read_csv(io.StringIO('\n'.join(current_table_data)), dtype=str)

    return SignalDict


def generate_complete_state_string(protected_mov, permitted_mov, rtor_mov, nm):
    state = ['r'] * nm
    if rtor_mov:
        for idx in map(int, rtor_mov.split(',')):
            if idx < nm and state[idx] == 'r':
                state[idx] = 's'
    if protected_mov:
        for idx in map(int, protected_mov.split(',')):
            if idx < nm:
                state[idx] = 'G'
    if permitted_mov:
        for idx in map(int, permitted_mov.split(',')):
            if idx < nm and (state[idx] == 'r' or state[idx] == 's'):
                state[idx] = 'g'
    return ''.join(state)


# format the XML file to make it more readable
def prettify_xml(file_path):
    with open(file_path, 'r') as file:
        xml_content = file.read()
    parsed_xml = minidom.parseString(xml_content)
    pretty_xml_as_string = parsed_xml.toprettyxml(indent="    ")

    with open(file_path, 'w') as file:
        file.write("\n".join([line for line in pretty_xml_as_string.splitlines() if line.strip()]))


def sumo_signal_import(path_net: str, path_MatchupTable: str, FixedTime: bool = False, control_dir: str = "Control") -> bool:
    """Import SUMO signal data from a MatchupTable to a SUMO network file.
    This function reads the signal data from the MatchupTable and updates the signal to the SUMO network file.

    Args:
        path_net (str): SUMO network file path.
        path_MatchupTable (str): Path to the MatchupTable file.
        FixedTime (bool): If True, the signal is fixed time, otherwise it is actuated. Default is False.

    Returns:
        bool: True if the import was successful, False otherwise.
    """

    # check if the file exists
    if not os.path.exists(path_net):
        raise FileNotFoundError(f"File not found: {path_net}")
    if not os.path.exists(path_MatchupTable):
        raise FileNotFoundError(f"File not found: {path_MatchupTable}")
    if not path_MatchupTable.endswith(".xlsx"):
        raise ValueError(f"Invalid file format: {path_MatchupTable}. Expected .xlsx file.")

    MatchupTable_UserInput = pd.read_excel(path_MatchupTable, skiprows=1, dtype=str)
    merged_columns = ["JunctionID_OpenDrive", "File_GridSmart", "Date_GridSmart",
                      "IntersectionName_GridSmart", "File_Synchro", "IntersectionID_Synchro", "Need calibration?"]
    MatchupTable_UserInput[merged_columns] = MatchupTable_UserInput[merged_columns].ffill()

    # Create synchro lookup table
    lookup_df = pd.DataFrame()
    lookup_df['INTID'] = MatchupTable_UserInput['IntersectionID_Synchro'].dropna().unique()
    lookup_df['SumoJunctionID'] = lookup_df['INTID'].apply(
        lambda intid: MatchupTable_UserInput.loc[
            MatchupTable_UserInput['IntersectionID_Synchro'] == intid, 'JunctionID_OpenDrive'
        ].dropna().iloc[0]
    )
    lookup_df['StartingBound'] = lookup_df['INTID'].apply(
        lambda intid: MatchupTable_UserInput.loc[
            (MatchupTable_UserInput['IntersectionID_Synchro'] == intid) &
            (MatchupTable_UserInput['Bearing'].astype(float) >= 180),
            'Turn_Synchro'
        ].dropna().iloc[0]
    )
    lookup_df = lookup_df.astype(str)

    synchro_file = MatchupTable_UserInput['File_Synchro'].dropna().iloc[0].strip()
    signal_path = pf.path2linux(Path(control_dir) / synchro_file)
    SignalDict = process_signal_data(signal_path)

    SignalInfo = {}
    unique_INTIDs = SignalDict['Lanes']['INTID'].dropna().unique()
    # Iterating through each unique 'INTID' and gathering corresponding data from each table
    for intid in unique_INTIDs:
        SignalInfo[intid] = {
            'Lanes': SignalDict['Lanes'][SignalDict['Lanes']['INTID'] == intid],
            'Timeplans': SignalDict['Timeplans'][SignalDict['Timeplans']['INTID'] == intid],
            'Phases': SignalDict['Phases'][SignalDict['Phases']['INTID'] == intid]
        }

    Synchro = {}
    for intid in unique_INTIDs:
        phases_df = SignalInfo[intid]['Phases']
        transposed_df = phases_df.set_index('RECORDNAME').transpose().reset_index()
        transposed_df.rename(columns={'index': 'Phase'}, inplace=True)
        Synchro[intid] = transposed_df
    # Filter Synchro to only keep entries where intid is in lookup_df['INTID']
    Synchro = {intid: df for intid, df in Synchro.items() if intid in lookup_df['INTID'].values}

    # Updating the Synchro dictionary
    for intid in Synchro:
        if 'RECORDNAME' in Synchro[intid].columns:
            Synchro[intid].drop(columns=['RECORDNAME'], inplace=True)

        Synchro[intid] = Synchro[intid][Synchro[intid]['Phase'] != 'INTID']

        if 'BRP' in Synchro[intid].columns:
            df_temp = Synchro[intid]

            df_temp = df_temp.dropna(subset=['BRP'])

            cols_to_check = [col for col in df_temp.columns if col not in ['Phase', 'BRP']]
            df_temp = df_temp[~(df_temp['BRP'].notna() & df_temp[cols_to_check].isna().all(axis=1))]

            if 'MinGreen' in df_temp.columns and 'MaxGreen' in df_temp.columns:
                df_temp = df_temp.dropna(subset=['MinGreen', 'MaxGreen'], how='all')
            Synchro[intid] = df_temp

        Synchro[intid]['Phase'] = Synchro[intid]['Phase'].str.replace('D', '', regex=False)

        brp_values = Synchro[intid]['BRP'].astype(str)

        Synchro[intid]['Barrier'] = brp_values.str[0].astype(int)
        Synchro[intid]['Ring'] = brp_values.str[1].astype(int)
        Synchro[intid]['Position'] = brp_values.str[2].astype(int)

    # Adjusting the Synchro dictionary to place each table under Synchro[intid]["Phases"]
    for intid in list(Synchro.keys()):
        current_df = Synchro[intid]
        Synchro[intid] = {"Phases": current_df}

    for intid in Synchro.keys():
        lanes_df = SignalDict['Lanes'][SignalDict['Lanes']['INTID'] == intid]
        timeplans_df = SignalDict['Timeplans'][SignalDict['Timeplans']['INTID'] == intid]
        Synchro[intid]["Lanes"] = lanes_df
        Synchro[intid]["Timeplans"] = timeplans_df

    # Extracting the new order based on the 'StartingBound' value
    for intid in Synchro.keys():
        cyclic_order = ['SBR', 'SBT', 'SBL', 'SWR', 'SWT', 'SWL', 'WBR', 'WBT', 'WBL',
                        'NWR', 'NWT', 'NWL', 'NBR', 'NBT', 'NBL', 'NER', 'NET', 'NEL',
                        'EBR', 'EBT', 'EBL', 'SER', 'SET', 'SEL']

        if str(intid) in lookup_df['INTID'].values:
            starting_bound = lookup_df.loc[lookup_df['INTID'] == str(intid), 'StartingBound'].values[0]
            if starting_bound.endswith('U'):
                print(f"  :There is mismatch between SUMO turn and Synchro turn for Synchro junction {intid}")
                starting_bound = starting_bound[:-1] + 'L'
            start_index = cyclic_order.index(starting_bound)
            ordered_columns = cyclic_order[start_index:] + cyclic_order[:start_index]
        else:
            ordered_columns = cyclic_order.copy()

        final_column_order = ['RECORDNAME', 'INTID'] + ordered_columns
        existing_columns = [col for col in final_column_order if col in Synchro[intid]["Lanes"].columns]
        Synchro[intid]["Lanes"] = Synchro[intid]["Lanes"][existing_columns].copy()
        Synchro[intid]["Lanes"].dropna(axis=1, how='all', inplace=True)

    # Associating Synchro[intid]["Phases"] with Synchro[intid]["Lanes"] by creating "Protected" and "Permitted" columns
    for intid in Synchro.keys():
        phases_df = Synchro[intid]["Phases"]
        lanes_df = Synchro[intid]["Lanes"]
        phases_df['Protected'] = ''
        phases_df['Permitted'] = ''
        phases_df['RTOR'] = ''

        protected_rows = ['Phase1', 'Phase2', 'Phase3', 'Phase4']
        permitted_rows = ['PermPhase1', 'PermPhase2', 'PermPhase3', 'PermPhase4']
        existing_protected_rows = lanes_df[lanes_df['RECORDNAME'].isin(protected_rows)]
        existing_permitted_rows = lanes_df[lanes_df['RECORDNAME'].isin(permitted_rows)]
        allow_rtor_row = lanes_df[lanes_df['RECORDNAME'] == 'Allow RTOR']

        for phase_idx, phase_row in phases_df.iterrows():
            P = phase_row['Phase']
            protected_columns = []
            for _, protected_row in existing_protected_rows.iterrows():
                columns_with_P = protected_row[protected_row == P].index.tolist()
                if 'INTID' in columns_with_P:
                    columns_with_P.remove('INTID')
                protected_columns.extend(columns_with_P)

            if protected_columns:
                phases_df.at[phase_idx, 'Protected'] = ','.join(protected_columns)

            permitted_columns = []
            for _, permitted_row in existing_permitted_rows.iterrows():
                columns_with_P = permitted_row[permitted_row == P].index.tolist()
                if 'INTID' in columns_with_P:
                    columns_with_P.remove('INTID')
                permitted_columns.extend(columns_with_P)

            if permitted_columns:
                phases_df.at[phase_idx, 'Permitted'] = ','.join(permitted_columns)

            RTOR_columns = []
            if not allow_rtor_row.empty:
                for _, rtor_row in allow_rtor_row.iterrows():
                    rtor_columns = rtor_row[rtor_row == "1"].index.tolist()
                    if 'INTID' in rtor_columns:
                        rtor_columns.remove('INTID')
                    RTOR_columns.extend(rtor_columns)
            RTOR_columns = [col for col in RTOR_columns if col.endswith('R')]
            if RTOR_columns:
                phases_df.at[phase_idx, 'RTOR'] = ','.join(RTOR_columns)

        Synchro[intid]["Phases"] = phases_df

        for idx, thru in Synchro[intid]["Phases"]["Protected"].items():
            movements = thru.split(",")
            for movement in movements:
                if movement.endswith("T"):
                    if movement in Synchro[intid]["Lanes"].columns:
                        lane_value = Synchro[intid]["Lanes"].loc[Synchro[intid]["Lanes"]["RECORDNAME"] == "Shared", movement].values

                        if len(lane_value) > 0:
                            lane_value = lane_value[0]
                        else:
                            continue

                        if pd.isna(lane_value) or lane_value == "0":
                            continue
                        left_turn = movement[:-1] + "L"
                        right_turn = movement[:-1] + "R"

                        turns_to_add = []
                        if lane_value == "1" and left_turn not in movements:
                            turns_to_add.append(left_turn)
                        elif lane_value == "2" and right_turn not in movements:
                            if right_turn in Synchro[intid]["Phases"]["RTOR"].at[idx]:
                                Synchro[intid]["Phases"]["RTOR"].at[idx] = Synchro[intid]["Phases"]["RTOR"].at[idx].replace(right_turn, "").strip(",")
                            turns_to_add.append(right_turn)

                        elif lane_value == "3":
                            if movement not in movements:
                                turns_to_add.append(movement)
                            if right_turn not in movements:
                                if right_turn in Synchro[intid]["Phases"]["RTOR"].at[idx]:
                                    Synchro[intid]["Phases"]["RTOR"].at[idx] = Synchro[intid]["Phases"]["RTOR"].at[idx].replace(right_turn, "").strip(",")
                                turns_to_add.append(right_turn)
                        if turns_to_add:
                            Synchro[intid]["Phases"]["Protected"].at[idx] = ",".join(movements + turns_to_add)
                    Synchro[intid]["Phases"]["RTOR"].at[idx] = Synchro[intid]["Phases"]["RTOR"].at[idx].replace(",,", ",").strip(",")

    with open(path_net, 'r') as file:
        NetworkData = file.read()

    tree = ET.ElementTree(ET.fromstring(NetworkData))
    TLLogic = {}
    for elem in tree.iter('tlLogic'):
        TLLogic[elem.attrib['id']] = {}

    for elem in tree.iter('connection'):
        if 'tl' in elem.attrib:
            tl = elem.attrib['tl']
            linkIndex = int(elem.attrib['linkIndex'])
            if tl in TLLogic:
                TLLogic[tl][linkIndex] = {'from': elem.attrib['from'], 'to': elem.attrib['to'], 'dir': elem.attrib['dir']}

    for tl in TLLogic:
        TLLogic[tl] = dict(sorted(TLLogic[tl].items()))

    # ## Please note: here we assume there is no phase only for U turn,
    # i.e. U turn is associated with a left turn or through movement
    lookup_mapping = lookup_df[['INTID', 'SumoJunctionID']]

    # Adding the "SumoJunctionID" attribute to each Synchro[intid]
    for _, row in lookup_mapping.iterrows():
        intid = row['INTID']
        tl = row['SumoJunctionID']
        Synchro[intid]["SumoJunctionID"] = tl

    # Matching Synchro[intid]["Lanes"] with TLLogic[tl] with additional checks
    for _, row in lookup_mapping.iterrows():
        intid = row['INTID']
        tl = row['SumoJunctionID']

        if tl in TLLogic:
            current_tllogic = TLLogic[tl]
            movement_values = []
            current_dir = None
            dir_sequence = []
            column_index = 0
            for index, values in current_tllogic.items():
                movement_values = []
                column_index = 0

                # Group by 'from'
                from_groups = {}
                for index, values in current_tllogic.items():
                    from_edge = values['from']
                    direction = values['dir']
                    from_groups.setdefault(from_edge, {'r': [], 's': [], 'l': []})  # no separate 't' key now
                    if direction == 't':
                        from_groups[from_edge]['l'].append(str(index))  # treat 't' as 'l'
                    else:
                        from_groups[from_edge][direction].append(str(index))

                for from_edge in from_groups:
                    for dir_key in ['r', 's', 'l']:
                        if from_groups[from_edge][dir_key]:
                            if column_index >= len(Synchro[intid]["Lanes"].columns) - 2:
                                print(f"  :SUMO junction (id = {tl}) has more movements than Synchro intersection (id = {intid}). Please check.")
                                break
                            movement_values.append(','.join(from_groups[from_edge][dir_key]))
                            column_index += 1

            # If there are remaining columns, print the message indicating the mismatch
            if column_index < len(Synchro[intid]["Lanes"].columns) - 3:
                print(f"  :Synchro intersection (id = {intid}) has more movements than SUMO junction (id = {tl}). Please check.")

            # Ensure that the number of columns matches the existing columns in Synchro[intid]["Lanes"]
            while len(movement_values) < len(Synchro[intid]["Lanes"].columns) - 2:
                movement_values.append('')

            # Create a new row with "Movement" as RECORDNAME, intid as INTID, and TLLogic values
            new_movement_row = pd.DataFrame([['Movement', intid] + movement_values], columns=Synchro[intid]["Lanes"].columns)
            Synchro[intid]["Lanes"] = pd.concat([Synchro[intid]["Lanes"], new_movement_row], ignore_index=True)

    # Matching "Movement" of Synchro[intid]["Lanes"] with "Protected" and "Permitted" in Synchro[intid]["Phases"]
    phase_flag = 0
    error_msg = ""
    for intid in Synchro.keys():
        try:
            phases_df = Synchro[intid]["Phases"]
            lanes_df = Synchro[intid]["Lanes"]
            phases_df['ProtectedMovement'] = ''
            phases_df['PermittedMovement'] = ''
            phases_df['RTORMovement'] = ''
            movement_mapping = {}
            movement_row = lanes_df[lanes_df['RECORDNAME'] == 'Movement'].iloc[0]
            for column in lanes_df.columns[2:]:
                movement_mapping[column] = movement_row[column] if not pd.isna(movement_row[column]) else ''

            for phase_idx, phase_row in phases_df.iterrows():
                protected_columns = phase_row['Protected'].split(',')
                protected_movements = [movement_mapping.get(col, '') for col in protected_columns if col in movement_mapping]
                protected_movements = [mv for mv in protected_movements if mv]
                phases_df.at[phase_idx, 'ProtectedMovement'] = ','.join(protected_movements)
                # For "PermittedMovement", extract and map the columns from "Permitted"
                permitted_columns = phase_row['Permitted'].split(',')
                permitted_movements = [movement_mapping.get(col, '') for col in permitted_columns if col in movement_mapping]
                permitted_movements = [mv for mv in permitted_movements if mv]
                phases_df.at[phase_idx, 'PermittedMovement'] = ','.join(permitted_movements)

                # For "RTORMovement", extract and map the columns from "RTOR"
                RTOR_columns = phase_row['RTOR'].split(',')
                RTOR_movements = [movement_mapping.get(col, '') for col in RTOR_columns if col in movement_mapping]
                RTOR_movements = [mv for mv in RTOR_movements if mv]  # Remove any empty strings
                phases_df.at[phase_idx, 'RTORMovement'] = ','.join(RTOR_movements)

            # Update the "Phases" DataFrame in Synchro
            phases_df['MinGreen'] = pd.to_numeric(phases_df['MinGreen'], errors='coerce')
            phases_df['MaxGreen'] = pd.to_numeric(phases_df['MaxGreen'], errors='coerce')
            phases_df = phases_df[~((phases_df['MinGreen'].fillna(0) == 0) & (phases_df['MaxGreen'].fillna(0) == 0))]

            Synchro[intid]["Phases"] = phases_df
            phase_flag += 1
        except Exception as e:
            phase_flag = True
            error_msg = str(e)
            # get sumo junction id form lookup_df
            sumo_id = lookup_df.loc[lookup_df['INTID'] == intid, 'SumoJunctionID'].values[0]
            print(f"  :Mismatch between SUMO junction {sumo_id} and "
                  f"Synchro junction {intid} in the input MatchupTable.xlsx. \n    : {error_msg}")
    # # Uncomment this section if you want to raise an error when no phases are found
    # if phase_flag:
    #     raise Exception(error_msg)
    # No matching phases found
    if phase_flag == 0:
        raise ValueError("No phases found in Synchro data.")
    elif phase_flag >= 0:
        # Not all phases matched
        if phase_flag != len(Synchro):
            print(f"  :Total {phase_flag}/{len(Synchro)} signal intersections.")

    tree = ET.parse(path_net)
    root = tree.getroot()
    last_edge = root.findall('edge')[-1]

    for tlLogic_elem in root.findall('tlLogic'):
        tl_id = tlLogic_elem.attrib['id']
        intid_row = lookup_df[lookup_df['SumoJunctionID'] == tl_id]
        if intid_row.empty:
            continue

        intid = intid_row['INTID'].values[0]
        synchro_data = Synchro[intid]
        offset = synchro_data["Timeplans"].loc[synchro_data["Timeplans"]['RECORDNAME'] == 'Offset', 'DATA'].values[0]
        detect_lengths = synchro_data["Lanes"].loc[synchro_data["Lanes"]['RECORDNAME'] == 'DetectSize1']
        detect_lengths_float = detect_lengths.iloc[:, 2:].apply(pd.to_numeric, errors='coerce')
        detector_length = detect_lengths_float.max(axis=1, skipna=True).max(skipna=True) * 0.3048 if not detect_lengths.empty else ''
        detector_length_left_turn = detect_lengths_float.min(axis=1, skipna=True).min(skipna=True) * 0.3048 if not detect_lengths.empty else ''

        total_cycle_length = synchro_data["Timeplans"].loc[synchro_data["Timeplans"]['RECORDNAME'] == 'Cycle Length', 'DATA'].values[0]
        new_tlLogic = ET.Element('tlLogic', id=tl_id, offset=str(offset), programID="NEMA", type="NEMA")

        # Add the param elements based on gathered data
        params = {
            "detector-length": str(detector_length),
            "detector-length-leftTurnLane": str(detector_length_left_turn),
            "total-cycle-length": str(total_cycle_length),
            "coordinate-mode": "true" if synchro_data["Timeplans"].loc[synchro_data["Timeplans"]['RECORDNAME'] == 'Control Type', 'DATA'].values[0] == '3' else "false",
            "whetherOutputState": "true",
            "show-detectors": "true",
            "controllerType": "Type 170" if synchro_data["Timeplans"].loc[synchro_data["Timeplans"]['RECORDNAME'] == 'Referenced To', 'DATA'].values[0] in ['1', '4'] else "TS2",
            "fixForceOff": "true" if synchro_data["Phases"]["InhibitMax"].astype(str).str.contains('1').any() else "false"
        }

        # Add the 'coordinatePhases' parameter conditionally
        if params["coordinate-mode"] == "true":
            reference_phase = synchro_data["Timeplans"].loc[synchro_data["Timeplans"]['RECORDNAME'] == 'Reference Phase', 'DATA'].values[0]
            RP = int(reference_phase)
            if RP <= 99:
                coordinate_phases_value = str(RP)
            else:
                coordinate_phases_value = f"{RP // 100},{RP - (RP // 100) * 100}"
            params["coordinatePhases"] = coordinate_phases_value

        # Add the 'minRecall' parameter
        min_recall_subset = synchro_data["Phases"][synchro_data["Phases"]["Recall"] == '1']
        min_recall_value = ",".join(min_recall_subset["Phase"].astype(str)) if not min_recall_subset.empty else ""
        params["minRecall"] = min_recall_value
        if FixedTime == 1:
            params["minRecall"] = ""

        # Add the 'maxRecall' parameter
        max_recall_subset = synchro_data["Phases"][synchro_data["Phases"]["Recall"] == '3']
        max_recall_value = ",".join(max_recall_subset["Phase"].astype(str)) if not max_recall_subset.empty else ""
        params["maxRecall"] = max_recall_value
        if FixedTime == 1:
            params["maxRecall"] = ",".join(synchro_data["Phases"]["Phase"])

        for key, value in params.items():
            ET.SubElement(new_tlLogic, 'param', key=key, value=value)

        # Adding <param key="ringX" value=""/> with the updated logic
        # max_ring = synchro_data["Phases"]["Ring"].astype(int).max()
        # previous_phase = 1
        # for ring_num in range(1, max_ring + 1):
        #     phases_in_ring = synchro_data["Phases"][synchro_data["Phases"]["Ring"].astype(int) == ring_num]["Phase"].astype(int).tolist()
        #     if not phases_in_ring:
        #         continue
        #     largest_phase = max(phases_in_ring)
        #     complete_list = []
        #     for i in range(previous_phase, largest_phase + 1):
        #         if i in phases_in_ring:
        #             complete_list.append(i)
        #         else:
        #             complete_list.append(0)
        #     previous_phase = largest_phase + 1
        #     ring_value = ",".join(map(str, complete_list))
        #     ET.SubElement(new_tlLogic, 'param', key=f"ring{ring_num}", value=ring_value)

        # Adding <param key="ring1" value=""/>  <param key="ring2" value=""/>
        # R1B1
        r1b1 = synchro_data["Phases"][(synchro_data["Phases"]['Ring'] == 1) & (synchro_data["Phases"]['Barrier'] == 1)]
        if not r1b1.empty:
            max_pos = r1b1['Position'].max()
            R1B1 = [0] * max_pos
            for _, row in r1b1.iterrows():
                R1B1[int(row['Position']) - 1] = int(row['Phase'])
        else:
            R1B1 = None

        # R1B2
        r1b2 = synchro_data["Phases"][(synchro_data["Phases"]['Ring'] == 1) & (synchro_data["Phases"]['Barrier'] == 2)]
        if not r1b2.empty:
            max_pos = r1b2['Position'].max()
            R1B2 = [0] * max_pos
            for _, row in r1b2.iterrows():
                R1B2[int(row['Position']) - 1] = int(row['Phase'])
        else:
            R1B2 = None

        # R2B1
        r2b1 = synchro_data["Phases"][(synchro_data["Phases"]['Ring'] == 2) & (synchro_data["Phases"]['Barrier'] == 1)]
        if not r2b1.empty:
            max_pos = r2b1['Position'].max()
            R2B1 = [0] * max_pos
            for _, row in r2b1.iterrows():
                R2B1[int(row['Position']) - 1] = int(row['Phase'])
        else:
            R2B1 = None

        # R2B2
        r2b2 = synchro_data["Phases"][(synchro_data["Phases"]['Ring'] == 2) & (synchro_data["Phases"]['Barrier'] == 2)]
        if not r2b2.empty:
            max_pos = r2b2['Position'].max()
            R2B2 = [0] * max_pos
            for _, row in r2b2.iterrows():
                R2B2[int(row['Position']) - 1] = int(row['Phase'])
        else:
            R2B2 = None

        # max_barrier = synchro_data["Phases"]["Barrier"].astype(int).max()
        # for barrier_num in range(1, max_barrier + 1):
        #     if barrier_num == 1:
        #         # Find phases with the largest "Barrier" value
        #         largest_barrier_row = synchro_data["Phases"][synchro_data["Phases"]["Barrier"].astype(int) == max_barrier]
        #         largest_phases = largest_barrier_row[largest_barrier_row["Position"].astype(int) == largest_barrier_row["Position"].astype(int).max()]["Phase"].astype(int).tolist()
        #         barrier_value = ",".join(map(str, sorted(largest_phases)))
        #         ET.SubElement(new_tlLogic, 'param', key="barrierPhases", value=barrier_value)
        #     else:
        #         # Find the previous barrier phases
        #         previous_barrier_row = synchro_data["Phases"][synchro_data["Phases"]["Barrier"].astype(int) == barrier_num - 1]
        #         largest_phases = previous_barrier_row[previous_barrier_row["Position"].astype(int) == previous_barrier_row["Position"].astype(int).max()]["Phase"].astype(int).tolist()
        #         barrier_value = ",".join(map(str, sorted(largest_phases)))
        #         ET.SubElement(new_tlLogic, 'param', key=f"barrier{barrier_num}Phases", value=barrier_value)

        # Apply fallback
        if R1B1 is None and R2B1 is not None:
            R1B1 = R2B1
        if R2B1 is None and R1B1 is not None:
            R2B1 = R1B1
        if R1B2 is None and R2B2 is not None:
            R1B2 = R2B2
        if R2B2 is None and R1B2 is not None:
            R2B2 = R1B2

        ET.SubElement(new_tlLogic, 'param', key="ring1", value=",".join(map(str, R1B1 + R1B2)))
        ET.SubElement(new_tlLogic, 'param', key="ring2", value=",".join(map(str, R2B1 + R2B2)))

        # Adding <param key="barrierPhases" value=""/>  <param key="barrier2Phases" value=""/>
        barrier2Phases = [R1B1[-1], R2B1[-1]]
        ET.SubElement(new_tlLogic, 'param', key="barrier2Phases", value=",".join(map(str, barrier2Phases)))

        # Add <param key="barrierPhases" value="R1B2_last,R2B2_last"/>
        barrierPhases = [R1B2[-1], R2B2[-1]]
        ET.SubElement(new_tlLogic, 'param', key="barrierPhases", value=",".join(map(str, barrierPhases)))

        # Check and swap barrier phases if needed
        if params["coordinate-mode"] == "true":
            # Extract the values of 'coordinatePhases', 'barrierPhases', and 'barrier2Phases'
            coordinate_phases_value = params.get("coordinatePhases", "")
            barrier_phases_value = new_tlLogic.find("./param[@key='barrierPhases']").attrib['value']
            barrier2_phases_element = new_tlLogic.find("./param[@key='barrier2Phases']")

            if barrier2_phases_element is not None:
                barrier2_phases_value = barrier2_phases_element.attrib['value']

                # Check if 'barrier2Phases' is the same as 'coordinatePhases'
                if barrier2_phases_value != coordinate_phases_value:
                    new_tlLogic.find("./param[@key='barrierPhases']").attrib['value'] = barrier2_phases_value
                    barrier2_phases_element.attrib['value'] = barrier_phases_value

                # Recheck if 'barrier2Phases' is still not the same as 'coordinatePhases'
                if barrier2_phases_element.attrib['value'] != coordinate_phases_value:
                    print("  :Error with barrier phases and coordinated phases at "
                          f"intersection {intid_row['INTID'].values[0]}, please modify manually.")

        nm = len(TLLogic[tl_id])
        for _, phase_row in synchro_data["Phases"].iterrows():
            state = generate_complete_state_string(phase_row['ProtectedMovement'], phase_row['PermittedMovement'], phase_row['RTORMovement'], nm)
            # maxDur = int(float(phase_row['MaxGreen'])) if pd.notna(phase_row['MaxGreen']) else int(int(total_cycle_length) / 3)
            maxDur = float(phase_row["MaxGreen"]) if pd.notna(phase_row['MaxGreen']) else round(float(total_cycle_length) / 3, 1)
            if pd.isna(phase_row['MaxGreen']):
                print(f"  :Maximum green('MaxGreen') is missing for phase {phase_row['Phase']} at "
                      f"intersection {intid_row['INTID'].values[0]}, {int(int(total_cycle_length) / 3)} sec is used."
                      " Manual change is probably needed to ensure the length of this ring equal to cycle length.")

            # minDur = phase_row['MinGreen'] if pd.notna(phase_row['MinGreen']) else min(6, int(maxDur))
            minDur = phase_row['MinGreen'] if pd.notna(phase_row['MinGreen']) else min(6, float(maxDur))
            if pd.isna(phase_row['MinGreen']):
                print(f"  :Minimum green('MinGreen') is missing for phase {phase_row['Phase']} at "
                      f"intersection {intid_row['INTID'].values[0]}, {min(6, maxDur)} sec is used")

            vehext = phase_row['VehExt'] if pd.notna(phase_row['VehExt']) else 2
            if pd.isna(phase_row['VehExt']):
                print(f"  :Added green per actuation ('vehext') is missing for phase {phase_row['Phase']} at "
                      f"intersection {intid_row['INTID'].values[0]}, 2 sec is used.")

            yellow = phase_row['Yellow'] if pd.notna(phase_row['Yellow']) else 3
            if pd.isna(phase_row['Yellow']):
                print(f"  :Yellow time('Yellow') is missing for phase {phase_row['Phase']} at "
                      f"intersection {intid_row['INTID'].values[0]}, 3 sec is used.")

            red = phase_row['AllRed'] if pd.notna(phase_row['AllRed']) else 0
            if pd.isna(phase_row['AllRed']):
                print(f"  :All red time ('AllRed') is missing for phase {phase_row['Phase']} at "
                      f"intersection {intid_row['INTID'].values[0]}, 0 sec is used.")

            # Add the phase element
            ET.SubElement(new_tlLogic, 'phase', duration="99", minDur=str(minDur),
                          maxDur=str(maxDur), vehext=str(vehext),
                          yellow=str(yellow), red=str(red),
                          name=str(phase_row['Phase']), state=state)

        # Replace the existing tlLogic element with the new one in the parent element
        parent = root.find(".//tlLogic/..")
        parent.remove(tlLogic_elem)
        # Insert the new tlLogic element right after the last <edge> element
        parent.insert(list(parent).index(last_edge) + 1, new_tlLogic)

    tree.write(path_net, encoding='UTF-8')

    prettify_xml(path_net)

    return True


if __name__ == "__main__":
    # Example usage
    path_net = r"C:\Users\xh8\ornl_work\github_workspace\Real-Twin-Dev\datasets\tss\New folder\chatt.net.xml"
    path_matchup_table = r"C:\Users\xh8\ornl_work\github_workspace\Real-Twin-Dev\datasets\tss\New folder\MatchupTable.xlsx"
    control_dir = r"C:\Users\xh8\ornl_work\github_workspace\Real-Twin-Dev\datasets\tss\Control"

    sumo_signal_import(path_net=path_net, path_MatchupTable=path_matchup_table, control_dir=control_dir, FixedTime=False)