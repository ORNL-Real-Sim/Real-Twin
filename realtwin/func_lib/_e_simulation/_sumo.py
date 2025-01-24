##############################################################################
# Copyright (c) 2024, Oak Ridge National Laboratory                          #
# All rights reserved.                                                       #
#                                                                            #
# This file is part of RealTwin and is distributed under a __TBD__           #
# license. For the licensing terms see the LICENSE file in the top-level     #
# directory.                                                                 #
#                                                                            #
# Contributors (Add you name below to acknowledge your contribution):        #
# Xiangyong Roy Luo                                                          #
##############################################################################
"""The module to handle the SUMO simulation for the real-twin developed by ORNL ARMS group."""

import os
import shutil
import xml.etree.ElementTree as ET
from xml.dom.minidom import parseString
import pandas as pd
import io
from lxml import etree
import pyufunc as pf


class SUMOPrep:
    """The class to handle the SUMO simulation for the real-twin developed by ORNL ARMS group.
    """
    def __init__(self):
        self.Network = {}
        # self.NetworkWithElevation = {}
        self.Demand = set()
        self.Signal = {}

    def importNetwork(self, ConcreteScn):
        """The function to import the network from the OpenDrive file and convert it to SUMO network file."""

        # self.Network = ConcreteScn.Supply.Network
        # self.NetworkWithElevation = ConcreteScn.Supply.NetworkWithElevation
        NetworkName = ConcreteScn.Supply.NetworkName

        # get output path from the configuration dict
        path_output = ConcreteScn.config_dict.get('output_dir')

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
        # Create the .flow.xml
        InflowDf = ConcreteScn.Demand.Inflow
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
        TurnDf = ConcreteScn.Route.TurningRatio
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
        TreeTurn.write(path_sumo_turn,
                       encoding='utf-8', xml_declaration=True)

        for Seed in SeedSet:
            # Create the .rou.xml for each random seed
            path_sumo_demand = pf.path2linux(
                os.path.join(self.SUMOPath, f'{NetworkName}_Seed{Seed}.rou.xml'))

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
                          {'value': f'{NetworkName}_Seed{Seed}.rou.xml'})

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

            path_sumo_cfg = pf.path2linux(os.path.join(self.SUMOPath, f'{NetworkName}_Seed{Seed}.sumocfg'))
            with open(path_sumo_cfg, 'w', encoding="utf-8") as file:
                file.write(xml_string)

        # 10 XML files 'config_1.smocfg' to 'config_10.smocfg' are created with different seed values

    def importSignal(self, ConcreteScn):
        """The function to import the signal from the signal file and convert it to SUMO signal file."""

        path_signal = ConcreteScn.config_dict["Control"].get("Signal")
        path_signal_abs = pf.path2linux(os.path.join(ConcreteScn.config_dict.get('input_dir'), path_signal))
        with open(path_signal_abs, 'r', encoding="utf-8") as file:
            lines = file.readlines()

        SignalDict = {}
        current_table = None
        current_table_data = []

        # Iterate over the lines
        remove_flag = 0
        for line in lines:
            line = line.strip()

            # Check if it's a line to be deleted
            if remove_flag == 1:
                remove_flag = 0
                continue

            # Check if it's a table name
            if line.startswith("["):
                remove_flag = 1
                if current_table is None:
                    current_table = line[1:-1]  # Remove the square brackets
                else:
                    # Store the previous table data in the dictionary
                    SignalDict[current_table] = pd.read_csv(
                        io.StringIO('\n'.join(current_table_data)), dtype=str)

                    # Start a new table
                    current_table = line[1:-1]  # Remove the square brackets
                    current_table_data = []
            else:
                current_table_data.append(line)

        # Store the last table in the dictionary
        SignalDict[current_table] = pd.read_csv(
            io.StringIO('\n'.join(current_table_data)), dtype=str)

        path_signal_lookup = ConcreteScn.config_dict["Control"].get("Synchro_lookup")
        path_signal_lookup_abs = pf.path2linux(
            os.path.join(ConcreteScn.config_dict.get('input_dir'), path_signal_lookup))
        IDRef = pd.read_csv(path_signal_lookup_abs, dtype=str)

        # Create a dictionary mapping INTID to OpenDriveJunctionID
        IDMap = dict(zip(IDRef['INTID'], IDRef['OpenDriveJunctionID']))

        SignalDict['Links']["INTID"] = SignalDict['Links']["INTID"].astype(
            str).replace(IDMap).astype(str)
        SignalDict['Lanes']["INTID"] = SignalDict['Lanes']["INTID"].astype(
            str).replace(IDMap).astype(str)
        SignalDict['Timeplans']["INTID"] = SignalDict['Timeplans']["INTID"].astype(
            str).replace(IDMap).astype(str)
        SignalDict['Phases']["INTID"] = SignalDict['Phases']["INTID"].astype(
            str).replace(IDMap).astype(str)

        Phases = SignalDict['Phases']
        Lanes = SignalDict['Lanes']
        IDs = Phases['INTID'].unique()
        Synchro = {}
        for ID in IDs:
            Synchro[ID] = {}
            plan_data = {"phase": [],
                         "time": [],
                         "SBR": [],
                         "SBT": [],
                         "SBL": [],
                         "WBR": [],
                         "WBT": [],
                         "WBL": [],
                         "NBR": [],
                         "NBT": [],
                         "NBL": [],
                         "EBR": [],
                         "EBT": [],
                         "EBL": []}
            Synchro[ID]['plan'] = pd.DataFrame(plan_data)
            Synchro[ID]['plan'] = Synchro[ID]['plan'].astype({"phase": int,
                                                              "time": float,
                                                              "SBR": str,
                                                              "SBT": str,
                                                              "SBL": str,
                                                              "WBR": str,
                                                              "WBT": str,
                                                              "WBL": str,
                                                              "NBR": str,
                                                              "NBT": str,
                                                              "NBL": str,
                                                              "EBR": str,
                                                              "EBT": str,
                                                              "EBL": str})
            Synchro[ID]['bound'] = {}

            for i in range(1, 17):
                phase = f'D{i}'
                # Create the key before trying to access it
                Synchro[ID][phase] = {}
                if phase in Phases.columns:
                    Synchro[ID][phase]['time'] = float(Phases.loc[(Phases["RECORDNAME"] == "MaxGreen") & (
                        Phases["INTID"] == ID), f'D{i}'].values[0])
                filtered_row = Lanes[(Lanes['RECORDNAME'] == 'Phase1') & (
                    Lanes['INTID'] == ID)]
                Synchro[ID][phase]['protected'] = filtered_row.columns[(
                    filtered_row == f'{i}').any()].tolist()
                filtered_row = Lanes[(Lanes['RECORDNAME'] == 'PermPhase1') & (
                    Lanes['INTID'] == ID)]
                Synchro[ID][phase]['permitted'] = filtered_row.columns[(
                    filtered_row == f'{i}').any()].tolist()
            for i in range(1, 17):
                new_row_1 = {}
                new_row_2 = {}
                new_row_3 = {}
                phase = f'D{i}'

                for j in range(0, len(Synchro[ID][phase]['protected'])):
                    new_row_1[Synchro[ID][phase]['protected'][j]] = 'G'
                for j in range(0, len(Synchro[ID][phase]['permitted'])):
                    new_row_1[Synchro[ID][phase]['permitted'][j]] = 'g'
                if f'D{i}' in Phases.columns:
                    new_row_1['time'] = float(Phases.loc[(Phases["RECORDNAME"] == "MaxGreen") & (
                        Phases["INTID"] == ID), f'D{i}'].values[0])
                    if float(Phases.loc[(Phases["RECORDNAME"] == "Yellow")
                                        & (Phases["INTID"] == ID),
                                        f'D{i}'].values[0]) != 0:
                        for j in range(0, len(Synchro[ID][phase]['protected'])):
                            new_row_2[Synchro[ID][phase]['protected'][j]] = 'y'
                        for j in range(0, len(Synchro[ID][phase]['permitted'])):
                            new_row_2[Synchro[ID][phase]['permitted'][j]] = 'y'
                        new_row_2['time'] = float(Phases.loc[(Phases["RECORDNAME"] == "Yellow") & (
                            Phases["INTID"] == ID), f'D{i}'].values[0])
                    if float(Phases.loc[(Phases["RECORDNAME"] == "AllRed")
                                        & (Phases["INTID"] == ID),
                                        f'D{i}'].values[0]) != 0:
                        for j in range(0, len(Synchro[ID][phase]['protected'])):
                            new_row_3[Synchro[ID][phase]['protected'][j]] = 'r'
                        for j in range(0, len(Synchro[ID][phase]['permitted'])):
                            new_row_3[Synchro[ID][phase]['permitted'][j]] = 'r'
                        new_row_3['time'] = float(Phases.loc[(Phases["RECORDNAME"] == "AllRed") & (
                            Phases["INTID"] == ID), f'D{i}'].values[0])
                    new_df1 = pd.DataFrame(
                        new_row_1, index=[0], columns=Synchro[ID]['plan'].columns)
                    new_df2 = pd.DataFrame(
                        new_row_2, index=[0], columns=Synchro[ID]['plan'].columns)
                    new_df3 = pd.DataFrame(
                        new_row_3, index=[0], columns=Synchro[ID]['plan'].columns)
                    Synchro[ID]['plan'] = pd.concat(
                        [Synchro[ID]['plan'], new_df1])
                    Synchro[ID]['plan'] = pd.concat(
                        [Synchro[ID]['plan'], new_df2])
                    Synchro[ID]['plan'] = pd.concat(
                        [Synchro[ID]['plan'], new_df3])

            Synchro[ID]['plan'].dropna(how='all', inplace=True)
            Synchro[ID]['plan'].reset_index(drop=True, inplace=True)
            Synchro[ID]['plan']["phase"] = Synchro[ID]['plan'].index + 1
            if (Synchro[ID]['plan']['SBR'].isna().all()
                & Synchro[ID]['plan']['SBT'].isna().all()
                    & Synchro[ID]['plan']['SBL'].isna().all()):
                Synchro[ID]['bound'] = ['WB', 'NB', 'EB']
            elif (Synchro[ID]['plan']['WBR'].isna().all()
                  & Synchro[ID]['plan']['WBT'].isna().all()
                  & Synchro[ID]['plan']['WBL'].isna().all()):
                Synchro[ID]['bound'] = ['SB', 'NB', 'EB']
            elif (Synchro[ID]['plan']['NBR'].isna().all()
                  & Synchro[ID]['plan']['NBT'].isna().all()
                  & Synchro[ID]['plan']['NBL'].isna().all()):
                Synchro[ID]['bound'] = ['SB', 'WB', 'EB']
            elif (Synchro[ID]['plan']['EBR'].isna().all()
                  & Synchro[ID]['plan']['EBT'].isna().all()
                  & Synchro[ID]['plan']['EBL'].isna().all()):
                Synchro[ID]['bound'] = ['SB', 'WB', 'NB']
            else:
                Synchro[ID]['bound'] = ['SB', 'WB', 'NB', 'EB']

            Synchro[ID]['plan'].fillna('r', inplace=True)
            Synchro[ID]['plan'].rename(
                columns={'SBT': 'SBS', 'WBT': 'WBS', 'NBT': 'NBS', 'EBT': 'EBS'}, inplace=True)
            Synchro[ID]['plan']['SBT'] = Synchro[ID]['plan']['SBL']
            Synchro[ID]['plan']['WBT'] = Synchro[ID]['plan']['WBL']
            Synchro[ID]['plan']['NBT'] = Synchro[ID]['plan']['NBL']
            Synchro[ID]['plan']['EBT'] = Synchro[ID]['plan']['EBL']
            Synchro[ID]['plan'] = Synchro[ID]['plan'][['phase', 'time', 'SBR', 'SBS', 'SBL', 'SBT',
                                                       'WBR', 'WBS', 'WBL', 'WBT', 'NBR', 'NBS',
                                                       'NBL', 'NBT', 'EBR', 'EBS', 'EBL', 'EBT']]

        # Read the XML file

        with open(self.Network, 'r', encoding="utf-8") as file:
            data = file.read()

        # Parse the XML file
        tree = ET.ElementTree(ET.fromstring(data))

        # Initialize the dictionary
        TLLogic = {}

        # Iterate over all <tlLogic> elements
        for elem in tree.iter('tlLogic'):
            # Use the 'id' attribute as the key and create an empty dictionary as its value
            TLLogic[elem.attrib['id']] = {}

        # Iterate over all <connection> elements
        for elem in tree.iter('connection'):
            # Check if the element has a 'tl' attribute
            if 'tl' in elem.attrib:
                # Get the 'tl' attribute
                tl = elem.attrib['tl']
                # Get the 'linkIndex' attribute
                linkIndex = elem.attrib['linkIndex']
                # Check if the tl exists in TLLogic
                if tl in TLLogic:
                    # Create a dictionary for this linkIndex
                    TLLogic[tl][linkIndex] = {'app': 1, 'dir': elem.attrib['dir'], 'value': elem.find(
                        'param').attrib['value'] if elem.find('param') is not None else None}

        for tl in TLLogic:
            dd = 1
            prev_linkIndex_data = None
            # Get the sorted link indices for this tl (as integers)
            sorted_linkIndices = sorted(TLLogic[tl].keys(), key=int)
            # Iterate over each linkIndex in this tl
            for linkIndex in sorted_linkIndices:
                # If this is not the first linkIndex and the data for this linkIndex is different from the previous one
                if prev_linkIndex_data is not None and TLLogic[tl][linkIndex]['value'] != prev_linkIndex_data:
                    dd += 1
                # Update the 'app' value for this linkIndex
                TLLogic[tl][linkIndex]['app'] = dd
                # Store the current data for the next iteration
                prev_linkIndex_data = TLLogic[tl][linkIndex]['value']

        # TLLogic['10']['5']

        tl_from_data = []
        for tl in TLLogic:
            if tl in IDs:
                max_app = max(TLLogic[tl][linkIndex]['app']
                              for linkIndex in TLLogic[tl])
                if max_app == len(Synchro[tl]['bound']):
                    tl_from_data.append(tl)
                    for linkIndex in TLLogic[tl]:
                        string_index = TLLogic[tl][linkIndex]['app'] - 1
                        string = Synchro[tl]['bound'][string_index]
                        TLLogic[tl][linkIndex]['dir'] = string + \
                            TLLogic[tl][linkIndex]['dir'].upper()

        tlLogicState = {}
        for item in tl_from_data:
            tlLogicState[item] = pd.DataFrame(
                {'phase': [], 'time': [], 'state': []})

        # let's print the created dictionary

        Order = ['SBR', 'SBS', 'SBL', 'SBT', 'WBR', 'WBS', 'WBL',
                 'WBT', 'NBR', 'NBS', 'NBL', 'NBT', 'EBR', 'EBS', 'EBL', 'EBT']

        # Iterate over each linkIndex in TLLogic[tl]
        StateDirDict = {}
        for tl in tl_from_data:
            StateDir = []
            for linkIndex in TLLogic[tl]:
                # Append the 'dir' value to the list
                StateDir.append(TLLogic[tl][linkIndex]['dir'])
            StateDir = sorted(StateDir, key=Order.index)
            StateDirDict[tl] = StateDir

        for tl in tl_from_data:
            phase_plan = Synchro[tl]['plan']
            for phase in Synchro[tl]['plan']['phase']:
                StateList = phase_plan[phase_plan['phase'] == phase][StateDirDict[tl]].iloc[0]
                if 'NBR' in StateList:
                    # all right turn on red
                    StateList['NBR'] = 'g'
                if 'SBR' in StateList:
                    # all right turn on red
                    StateList['SBR'] = 'g'
                if 'EBR' in StateList:
                    # all right turn on red
                    StateList['EBR'] = 'g'
                if 'WBR' in StateList:
                    # all right turn on red
                    StateList['WBR'] = 'g'
                State = ''.join(StateList)
                PhaseTime = phase_plan[phase_plan['phase'] == phase]['time'].iloc[0]
                new_row = {'phase': phase, 'time': PhaseTime, 'state': State}
                new_row_df = pd.DataFrame(
                    new_row, index=[0], columns=tlLogicState[tl].columns)
                tlLogicState[tl] = pd.concat([tlLogicState[tl], new_row_df])
                tlLogicState[tl]['phase'] = tlLogicState[tl]['phase'].astype(
                    int)
            tlLogicState[tl].reset_index(drop=True, inplace=True)
        # tlLogicState['12']

        # Parse the XML file
        parser = etree.XMLParser(remove_blank_text=True, resolve_entities=False)
        tree = etree.parse(self.Network, parser)
        root = tree.getroot()

        # List of tlLogic ids

        # Define the namespace
        # ns = {'default': 'http://sumo.dlr.de/xsd/net_file.xsd'}

        # Iterate over each tlLogic element in the root
        for tlLogic in root.xpath('.//tlLogic'):
            # If the id attribute of the tlLogic element is in tlLogic_ids
            if tlLogic.get('id') in tl_from_data:
                id_val = tlLogic.get('id')
                phase_set = tlLogicState[id_val]
                # Remove all phase elements
                for phase in tlLogic.findall('phase'):
                    tlLogic.remove(phase)
                # Add a new phase element
                for i in range(0, len(phase_set)):
                    new_phase = etree.SubElement(tlLogic, 'phase')
                    new_phase.set('duration', str(
                        tlLogicState[f'{id_val}']['time'].iloc[i]))
                    new_phase.set(
                        'state', tlLogicState[f'{id_val}']['state'].iloc[i])
            elif len(tlLogic.findall('param')) == 0:
                for phase in tlLogic.findall('phase'):
                    if phase.get('state') == 'y' or phase.get('state') == 'r':
                        tlLogic.remove(phase)

        # Save the modified XML
        tree.write(self.Network,
                   pretty_print=True, xml_declaration=True, encoding='UTF-8')
