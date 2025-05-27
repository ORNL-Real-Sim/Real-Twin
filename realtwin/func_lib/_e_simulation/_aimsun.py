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

"""The module to prepare the Aimsun network and demand data for simulation."""

import os
from xml.etree import ElementTree as ET
import pandas as pd


class AimsunPrep:
    '''
    constructor
    '''
    def __init__(self):
        self.Network = {}
        # self.NetworkWithElevation = {}
        self.Demand = {}
        self.Signal = {}

    def is_empty(self):
        """Check if the AimsunPrep object is empty."""
        pass

    def importDemand(self, ConcreteScn, SimulationStartTime, SimulationEndTime):
        """Import demand data from the concrete scenario."""

        NetworkName = ConcreteScn.Supply.NetworkName
        AimsunPath = os.path.join('MyNetwork', 'AIMSUN')
        if os.path.exists(AimsunPath):
            os.remove(AimsunPath)  # Delete the file
        os.mkdir(AimsunPath)
        # Create the .flow.xml
        InflowDf = ConcreteScn.Demand.Inflow
        InflowDf = InflowDf[(InflowDf['IntervalStart'] >= SimulationStartTime)
                            & (InflowDf['IntervalEnd'] <= SimulationEndTime)]

        # Read the OpenDRIVE (.xodr) file
        file_path = f'MyNetwork/OpenDrive/{NetworkName}.xodr'
        with open(file_path, 'r', encoding="utf-8") as f:
            xodr_content = f.read()

        # Parse the XML content
        tree = ET.ElementTree(ET.fromstring(xodr_content))
        root = tree.getroot()

        # Initialize an empty list to store the results
        data_list = []

        # Iterate through all the 'road' elements in the XML file
        for road in root.findall('road'):
            road_id = road.attrib['id']
            link = road.find('link')

            # Initialize AimsunID as None
            aimsun_id = None

            # Extract predecessor and successor information
            if link is not None:
                predecessor = link.find('predecessor')
                successor = link.find('successor')

                # Both predecessor and successor are present
                if predecessor is not None and successor is not None:
                    if predecessor.attrib['elementType'] != "road" and successor.attrib['elementType'] != "road":
                        aimsun_id = f"{predecessor.attrib['elementId']}-{successor.attrib['elementId']}"

                # Only predecessor is present
                elif predecessor is not None:
                    if predecessor.attrib['elementType'] != "road":
                        aimsun_id = f"{predecessor.attrib['elementId']}-out"

                # Only successor is present
                elif successor is not None:
                    if successor.attrib['elementType'] != "road":
                        aimsun_id = f"in-{successor.attrib['elementId']}"

                # Neither predecessor nor successor is present
            else:
                aimsun_id = "in-out"

            # Append to list
            data_list.append({'OpenDriveFromID': road_id, 'AimsunID': aimsun_id})

        # Create DataFrame from the list
        RoadLookup = pd.DataFrame(data_list)
        RoadLookup = RoadLookup.dropna(subset=['AimsunID'])

        # Step 1: Group the DataFrame by 'AimsunID' and identify groups with the same 'AimsunID'
        grouped = RoadLookup.groupby('AimsunID')

        # Initialize an empty dictionary to store the new AimsunID mappings
        new_aimsun_ids = {}

        # Step 2: For each group, locate the corresponding roads in the OpenDRIVE file
        for aimsun_id, group in grouped:
            # Skip groups with only one member
            if len(group) <= 1:
                continue

            road_x_values = []

            # Step 3: Extract the x of the first <geometry> in <planView> for each road
            for _, row in group.iterrows():
                road_id = row['OpenDriveFromID']
                road_element = root.find(f"./road[@id='{road_id}']")
                if road_element is not None:
                    plan_view = road_element.find('planView')
                    if plan_view is not None:
                        first_geometry = plan_view.find('geometry')
                        if first_geometry is not None:
                            x = float(first_geometry.attrib.get('x', 0))
                            road_x_values.append((road_id, x))

            # Step 4: Sort the roads based on the x value
            road_x_values.sort(key=lambda x: x[1])

            # Step 5: Rename the AimsunID based on the sequence and conditions
            for i, (road_id, _) in enumerate(road_x_values):
                if aimsun_id.startswith("in-") and not aimsun_id.endswith("-out"):
                    new_aimsun_id = f"in{i + 1}-{aimsun_id.split('-')[1]}"
                elif aimsun_id.endswith("-out"):
                    new_aimsun_id = f"{aimsun_id.split('-')[0]}-out{i + 1}"
                else:
                    new_aimsun_id = aimsun_id  # Keep as is

                new_aimsun_ids[road_id] = new_aimsun_id

        # Update the DataFrame with the new AimsunID values
        RoadLookup['NewAimsunID'] = RoadLookup['OpenDriveFromID'].map(new_aimsun_ids).fillna(RoadLookup['AimsunID'])

        # Add "1-" after the "-" in each NewAimsunID
        RoadLookup['NewAimsunID'] = RoadLookup['NewAimsunID'].apply(lambda x: x.replace('-', '-1-'))

        RoadLookup.drop('AimsunID', axis=1, inplace=True)

        # Rename the 'NewAimsunID' column to 'AimsunID'
        RoadLookup.rename(columns={'NewAimsunID': 'AimsunID'}, inplace=True)

        # Convert the 'OpenDriveFromID' columns to the same data type (string) in both DataFrames
        InflowDf = InflowDf.copy()
        InflowDf['OpenDriveFromID'] = InflowDf['OpenDriveFromID'].astype(str)
        # RoadLookup = RoadLookup.copy
        RoadLookup['OpenDriveFromID'] = RoadLookup['OpenDriveFromID'].astype(str)

        # Perform the merge operation again
        merged_df = pd.merge(InflowDf, RoadLookup, on='OpenDriveFromID', how='left').copy()
        # Keep only the specified columns in the merged DataFrame and rearrange the column order
        Inflow = merged_df[['IntervalStart', 'IntervalEnd', 'AimsunID', 'Count']]
        Inflow.loc[:, 'IntervalStart'] = Inflow['IntervalStart'].astype(int)
        Inflow.loc[:, 'IntervalEnd'] = Inflow['IntervalEnd'].astype(int)
        Inflow.to_csv('MyNetwork/Aimsun/inflow.txt', index=False, header=False, sep=',')
        # Display the merged DataFrame
        self.Demand = 'MyNetwork/Aimsun/inflow.txt'
