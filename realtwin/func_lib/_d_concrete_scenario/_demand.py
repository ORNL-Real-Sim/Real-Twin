##############################################################################
# Copyright (c) 2024, Oak Ridge National Laboratory                          #
# All rights reserved.                                                       #
#                                                                            #
# This file is part of RealTwin and is distributed under a GPL               #
# license. For the licensing terms see the LICENSE file in the top-level     #
# directory.                                                                 #
#                                                                            #
# Contributors: ORNL Real-Twin Team                                          #
# Contact: realtwin@ornl.gov                                                 #
##############################################################################
'''
class to host demand element of Concrete scenario
'''
import os
import xml.etree.ElementTree as ET
import pandas as pd
import pyufunc as pf


class Demand:
    '''The demand class to host the demand element of Concrete scenario
    '''
    def __init__(self):
        self.Inflow = {}
        # Inflow: ['BeginInterval', 'EndInterval', 'FromEdge', 'Count', 'VehicleType']
        # TurningRatio: ['BeginInterval', 'EndInterval', 'FromEdge','ToEdge', 'Bound','TurnPercent','TurnDirection']

    def is_empty(self):
        """Check if the Demand object is empty."""
        pass

    def generate_traffic(self, AbsScn):
        '''Generate the traffic demand for the concrete scenario'''

        Count = AbsScn.Traffic.Volume
        NetworkPath = AbsScn.Network.OpenDriveNetwork.OpenDrive_network[0]
        # Read this for now

        if not AbsScn.Traffic.VolumeLookupTable.empty:
            IDRef = AbsScn.Traffic.VolumeLookupTable
        else:
            path_lookup = AbsScn.input_config["Traffic"].get("GridSmart_lookup")
            path_lookup_abs = pf.path2linux(os.path.join(AbsScn.input_config.get("input_dir"), path_lookup))
            IDRef = pd.read_csv(path_lookup_abs)
        IDRef = IDRef.dropna(subset=['OpenDriveFromID', 'OpenDriveToID'])
        IDRef = IDRef.astype({'OpenDriveFromID': int, 'OpenDriveToID': int})
        IDRef = IDRef.astype(str)

        MergedDf1 = pd.merge(Count, IDRef, on=['IntersectionName', 'Turn'], how='left')
        Count['OpenDriveFromID'] = MergedDf1['OpenDriveFromID']
        Count['OpenDriveToID'] = MergedDf1['OpenDriveToID']
        Count = Count.groupby(['IntervalStart',
                               'IntervalEnd',
                               'IntersectionName',
                               'OpenDriveFromID'], as_index=False)['Count'].sum()

        Count = Count.dropna(subset=['OpenDriveFromID'])
        Count = Count.astype({'Count': int})
        # Omit exiting approach to avoid replicate problem for not
        # Count2 = Count.groupby(['IntervalStart', 'IntervalEnd', 'IntersectionName',
        # 'OpenDriveToID'], as_index=False)['Count'].sum()

        # Load the XML file
        tree = ET.parse(NetworkPath)
        root = tree.getroot()

        # Find all roads with the specified condition
        InflowRoad = []

        for road in root.findall('road'):
            link = road.find('link')
            if link is not None:
                predecessor = link.find('predecessor')
                if predecessor is not None and predecessor.get('elementType') == 'junction':
                    junction_id = predecessor.get('elementId')

                    # Find the junction element with the specified id
                    junction = root.find(f".//junction[@id='{junction_id}']")

                    # Check if the junction has only one connection
                    if junction is not None and len(junction.findall('connection')) == 1:
                        road_id = road.get('id')
                        InflowRoad.append(road_id)

        InflowCount = Count[Count['OpenDriveFromID'].isin(InflowRoad)]

        self.Inflow = InflowCount.copy()
