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
class to host route element of Concrete scenario
'''
import os
import pandas as pd
import pyufunc as pf


class Route:
    '''The route class to host the route element of Concrete scenario
    '''
    def __init__(self):
        self.TurningRatio = {}

    def is_empty(self):
        """Check if the Route object is empty."""
        pass

    def generate_route(self, AbsScn):
        """Generate route data from the abstract scenario."""
        # =================================
        # Generate route
        # =================================
        Turn = AbsScn.Traffic.TurningRatio
        # Read this for now
        if not AbsScn.Traffic.VolumeLookupTable.empty:
            IDRef = AbsScn.Traffic.VolumeLookupTable
        else:
            path_lookup = AbsScn.input_config["Traffic"].get("GridSmart_lookup")
            path_lookup_abs = pf.path2linux(os.path.join(AbsScn.input_config.get("input_dir"), path_lookup))
            IDRef = pd.read_csv(path_lookup_abs)
        IDRef = IDRef.dropna(subset=['OpenDriveFromID', 'OpenDriveToID'])
        # IDRef = IDRef.astype({'OpenDriveFromID': int, 'OpenDriveToID': int})
        IDRef = IDRef.astype(str)

        MergedDf2 = pd.merge(Turn, IDRef, on=['IntersectionName', 'Turn'], how='left')
        Turn['OpenDriveFromID'] = MergedDf2['OpenDriveFromID']
        Turn['OpenDriveToID'] = MergedDf2['OpenDriveToID']
        Turn = Turn.dropna(subset=['OpenDriveFromID', 'OpenDriveToID'])
        self.TurningRatio = Turn.copy()
