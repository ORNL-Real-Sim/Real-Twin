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


class TrafficControl:
    '''The demand class to host the demand element of Concrete scenario
    '''
    def __init__(self):
        self.Signal = {}

    def is_empty(self):
        """Check if the TrafficControl object is empty."""
        pass

    def generate_control(self, AbsScn):
        """Generate control data from the abstract scenario."""
        # =================================
        # load class from AbstractScenario
        # =================================
        # SignalDict = AbsScn.dataObjDict['Control']['Signal'].Signal
        # # Read this for now
        # IDRef = pd.read_csv('Synchro_lookuptable.csv')
        # MergedDf3 = pd.merge(SignalDict ,IDRef, on=['INTID'], how='left')

        pass
