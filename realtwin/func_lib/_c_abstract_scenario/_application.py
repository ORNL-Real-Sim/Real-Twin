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
class to host Application element of Abstract scenario
'''


class Application():
    '''The application class to host the Application element of Abstract scenario
    '''
    def __init__(self):
        pass

    def isEmpty(self):
        """Check if the Application element is empty"""
        pass

    def setValue(self, UserInputPath):
        """Set the value of the Application element from the user input file"""
        # self.CAVparameter == pd.read_csv(UserInputPath)
        pass
