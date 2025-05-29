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


class VissimPrep:
    """Class to prepare VISSIM simulation environment."""

    def __init__(self, **kwargs):
        """Initialize the VissimPrep class with optional parameters."""
        self.kwargs = kwargs
        self.verbose = kwargs.get('verbose', True)

    def prepare(self):
        """Prepare the VISSIM simulation environment."""
        if self.verbose:
            print("  :Preparing VISSIM simulation environment...")
