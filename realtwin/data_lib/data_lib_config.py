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

# Pre-selected routes for the behavior calibration for demo data

# time in seconds
# edge_list is the list of edge IDs in the route, user can open generated SUMO net file manually see the edge IDs
sel_behavior_routes = {
    "chattanooga": {"route_1": {"time": 240,
                                "edge_list": ["-312", "-293", "-297", "-288", "-2881", "-286", "-302",
                                              "-3221", "-322", "-313", "-284", "-2841", "-328", "-304"]},
                    "route_2": {"time": 180,
                                "edge_list": ["-2801", "-280", "-307", "-327", "3271", "-281", "-315", "3151",
                                              "-321", "-300", "-2851", "-285", "-290", "-298", "-295"]}},

}
