'''
##############################################################
# Created Date: Friday, May 23rd 2025
# Contact Info: luoxiangyong01@gmail.com
# Author/Copyright: Mr. Xiangyong Luo
##############################################################
'''

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
