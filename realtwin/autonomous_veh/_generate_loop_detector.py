##############################################################################
# Copyright (c) 2024-, Oak Ridge National Laboratory                          #
# All rights reserved.                                                       #
#                                                                            #
# This file is part of RealTwin and is distributed under a GPL               #
# license. For the licensing terms see the LICENSE file in the top-level     #
# directory.                                                                 #
#                                                                            #
# Contributors: ORNL Real-Twin Team                                          #
# Contact: realtwin@ornl.gov                                                 #
##############################################################################


import os
import sys
from pathlib import Path
from datetime import datetime
from xml.dom import minidom
import xml.etree.ElementTree as ET


def xml_prettify(element: str) -> str:
    """Return a pretty-printed XML string for the Element."""
    rough_string = ET.tostring(element, 'utf-8')
    re_parsed = minidom.parseString(rough_string)
    return re_parsed.toprettyxml(indent="    ")


def generate_sumo_loop_detector_add_xml(path_net: str | Path, *,
                                        detector_type: str = "E1",
                                        add_fname: str = "detector.add.xml",
                                        detector_output_fname: str = "",
                                        dest_dir: str = "") -> bool:
    """""Generate the .add.xml file for SUMO and add loop detectors for each lane that has a detector.

    Args:
        utdf_dict (dict): A dictionary containing UTDF data.
        detector_type (str): The type of detector to be added. Defaults to "E1".
            Accepted type: E1: Inductive loop detector, E2: Lane area detector, E0: Instant induction loops.
        fname (str, optional): SUMO additional file. Defaults to "detector.add.xml".
        detector_output_fname (str): The output file name to record loop detectors data in simulation.
            the output file name in default is: detector_output_YYYYMMDDHHMM.xml

    See Also:
        For different detector types, please refer to the SUMO documentation:
            https://sumo.dlr.de/docs/Simulation/Output/#simulated_detectors

        E1: Inductive loop detector
            https://sumo.dlr.de/docs/Simulation/Output/Induction_Loops_Detectors_%28E1%29.html

        E2: Lane area detector,
            https://sumo.dlr.de/docs/Simulation/Output/Lanearea_Detectors_%28E2%29.html

        E0: Instant induction loops,
            https://sumo.dlr.de/docs/Simulation/Output/Instantaneous_Induction_Loops_Detectors.html
    """

    # get all lanes from the SUMO network
    net = ET.parse(path_net)
    root = net.getroot()

    # get all traffic signal id
    tlLights = root.findall(".//tlLogic")
    tl_ids = [tl.get("id") for tl in tlLights]

    # assume tl_ids are signalized intersection ids

    edges = root.findall(".//edge")
    # edges_not_internal = [edge for edge in edges if edge.get("function") != "internal"]
    edges_not_internal = edges

    lane_lookup_dict = {}
    for edge in edges_not_internal:
        # if the to id of the edge is in tl_ids, add the lane to the lookup dict
        if edge.get("to") in tl_ids:
            for lane in edge.findall(".//lane"):
                lane_id = lane.get("id")
                lane_length = float(lane.get("length"))
                lane_lookup_dict[lane_id] = {
                    "length": lane_length,
                    "numDetects": 1
                }

    # get detector tag
    if detector_type == "E1":
        detector_tag = "inductionLoop"
    elif detector_type == "E2":
        detector_tag = "laneAreaDetector"
    elif detector_type == "E0":
        detector_tag = "instantInductionLoop"
    else:
        raise ValueError(
            f"Unknown detector type: {detector_type}. Accepted types are E1, E2, E0.")

    add_elem = ET.Element("additional")

    if not str(add_fname).endswith(".add.xml"):
        add_fname = f"{add_fname}.add.xml"

    if detector_output_fname:
        if not str(detector_output_fname).endswith(".xml"):
            detector_output_fname = f"{detector_output_fname}.xml"
    else:
        detector_output_fname = f"detector_output_{datetime.now().strftime(r'%Y%m%d')}.xml"

    for lane_id, lane_info in lane_lookup_dict.items():
        # Check if the lane has a detector, num_detectors not None
        if lane_info.get("numDetects"):
            # Create the detector element
            detector = ET.SubElement(add_elem, detector_tag)
            detector.set("id", f"{lane_id}_detector")
            detector.set("lane", f"{lane_id}")
            detector.set("pos", "-8")  # must assigned, backward of the lane
            detector.set("friendlyPos", "true")
            # detector.set("vTypes", "")
            detector.set("file", f"{detector_output_fname}")  # output file name

            # # Add the detector to the root element and save root file to path_net
            # root = ET.Element("additional")
            # root.append(detector)

    xml_str = xml_prettify(add_elem)
    path_output = Path(dest_dir) / add_fname if dest_dir else Path(add_fname)
    with open(path_output, "w") as f:
        f.write(xml_str)
    return True


if __name__ == "__main__":
    path_net = Path(
        r"C:\Users\xh8\ORNL_work\gitlab_workspace\ScenarioGenerator\datasets\avtes\chatt3_updated_signal_v2.net.xml")
    detector_type = "E1"
    add_fname = "../datasets/avtes/detector.add.xml"
    sim_output_fname = "detector_output.xml"
    generate_sumo_loop_detector_add_xml(path_net, detector_type, add_fname, sim_output_fname)