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

""" Utilities for autonomous vehicle simulation in SUMO."""

from __future__ import absolute_import
import colorsys
import xml.etree.ElementTree as ET
import traci
import os
import sys
from pathlib import Path
from xml.dom import minidom

if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
sys.path.append(os.path.join('c:', os.sep, 'whatever',
                'path', 'to', 'sumo', 'tools'))


def prettify_xml(xml_tree: ET.ElementTree) -> str:
    """Return a pretty-printed XML string for the ElementTree."""

    xml_tree_str = ET.tostring(xml_tree.getroot(), encoding='utf-8', method='xml')
    parsed_xml = minidom.parseString(xml_tree_str)
    return parsed_xml.toprettyxml(indent="    ")


def generate_rgb_colors(num_colors: int) -> list:
    """Generate `n` visually distinct RGB colors.
    Returns a list of (r, g, b) tuples with values in 0-255.
    """
    colors = []
    for i in range(num_colors):
        # evenly spaced hue around [0, 1)
        h = i / num_colors
        s = 1.0      # full saturation
        v = 1.0      # full brightness
        # convert to floats in [0,1]
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        # rescale to 0–255 ints
        colors.append(f"{int(r * 255)},{int(g * 255)},{int(b * 255)}")
    return colors


# Function to generate dictionary with all vehicle type attributes needed to create new vehicle type in SUMO
def create_veh_type_attributes(av_config: dict) -> dict:
    """ Create a dictionary with all vehicle type attributes needed to create new vehicle types in SUMO."""

    vTypeAttributes = ["id", "vClass",
                       "carFollowModel", "laneChangeModel",
                       "probability",
                       "accel", "decel", "minGap", "sigma", "tau", "emergencyDecel"]

    # Extract values from the configuration
    veh_types = av_config.get("veh_types", [])
    CFmodel_USER = av_config.get("CFmodel", {})
    pct_penetration = av_config.get("pct_penetration")
    car_follow_model = av_config.get("car_follow_model", "Krauss")
    lane_change_model = av_config.get("lane_change_model", "LC2013")

    # Create a dictionary for vehicle type attributes
    veh_type_attributes = {vType: dict.fromkeys(vTypeAttributes) for vType in veh_types}
    for i, vType in enumerate(veh_types):
        veh_type_attributes[vType]['id'] = vType
        veh_type_attributes[vType]['vClass'] = "passenger"
        veh_type_attributes[vType]['carFollowModel'] = car_follow_model
        veh_type_attributes[vType]['laneChangeModel'] = lane_change_model
        veh_type_attributes[vType]['probability'] = pct_penetration[i] * 0.01

        for i in range(5, len(vTypeAttributes)):
            veh_type_attributes[vType][vTypeAttributes[i]] = CFmodel_USER[vType][car_follow_model][vTypeAttributes[i]]

    return veh_type_attributes


# # Incorporate new vehicle types and vehicle type distributions in SUMO flow file to create a new modified flow file
def update_sumo_flow_xml(path_flow: str, veh_types: list, veh_type_attributes: dict):
    """ Update the SUMO flow XML file with new vehicle types and their attributes."""

    # Create the root element
    root = ET.Element("routes")

    # Create child elements
    vTypeDistribution = ET.SubElement(root, "vTypeDistribution")
    vTypeDistribution.set("id", "vdis")

    num_veh_types = len(veh_types)
    colors_list = generate_rgb_colors(num_veh_types)
    for i, vType in enumerate(veh_types):
        keysList = list(veh_type_attributes[vType].keys())
        vType_new = ET.SubElement(vTypeDistribution, "vType")
        for key in keysList:
            vType_new.set(key, str(veh_type_attributes[vType][key]))
        vType_new.set("color", colors_list[i])

    # Create an XML file
    vTypeTree = ET.ElementTree(root)
    # flow_dir = Path(path_flow).parent
    # vTypeTree.write(flow_dir / "vTypeDistTree.xml")

    # Parse the existing flow XML file
    origFlowTree = ET.parse(path_flow)

    # Get all the elements from the previous flow XML file
    origFlowTreeElements = origFlowTree.findall('flow')

    # Iterate over the elements and add them to the second XML file
    for element in origFlowTreeElements:
        element.set('type', 'vdis')
        vTypeTree.getroot().append(element)

    # Write the second XML file to a new file
    vTypeTree.write(path_flow)
    print(f"  :flow file updated with new vehicle types and saved at: {path_flow}")
    return True


def create_sumo_rou_xml(path_net: str, path_flow: str, path_turn: str, path_rou: str) -> bool:
    """ Create a SUMO route file using the provided network, flow, and turn ratio files."""
    # os.system('jtrrouter --route-files='+filepath+'vTypeModifiedFlow.xml --turn-ratio-files='+filepath+'chattnew.turn.xml --net-file='+filepath+'chatt3.net.xml --output-file='+filepath+'chatt3_modified_test.rou.xml --accept-all-destinations 1 --vtype-output='+filepath+'vtype_output_test.xml')

    path_vtype_output = Path(path_rou).parent / 'vtype_output.xml'

    os.system(
        f'jtrrouter --route-files={path_flow} --turn-ratio-files={path_turn} --net-file={path_net} --output-file={path_rou} --accept-all-destinations 1 --vtype-output={str(path_vtype_output)}')
    print(f"  :Route File Created: {path_rou}")
    return True


def add_veh_types_to_rou(path_rou: str, veh_types: list, veh_type_attributes: dict) -> bool:
    """ Add vehicle types to the SUMO route file."""

    Tree = ET.parse(path_rou)
    root = Tree.getroot()

    num_veh_types = len(veh_types)
    colors_list = generate_rgb_colors(num_veh_types)
    for i, vType in enumerate(veh_types):
        keysList = list(veh_type_attributes[vType].keys())
        vType_new = ET.SubElement(root, "vType")
        for key in keysList:
            vType_new.set(key, str(veh_type_attributes[vType][key]))
        vType_new.set("color", colors_list[i])
        root.insert(0, vType_new)
    Tree.write(path_rou)
    color_str = ', '.join([f'{vType}: {colors_list[i]}' for i, vType in enumerate(veh_types)])
    print(f"  :Added Vehicle Types into Route File, color assigned: {color_str}")
    return True


def create_sumo_config(path_cfg: str,
                       sim_name: str = "chatt",
                       sim_start: int = 0,
                       sim_end: int = 3600):
    """ Create a SUMO configuration file with default settings.

    Args:
        path_cfg (str): The path where the configuration file will be saved.
        sim_name (str): The name of the simulation. Defaults to "chatt".
    """

    # create the root element
    root = ET.Element("configuration")

    # input element
    input_elem = ET.SubElement(root, "input")

    input_net_file = ET.SubElement(input_elem, "net-file")
    input_net_file.set("value", f"{sim_name}.net.xml")

    input_rou_file = ET.SubElement(input_elem, "route-files")
    input_rou_file.set("value", f"{sim_name}.rou.xml")

    input_additional_files = ET.SubElement(input_elem, "additional-files")
    input_additional_files.set("value", "detector.add.xml")

    # output element
    output_elem = ET.SubElement(root, "output")

    output_full_output = ET.SubElement(output_elem, "full-output")
    output_full_output.set("value", f"{sim_name}_Full_Output.xml")

    output_amitran_output = ET.SubElement(output_elem, "amitran-output")
    output_amitran_output.set("value", f"{sim_name}_Amitran_Output.xml")

    output_vehroute_output = ET.SubElement(output_elem, "vehroute-output")
    output_vehroute_output.set("value", f"{sim_name}_VehRoute_Output.xml")

    # time element
    time_elem = ET.SubElement(root, "time")

    time_begin = ET.SubElement(time_elem, "begin")
    time_begin.set("value", f"{sim_start}")

    time_end = ET.SubElement(time_elem, "end")
    time_end.set("value", f"{sim_end}")  # 1 hour simulation

    time_step_length = ET.SubElement(time_elem, "step-length")
    time_step_length.set("value", "0.1")  # 100 ms time step

    # gui element
    gui_elem = ET.SubElement(root, "gui_only")
    gui_start = ET.SubElement(gui_elem, "start")
    gui_start.set("value", "t")  # start the GUI

    # report element
    report_elem = ET.SubElement(root, "report")

    report_no_warnings = ET.SubElement(report_elem, "no-warnings")
    report_no_warnings.set("value", "true")

    report_no_step_log = ET.SubElement(report_elem, "no-step-log")
    report_no_step_log.set("value", "true")

    # write the XML to the file
    tree = ET.ElementTree(root)

    formatted_string = prettify_xml(tree)
    with open(path_cfg, 'w') as file:
        file.write(formatted_string)

    # tree.write(path_cfg, encoding='utf-8', xml_declaration=True)
    print(f"  :SUMO configuration file created at: {path_cfg}")
    return True


def run_sumo_simulation(path_cfg: str, sim_time: int = 3600) -> bool:
    """ Run a SUMO simulation using the provided configuration file.

    Args:
        path_cfg (str): the path to the SUMO configuration file.
        sim_time (int, optional): the duration of the simulation in seconds. Defaults to 3600.

    Returns:
        bool: True if the simulation ran successfully, False otherwise.
    """
    traci.start(["sumo-gui", "-c", f"{path_cfg}"])
    while traci.simulation.getTime() < sim_time:
        traci.simulationStep()
    traci.close()
    print(f"  :Simulation completed successfully. Output files are saved in: {Path(path_cfg).parent}")
    return True