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


from pathlib import Path
from lxml import etree
import pandas as pd
import re
import subprocess


def name_without_suffixes(p: Path) -> str:
    """Return the name of the file without any suffixes."""
    # peel off each suffix until none remain
    while p.suffix:
        p = p.with_suffix('')
    return p.name


def parse_SUMO_ID(net_file: str) -> bool:
    """ Parse the SUMO network file to extract IDs (non internal) and update the net file.

    Args:
        net_file (str): Path to the SUMO network file.

    Returns:
        bool: True if the parsing and updating were successful, False otherwise.
    """

    try:
        # Parse the SUMO network file
        tree = etree.parse(net_file)
        root = tree.getroot()

        # Get all non-internal edge IDs
        sumoID_org = []
        sumoID_org.extend(
            edge.get("id")
            for edge in root.findall("edge")
            if edge.get("function") != "internal"
        )
        # Build replacement lists
        oldedgeID = []
        newedgeID = []

        for eid in sumoID_org:
            if not eid.startswith("-"):
                oldedgeID.append(eid)
                new_id = "-" + eid
                if new_id in sumoID_org:
                    new_id = "-" + new_id
                newedgeID.append(new_id)

        edgeID_replacement = pd.DataFrame({
            "oldedgeID": oldedgeID,
            "newedgeID": newedgeID
        })

        with open(net_file, "r", encoding="utf-8") as f:
            original_text = f.read()

        # Perform replacements in the file content
        for _, row in edgeID_replacement.iterrows():
            old_id = row["oldedgeID"]
            new_id = row["newedgeID"]
            original_text = re.sub(f'"{old_id}"', f'"{new_id}"', original_text)
            original_text = re.sub(f'lane id="{old_id}_', f'lane id="{new_id}_', original_text)
            original_text = re.sub(f' {old_id}_', f' {new_id}_', original_text)
            original_text = re.sub(f'incLanes="{old_id}_', f'incLanes="{new_id}_', original_text)

        with open(net_file, "w", encoding="utf-8") as f:
            f.write(original_text)

        return True

    except Exception as e:
        print(f"Error parsing SUMO ID: {e}")
        return False


def parse_SUMO_to_OpenDrive(net_file: str) -> bool:
    """  Parse the SUMO network file to OpenDrive file."""
    file_dir = Path(net_file).parent
    file_name = name_without_suffixes(Path(net_file))
    output_file = file_dir / f"{file_name}.xodr"

    cmd = [
        "netconvert",
        "-s", f"{net_file}",
        "--opendrive-output", f"{output_file}",
        "--output.original-names", "true",
        "--junctions.scurve-stretch", "1.0"
    ]

    process = subprocess.Popen(cmd, shell=True)
    process.wait()

    if process.returncode != 0:
        print(f"Error converting SUMO to OpenDrive: {process.returncode}")
        return False
    return True


def match_SUMO_to_OpenDrive(xodr_file: str) -> bool:
    """ Match SUMO and OpenDrive IDs in the OpenDrive file."""

    try:

        with open(xodr_file, 'r', encoding='utf-8') as f:
            original_text = f.read()

        tree = etree.parse(xodr_file)
        root = tree.getroot()

        # Extract old/new edge IDs from <road>
        old_edge_ids = []
        new_edge_ids = []
        warning_printed = False
        for road in root.findall("road"):
            road_id = road.get("id")
            user_data = road.find("userData")
            if user_data is not None:
                sumo_id = user_data.get("value")

                if sumo_id.startswith("-"):
                    old_edge_ids.append(road_id)
                    new_id = f'{sumo_id[1:]}'
                    new_edge_ids.append(new_id)
                elif not warning_printed:
                    print("To ensure Real-Twin works properly, please make all SUMO edge ID start with '-'")
                    warning_printed = True

        # Extract old/new junction IDs from <junction>
        old_junction_ids = []
        new_junction_ids = []

        for junction in root.findall("junction"):
            junction_id = junction.get("id")
            junction_name = junction.get("name")
            old_junction_ids.append(junction_id)
            new_junction_ids.append(junction_name)

        edgeID = pd.DataFrame({'oldedgeID': old_edge_ids, 'newedgeID': new_edge_ids})
        junctionID = pd.DataFrame({'oldjunctionID': old_junction_ids, 'newjunctionID': new_junction_ids})

        # Apply edge ID replacements
        for _, row in edgeID.iterrows():
            old_id = row['oldedgeID']
            new_id = row['newedgeID']
            original_text = original_text.replace(f'id="{old_id}" junction=', f'id="{new_id}" junction=')
            original_text = original_text.replace(f'elementType="road" elementId="{old_id}"', f'elementType="road" elementId="{new_id}"')
            original_text = original_text.replace(f'incomingRoad="{old_id}"', f'incomingRoad="{new_id}"')
            original_text = original_text.replace(f'connectingRoad="{old_id}"', f'connectingRoad="{new_id}"')

        # Apply junction ID replacements
        for _, row in junctionID.iterrows():
            old_id = row['oldjunctionID']
            new_id = row['newjunctionID']
            original_text = original_text.replace(f'id="{old_id}">', f'id="{new_id}">')
            original_text = original_text.replace(f'elementType="junction" elementId="{old_id}"', f'elementType="junction" elementId="{new_id}"')
            original_text = original_text.replace(f'junction="{old_id}"', f'junction="{new_id}"')

        with open(xodr_file, 'w', encoding='utf-8') as f:
            f.write(original_text)

        return True

    except Exception as e:
        print(f"Error matching SUMO to OpenDrive: {e}")
        return False


def parse_SUMO_TLS_ID(path_matchup_table: str, path_net_file: str) -> bool:
    """Parse the SUMO network file to extract traffic light system IDs and update the net file.

    Args:
        path_matcup_table (str): Path to the SUMO matcup table file.
        path_net_file (str): Path to the SUMO network file.

    Returns:
        bool: True if the parsing and updating were successful, False otherwise.
    """
    try:
        # Read the matcup table
        MatchupTable_UserInput = pd.read_excel(path_matchup_table, skiprows=1, dtype=str)
        merged_columns = ["JunctionID_OpenDrive", "File_GridSmart", "Date_GridSmart",
                          "IntersectionName_GridSmart", "File_Synchro", "IntersectionID_Synchro", "Need calibration?"]
        MatchupTable_UserInput[merged_columns] = MatchupTable_UserInput[merged_columns].ffill()

        signalized_junction = MatchupTable_UserInput[MatchupTable_UserInput['Turn_Synchro'].notna()]['JunctionID_OpenDrive'].unique()
        signalized_junction_string = ",".join(signalized_junction.astype(str))

        cmd = f'cmd /c "netconvert -s {path_net_file} --o {path_net_file} --tls.discard-loaded true --tls.set {signalized_junction_string}"'
        process = subprocess.Popen(cmd, shell=True)
        process.wait()

        return True

    except Exception as e:
        print(f"Error parsing SUMO TLS ID: {e}")
        return False
