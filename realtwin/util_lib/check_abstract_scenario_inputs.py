'''
##############################################################
# Created Date: Tuesday, May 6th 2025
# Contact Info: luoxiangyong01@gmail.com
# Author/Copyright: Mr. Xiangyong Luo
##############################################################
'''

import os
from pathlib import Path
import pyufunc as pf


def check_abstract_inputs(input_dir: str) -> bool:
    # check If Control folder is empty
    path_control = pf.path2linux(Path(input_dir) / "Control")
    if not os.path.exists(path_control):
        raise Exception(f"  :Error: Control folder does not exist: {path_control}")

    if not pf.get_filenames_by_ext(path_control, file_ext="csv"):
        raise Exception(f"  :Error: Control folder is empty: {path_control},"
                        "Please include Synchro UTDF file (signal) inside Control folder"
                        " and add the control file name to the input configuration file.")

    # check If Traffic folder is empty
    path_traffic = pf.path2linux(Path(input_dir) / "Traffic")
    if not os.path.exists(path_traffic):
        raise Exception(f"  :Error: Traffic folder does not exist: {path_traffic}")

    if not pf.get_filenames_by_ext(path_traffic, file_ext="*"):
        raise Exception(f"  :Error: Traffic folder is empty: {path_traffic},"
                        "Please include turn movement file for each intersection inside Traffic folder"
                        " and add the file names to the MatchupTable.xlsx "
                        "(You will notice the generated MatchupTable.xlsx inside your input folder).")

    # check if the MatchupTable.xlsx exists
    path_matchup = pf.path2linux(Path(input_dir) / "MatchupTable.xlsx")
    if not os.path.exists(path_matchup):
        raise Exception(f"  :Error: Matchup table does not exist: {path_matchup}"
                        "Please generate the Matchup table by run generate_inputs()")
    return True
