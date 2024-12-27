##############################################################################
# Copyright (c) 2024, Oak Ridge National Laboratory                          #
# All rights reserved.                                                       #
#                                                                            #
# This file is part of RealTwin and is distributed under a __TBD__           #
# license. For the licensing terms see the LICENSE file in the top-level     #
# directory.                                                                 #
#                                                                            #
# Contributors (Add you name below to acknowledge your contribution):        #
# Xiangyong Roy Luo                                                          #
##############################################################################
'''
##############################################################
# Created Date: Thursday, December 26th 2024
# Contact Info: luoxiangyong01@gmail.com
# Author/Copyright: Mr. Xiangyong Luo
##############################################################
'''

from pathlib import Path
from datetime import datetime
import os
import subprocess
import pyufunc as pf


def run_pylint_checker(*, disable_ids: list[str] = None, ignore_paths: list[str] = None) -> bool:
    '''Run pylint checker on the project

    Args:
        disable_ids: list of strings, the pylint ids to ignore
        ignore_paths: list of strings, the paths to ignore

    Note:
        This function for developers to run pylint checker on the project, not for the external users

    Raises:
        ValueError: if the pylint_checker.py is not under the tests or real-twin directory
        ValueError: if ignore_ids is not a list of strings

    Returns:
        bool: True if the pylint checker finished successfully, False otherwise
    '''

    # generate the project path
    dir_current = pf.path2linux(str(Path(__file__).resolve().parent))
    if dir_current.endswith('tests'):
        project_path = pf.path2linux(str(Path(__file__).resolve().parents[1]))
    elif dir_current.endswith('real-twin'):
        project_path = dir_current
    else:
        raise ValueError("  :Could not run pylint checker,"
                         " please confirm the pylint_checker.py under the tests or real-twin directory")

    # crate the current datetime
    current_datetime = datetime.now().strftime('%Y-%m-%d-%H-%M')

    # generate output file name
    output_filename = f'pylint_report_{current_datetime}.txt'
    output_abs_path = f'{project_path}/docs/code_evaluation/{output_filename}'
    if not os.path.exists(f'{project_path}/docs/code_evaluation/'):
        os.makedirs(f'{project_path}/docs/code_evaluation/')

    # check and generate the disable ids
    if not isinstance(disable_ids, list):
        raise ValueError('  :ignore_ids should be a list of strings')
    disable_ids_str = f"--disable={','.join(disable_ids)}"

    # check and generate the ignore paths
    if not isinstance(ignore_paths, list):
        raise ValueError('  :ignore_paths should be a list of strings')

    # convert the ignore paths to absolute paths
    ignore_paths = [pf.path2linux(os.path.abspath(path)) for path in ignore_paths]

    # check if the root drive is the capitalized letter, if not, capitalize it
    ignore_paths += [path[0].upper() + path[1:] if path[1] == ':' else path for path in ignore_paths]
    ignore_paths_str = f"""--ignore-paths={",".join(ignore_paths)}"""

    # include pylint configuration file if exists
    if os.path.exists(f'{project_path}/.pylintrc'):
        config_file_str = f'--rcfile={project_path}/.pylintrc'
        # print(config_file_str)
    else:
        config_file_str = ''

    # run pylint checker to the project
    try:
        subprocess.run(['pylint',
                        config_file_str,
                        ignore_paths_str,
                        disable_ids_str,
                        project_path,
                        f"--output={output_abs_path}",
                        "--msg-template='{path}:{line}:{column}:\n    {msg_id}({obj}): {msg} ({symbol})'",
                        '--exit-zero'],
                       check=True,
                       stdout=subprocess.PIPE)
        print('  :Pylint code checker finished successfully!'
              f' Check the report at {output_abs_path}')
        return True

    except subprocess.CalledProcessError as e:
        # check if the report generated
        if os.path.exists(output_abs_path) and os.path.getsize(output_abs_path) > 0:
            print('  :Pylint checker finished successfully!'
                  f' Check the report at {output_abs_path}')
            return True

        print(f'  :Pylint code checker failed! \n  :Error: {e}')
    return False


if __name__ == '__main__':
    run_pylint_checker(
        ignore_paths=["../te.py",],
        disable_ids=["C0301", "C0413", "W0719", "R1711", "W0511"])
