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


def run_pylint_checker(ignore_ids: list[str] = ["C0301"]) -> None:
    '''
    Run pylint checker on the project

    Args:
        ignore_ids: list of strings, the pylint ids to ignore

    Raises:
        ValueError: if the pylint_checker.py is not under the tests or real-twin directory
        ValueError: if ignore_ids is not a list of strings

    Returns:
        None
    '''

    # generate the project path
    dir_current = pf.path2linux(str(Path(__file__).resolve().parent))
    if dir_current.endswith('tests'):
        project_path = pf.path2linux(str(Path(__file__).resolve().parents[1]))
    elif dir_current.endswith('real-twin'):
        project_path = dir_current
    else:
        raise ValueError("Could not run pylint checker,"
                         " please confirm the pylint_checker.py under the tests or real-twin directory")

    # crate the current datetime
    current_datetime = datetime.now().strftime('%Y-%m-%d-%H-%M')

    # generate output file name
    output_filename = f'pylint_report_{current_datetime}.txt'
    output_abs_path = f'{project_path}/docs/code_evaluation/{output_filename}'
    if not os.path.exists(f'{project_path}/docs/code_evaluation/'):
        os.makedirs(f'{project_path}/docs/code_evaluation/')

    # check and generate the ignore ids
    if not isinstance(ignore_ids, list):
        raise ValueError('ignore_ids should be a list of strings')

    ignore_str = f"--disable={','.join(ignore_ids)}"

    # run pylint checker to the project
    try:
        subprocess.run(['pylint',
                        ignore_str,
                        project_path,
                        f"--output={output_abs_path}",
                        "--msg-template='{path}:{line}:{column}:\n    {msg_id}({obj}): {msg} ({symbol})'",
                        '--exit-zero'],
                       check=True,
                       stdout=subprocess.PIPE)
        print('  :Pylint checker finished successfully!'
              f' Check the report at {output_abs_path}')
        return

    except subprocess.CalledProcessError as e:
        # check if the report generated
        if os.path.exists(output_abs_path) and os.path.getsize(output_abs_path) > 0:
            print('  :Pylint checker finished successfully!'
                  f' Check the report at {output_abs_path}')
            return

        print(f'  :Pylint checker failed! \n  :Error: {e}')
    return None


if __name__ == '__main__':
    run_pylint_checker()
