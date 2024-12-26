'''
##############################################################
# Created Date: Thursday, December 26th 2024
# Contact Info: luoxiangyong01@gmail.com
# Author/Copyright: Mr. Xiangyong Luo
##############################################################
'''

from pathlib import Path
from datetime import datetime
import subprocess


def run_pylint_checker() -> None:
    '''
    Run pylint checker on the project
    '''

    # crate the current datetime
    current_datetime = datetime.now().strftime('%Y-%m-%d %H:%M')

    # generate output file name
    output_file = f'pylint_report_{current_datetime}.txt'

    # generate the project path
    project_path = Path(__file__).resolve().parents[1]

    # run pylint checker on the project
    result = subprocess.run(['pylint', project_path, f"--output={output_file}"], check=True)

    if result.returncode == 0:
        print('Pylint checker finished successfully!'
              f'Check the report at {project_path}/{output_file}')
    print('Pylint checker failed!')

    return None


if __name__ == '__main__':
    run_pylint_checker()
