'''
##############################################################
# Created Date: Wednesday, July 16th 2025
# Contact Info: luoxiangyong01@gmail.com
# Author/Copyright: Mr. Xiangyong Luo
##############################################################
'''

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from IPython import get_ipython


def is_running_in_notebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True  # Running in a Jupyter Notebook or IPython
        else:
            return False  # Running in a different IPython environment (e.g., IPython shell)
    except NameError:
        return False  # Not running in an IPython environment (likely a standard terminal)
