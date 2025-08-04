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
