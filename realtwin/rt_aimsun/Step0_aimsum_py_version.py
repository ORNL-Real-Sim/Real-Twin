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

import sys
import platform
from PyANGConsole import *
from PyANGKernel import *
from PyANGBasic import *


def main(argv):

    # Start a Console
    console = ANGConsole()
    # Load a network
    if console.open(argv[1]):
        import sys
        print(f"Aimsun Python Version:{platform.python_version()}")

        console.save(argv[1])
        console.close()
    else:
        console.getLog().addError("Aimsun Python Version:Not Found")
        print("Aimsun Python Version:Not Found")


if __name__ == "__main__":
    sys.exit(main(sys.argv))
