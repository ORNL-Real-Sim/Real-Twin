##############################################################
# Created Date: Monday, August 17th 2026
# Contact Info: luoxiangyong01@gmail.com
# Author/Copyright: Mr. Xiangyong Luo
##############################################################

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
