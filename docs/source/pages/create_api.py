'''
##############################################################
# Created Date: Monday, March 24th 2025
# Contact Info: luoxiangyong01@gmail.com
# Author/Copyright: Mr. Xiangyong Luo
##############################################################
'''

from __future__ import absolute_import
from pathlib import Path
import sys
import os

root = Path(__file__).resolve().parents[3]
sys.path = [str(root)] + sys.path

import realtwin
import realtwin.func_lib
import realtwin.func_lib._g_analyzer
import realtwin.func_lib._h_vis
from realtwin import RealTwin

collected_functions = {
    "realtwin._realtwin": ["RealTwin"],
    "realtwin.utils_lib": realtwin.utils_lib.__all__,
    "realtwin.func_lib._a_install_simulator": realtwin.func_lib._a_install_simulator.__all__,
    "realtwin.func_lib._b_load_inputs": realtwin.func_lib._b_load_inputs.__all__,
    "realtwin.func_lib._c_abstract_scenario": realtwin.func_lib._c_abstract_scenario.__all__,
    "realtwin.func_lib._d_concrete_scenario": realtwin.func_lib._d_concrete_scenario.__all__,
    "realtwin.func_lib._e_simulation": realtwin.func_lib._e_simulation.__all__,
    "realtwin.func_lib._f_calibration": realtwin.func_lib._f_calibration.__all__,
    "realtwin.func_lib._g_analyzer": realtwin.func_lib._g_analyzer.__all__,
    "realtwin.func_lib._h_vis": realtwin.func_lib._h_vis.__all__,
}


# remove all documents in the api folder
api_folder = "api"
for file in os.listdir(api_folder):
    if file.endswith(".rst"):
        os.remove(os.path.join(api_folder, file))

# Create the API documentation
total_functions = sum(len(module_list) for module_list in collected_functions.values())
for relative_path, module_list in collected_functions.items():

    if not module_list:
        continue

    for module in module_list:
        file_name = f"api/{relative_path}.{module}.rst"

        heading_message = f"{relative_path}.{module}"

        with open(file_name, "w", encoding="utf-8") as f:

            heading_message_new = heading_message.replace("_", "\\_")
            heading_dashes = "=" * len(heading_message_new)

            f.write(heading_message_new + "\n")
            f.write(heading_dashes + "\n\n")

            # if gmns_geo is in the module name, then it is a subpackage
            if "gmns_geo" in file_name:
                f.write(f".. automodule:: {relative_path}.{module}" + "\n\n")
            else:
                f.write(f".. automodule:: {relative_path}\n")
                f.write("   :members:\n")
                f.write("   :undoc-members:\n")
                f.write("   :show-inheritance:\n\n")
                f.write(f".. autofunction:: {module}" + "\n\n")
                f.write(f".. autoclass:: {module}" + "\n\n")

            f.close()

print(f"Successfully updated {total_functions} API documentations!")
