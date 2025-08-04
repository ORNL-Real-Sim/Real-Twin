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

"""Load configuration for autonomous vehicles in SUMO."""

import contextlib
import yaml
from pathlib import Path
try:
    from ._carfollowing_lanechanging_model import CFmodel, CF_DEFAULT_PARAMETERS
except ImportError:
    from realtwin.autonomous_veh._carfollowing_lanechanging_model import CFmodel, CF_DEFAULT_PARAMETERS


def load_av_configs(path_config: str | Path) -> dict:
    """Load the autonomous vehicle configuration from a YAML file.

    Args:
        path_config (str): Path to the configuration file.

    Returns:
        dict: Configuration parameters loaded from the file.
    """

    if isinstance(path_config, (str, Path)):
        path_config = str(path_config)
    else:
        raise TypeError("path_config must be a string or Path object.")

    # TDD: check if the file exists and is a valid YAML file
    if not Path(path_config).is_file():
        raise FileNotFoundError(f"Configuration file not found: {path_config}")

    if not (path_config.endswith('.yml') or path_config.endswith('.yaml')):
        raise ValueError("Configuration file must be a YAML file with .yml or .yaml extension.")

    # load the YAML file
    with open(path_config, 'r', encoding="utf-8") as yaml_data:
        config = yaml.safe_load(yaml_data)

    # check veh penetration
    pct_penetration = config.get('pct_penetration', 0.0)
    if not (0 <= pct_penetration <= 100):
        pct_penetration = 0.0
        config['pct_penetration'] = 0.0

    # check veh types, from user defined veh_types
    veh_types = config.get('veh_types', [])
    if not veh_types:
        raise ValueError("Vehicle types must be specified in the configuration file.")

    CFmodel_USER = {
        veh_type: CF_DEFAULT_PARAMETERS.copy()
        for veh_type in veh_types
        if veh_type not in CFmodel
    }
    # update the CFmodel parameters for each veh_type
    CF_model_names = list(CF_DEFAULT_PARAMETERS.keys())
    for i in range(len(veh_types)):
        # copy the default CFmodel for each veh_type
        CFmodel_USER[veh_types[i]] = CFmodel[veh_types[i]].copy()

        # update default parameters with user defined parameters
        for model_name in CF_model_names:
            # get user defined parameters for each model
            model_key = f"{model_name}Parameters"
            user_model = config.get(model_key, {})

            # update each parameter for the veh_type
            for model_param in user_model:
                with contextlib.suppress(IndexError):
                    # get the parameter value from user input
                    param_values = user_model[model_param][i]
                    CFmodel_USER[veh_types[i]][model_name][model_param] = param_values
            # delete the model_key from config to avoid confusion
            if model_key in config:
                del config[model_key]

    # Add the updated CFmodel to the config dictionary for future use
    config["CFmodel"] = CFmodel_USER

    return config
