##############################################################################
# Copyright (c) 2024, Oak Ridge National Laboratory                          #
# All rights reserved.                                                       #
#                                                                            #
# This file is part of RealTwin and is distributed under a __TBD__           #
# license. For the licensing terms see the LICENSE file in the top-level     #
# directory.                                                                 #
#                                                                            #
# Contributors: ORNL Real-Twin Team                                          #
# Contact: realtwin@ornl.gov                                                 #
##############################################################################
'''
class to host a unique AbstractScenario definition
'''
import os
import warnings
import io
import copy
from pathlib import Path

# import four elements of AbstractScenario
from ._traffic import Traffic
from ._network import Network
from ._control import Control
from ._application import Application

import pandas as pd
import pyufunc as pf


def time_to_seconds(time_str) -> int:
    """convert time string to seconds

    Args:
        time_str (str): the time string in format 'HH:MM'
    """
    # hour, minute = map(int, time_str.split(':'))
    hour, minute = [int(x) for x in time_str.split(':')]
    return (hour * 3600) + (minute * 60)


def load_traffic_volume(demand_data: str | pd.DataFrame) -> pd.DataFrame:
    """load traffic volume data from file

    Args:
        demand_data (str): the demand file path

    Returns:
        pd.DataFrame: the converted demand data in DataFrame
    """
    # TDD check whether the file exists
    if not isinstance(demand_data, (str, pd.DataFrame)):
        warnings.warn(f"\n  :demand_data is not a string or dataframe: {demand_data}"
                      "\n  :No traffic volume data loaded from input file")
        return None

    if isinstance(demand_data, str):
        if not os.path.isfile(demand_data):
            warnings.warn(f"  :File not found: {demand_data}"
                          "\n  :No traffic volume data loaded from input file")
            return None

        # read the csv file and fill the nan values with 0
        traffic_volume = pd.read_csv(demand_data)
        traffic_volume.fillna(0, inplace=True)

        # Create a copy of the DataFrame
        df_volume = traffic_volume.copy()

    elif isinstance(demand_data, pd.DataFrame):
        # If demand_data is already a DataFrame, use it directly
        df_volume = demand_data.copy()

    # Apply the conversion function to the 'Time' column and create a new 'Seconds' column
    df_volume['IntervalStart'] = df_volume['Time'].apply(time_to_seconds)
    df_volume['IntervalEnd'] = df_volume['IntervalStart'] + 15 * 60
    df_volume = df_volume.drop('Time', axis=1)

    # Reshape the DataFrame to the long format
    df_volume = df_volume.melt(id_vars=['IntersectionName', 'IntervalStart', 'IntervalEnd'],
                               var_name='Turn',
                               value_name='Count')

    # clean Count column: change "" to 0 and convert to int
    df_volume['Count'] = df_volume['Count'].replace("", 0).astype(int)

    # Sort the DataFrame by IntersectionName and Turn columns
    df_volume.sort_values(['IntersectionName', 'IntervalStart', 'IntervalEnd'], inplace=True)

    # Reset the index
    df_volume.reset_index(drop=True, inplace=True)

    return df_volume


def load_traffic_turning_ratio(df_volume: pd.DataFrame) -> pd.DataFrame:
    """load traffic turning ratio data from file

    Args:
        df_volume (pd.DataFrame): the volume data in DataFrame

    Returns:
        pd.DataFrame: the converted turning ratio data in DataFrame
    """

    TurnDfTemp = copy.deepcopy(df_volume)

    # NBT to N and T
    TurnDfTemp['Bound'] = TurnDfTemp['Turn'].str[0]
    TurnDfTemp['Direction'] = TurnDfTemp['Turn'].str[-1]
    TurnDfTemp["IntervalStart"] = TurnDfTemp["IntervalStart"].astype(str)
    TurnDfTemp["IntervalEnd"] = TurnDfTemp["IntervalEnd"].astype(str)
    FlowTemp = TurnDfTemp.groupby(['IntervalStart', 'IntervalEnd', 'IntersectionName', 'Bound'],
                                  as_index=False)['Count'].sum()
    df_turning_ratio = pd.merge(TurnDfTemp, FlowTemp,
                                on=['IntervalStart', 'IntervalEnd',
                                    'IntersectionName', 'Bound'],
                                how='left')
    df_turning_ratio['TurnRatio'] = df_turning_ratio['Count_x'] / \
        df_turning_ratio['Count_y']
    df_turning_ratio = df_turning_ratio.drop(['Count_x', 'Count_y'], axis=1)
    df_turning_ratio.reset_index(drop=True, inplace=True)

    return df_turning_ratio


def load_control_signal(path_signal: str) -> dict:
    """load control signal data from file

    Args:
        path_signal (str): the signal file path

    Returns:
        dict: the converted signal data in dictionary
    """

    # TDD check whether the file exists
    if not isinstance(path_signal, str):
        warnings.warn(f"\n  :File path is not a string: {path_signal}"
                      "\n  :No signal data loaded from input file")
        return None

    if not os.path.isfile(path_signal):
        warnings.warn(f"  :File not found: {path_signal}")
        return None

    # read the signal file
    with open(path_signal, 'r', encoding="utf-8") as file:
        signal = file.readlines()

    SignalDict = {}
    current_table = None
    current_table_data = []
    # Iterate over the lines
    remove_flag = 0
    for line in signal:
        line = line.strip()

        # Check if it's a line to be deleted
        if remove_flag == 1:
            remove_flag = 0
            continue

        # Check if it's a table name
        if line.startswith("["):
            remove_flag = 1
            if current_table is None:
                current_table = line[1:-1]  # Remove the square brackets
            else:
                # Store the previous table data in the dictionary
                SignalDict[current_table] = pd.read_csv(io.StringIO('\n'.join(current_table_data)))

                # Start a new table
                current_table = line[1:-1]  # Remove the square brackets
                current_table_data = []
        else:
            current_table_data.append(line)

    # Store the last table in the dictionary
    SignalDict[current_table] = pd.read_csv(io.StringIO('\n'.join(current_table_data)))

    return SignalDict


class AbstractScenario:
    """Initialize an Abstract Scenario"""

    def __init__(self, input_config: dict = None):

        self.input_config = input_config

        self.Traffic = Traffic()
        self.Network = Network(output_dir=self.input_config.get('output_dir'))
        self.Control = Control()
        self.Application = Application()

    def create_SUMO_network(self):
        """ Create SUMO Network From Vertices"""

        # TDD check whether the input_config is not None
        if not self.input_config:
            warnings.warn("  :input_config is None, no data to update")
            return

        # update Network
        # network_dict = self.input_config.get('Network', None)
        if network_dict := self.input_config.get('Network'):
            self.Network.NetworkName = network_dict.get('NetworkName', "network")
            self.Network.NetworkVertices = network_dict.get('NetworkVertices', "")
            self.Network.ElevationMap = network_dict.get('ElevationMap', "No elevation map provided!")

            # update the OpenDriveNetwork output directory
            self.Network._output_dir = self.input_config.get('output_dir')
            self.Network.OpenDriveNetwork._output_dir = self.Network._output_dir

            # update and crate OpenDriveNetwork
            self.Network.OpenDriveNetwork._net_name = self.Network.NetworkName
            self.Network.OpenDriveNetwork._net_vertices = self.Network.NetworkVertices
            self.Network.OpenDriveNetwork._ele_map = self.Network.ElevationMap
            self.Network.OpenDriveNetwork.create_SUMO_network()

    def create_OpenDrive_network(self):
        """create OpenDriveNetwork object"""
        # TDD check whether the input_config is not None
        if not self.input_config:
            warnings.warn("  :input_config is None, no data to update")
            return

        self.Network.OpenDriveNetwork.create_OpenDrive_network()

        # re-write sumo network based on the OpenDriveNetwork
        net_name = self.Network.OpenDriveNetwork._net_name
        path_open_drive = pf.path2linux(Path(self.Network._output_dir) / f"OpenDrive/{net_name}.xodr")
        path_sumo_net = pf.path2linux(Path(self.Network._output_dir) / f"OpenDrive/{net_name}.net.xml")
        os.system(f'cmd/c "netconvert --opendrive {path_open_drive}'
                  f' -o {path_sumo_net} --no-internal-links"')

    def update_AbstractScenario_from_input(self, df_volume: pd.DataFrame = None, signal_dict: dict = None):
        """ update values from config dict to specific data object"""

        # TDD check whether the input_config is not None
        if not self.input_config:
            warnings.warn("  :input_config is None, no data to update")
            return

        # # update Network
        # # network_dict = self.input_config.get('Network', None)
        # if network_dict := self.input_config.get('Network'):
        #     self.Network.NetworkName = network_dict.get('NetworkName', "network")
        #     self.Network.NetworkVertices = network_dict.get('NetworkVertices', "")
        #     self.Network.ElevationMap = network_dict.get('ElevationMap', "No elevation map provided!")
        #     # update the OpenDriveNetwork output directory
        #     self.Network._output_dir = self.input_config.get('output_dir', "RT_Network")
        #     self.Network.OpenDriveNetwork._output_dir = self.Network._output_dir
        #     # update and crate OpenDriveNetwork
        #     self.Network.OpenDriveNetwork._net_name = self.Network.NetworkName
        #     self.Network.OpenDriveNetwork._net_vertices = self.Network.NetworkVertices
        #     self.Network.OpenDriveNetwork._ele_map = self.Network.ElevationMap
        #     self.Network.OpenDriveNetwork.setValue()

        # update Traffic
        if df_volume is not None:
            self.Traffic.Volume = load_traffic_volume(df_volume)
            self.Traffic.TurningRatio = load_traffic_turning_ratio(self.Traffic.Volume)
        else:
            if traffic_dict := self.input_config.get('Traffic'):
                path_volume = traffic_dict.get('Volume', None)
                path_volume_abs = pf.path2linux(os.path.join(self.input_config.get("input_dir"), path_volume))
                self.Traffic.Volume = load_traffic_volume(path_volume_abs)
                self.Traffic.TurningRatio = load_traffic_turning_ratio(self.Traffic.Volume)

        # update Control
        if signal_dict is not None:
            self.Control.Signal = signal_dict
        else:
            if control_data := self.input_config.get('Control'):
                if isinstance(control_data, dict):
                    path_signal = control_data.get('Signal', None)
                    path_signal_abs = pf.path2linux(os.path.join(self.input_config.get("input_dir"), path_signal))
                    self.Control.Signal = load_control_signal(path_signal_abs)
                elif isinstance(control_data, str):
                    path_signal = control_data
                    path_signal_abs = pf.path2linux(Path(self.input_config.get("input_dir")) / "Control" / path_signal)
                    if os.path.isfile(path_signal_abs):
                        self.Control.Signal = load_control_signal(path_signal_abs)
                    else:
                        raise FileNotFoundError(
                            f"  :File not found: {path_signal_abs}. No signal data loaded from input file")
                else:
                    raise ValueError("  :Invalid control data in configuration file. ")

    def fillAbstractScenario(self):
        """Fill the AbstractScenario with the data from the input_config"""
        pass
