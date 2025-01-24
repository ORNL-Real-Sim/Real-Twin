'''
class to host a unique AbstractScenario definition
'''

# import four elements of AbstractScenario
from ._traffic import Traffic
from ._network import Network
from ._control import Control
from ._application import Application

import pandas as pd
import os
import warnings
import io
import pyufunc as pf


def time_to_seconds(time_str) -> int:
    """convert time string to seconds

    Args:
        time_str (str): the time string in format 'HH:MM'
    """
    # hour, minute = map(int, time_str.split(':'))
    hour, minute = [int(x) for x in time_str.split(':')]
    return (hour * 3600) + (minute * 60)


def load_traffic_volume(path_demand: str) -> pd.DataFrame:
    """load traffic volume data from file

    Args:
        path_demand (str): the demand file path

    Returns:
        pd.DataFrame: the converted demand data in DataFrame
    """
    # TDD check whether the file exists
    if not isinstance(path_demand, str):
        warnings.warn(f"\n  :File path is not a string: {path_demand}"
                      "\n  :No traffic volume data loaded from input file")
        return None

    if not os.path.exists(path_demand):
        warnings.warn(f"\n  :File not found: {path_demand}")
        return None

    # read the csv file and fill the nan values with 0
    traffic_volume = pd.read_csv(path_demand)
    traffic_volume.fillna(0, inplace=True)

    # Create a copy of the DataFrame
    df_volume = traffic_volume.copy()

    # Apply the conversion function to the 'Time' column and create a new 'Seconds' column
    df_volume['IntervalStart'] = df_volume['Time'].apply(time_to_seconds)
    df_volume['IntervalEnd'] = df_volume['IntervalStart'] + 15 * 60
    df_volume = df_volume.drop('Time', axis=1)

    # Reshape the DataFrame to the long format
    df_volume = df_volume.melt(id_vars=['IntersectionName', 'IntervalStart', 'IntervalEnd'],
                               var_name='Turn',
                               value_name='Count')

    # Sort the DataFrame by IntersectionName and Turn columns
    df_volume.sort_values(['IntersectionName', 'IntervalStart', 'IntervalEnd'], inplace=True)

    # Reset the index
    df_volume.reset_index(drop=True, inplace=True)

    return df_volume


def load_traffic_turning_ratio(path_turning_ratio: str) -> pd.DataFrame:
    """load traffic turning ratio data from file

    Args:
        path_turning_ratio (str): the turning ratio file path

    Returns:
        pd.DataFrame: the converted turning ratio data in DataFrame
    """

    # TDD check whether the file exists
    if not isinstance(path_turning_ratio, str):
        warnings.warn(f"\n  :File path is not a string: {path_turning_ratio}"
                      "\n  :No traffic turning ratio data loaded from input file")
        return None

    if not os.path.exists(path_turning_ratio):
        warnings.warn(f"  :File not found: {path_turning_ratio}")
        return None

    # read the csv file and fill the nan values with 0
    turning_ratio = pd.read_csv(path_turning_ratio)
    turning_ratio.fillna(0, inplace=True)

    # Create a copy of the DataFrame
    TurnDf = turning_ratio.copy()

    # Apply the conversion function to the 'Time' column and create a new 'Seconds' column
    TurnDf['IntervalStart'] = TurnDf['Time'].apply(time_to_seconds)
    TurnDf['IntervalEnd'] = TurnDf['IntervalStart'] + 15 * 60
    TurnDf = TurnDf.drop('Time', axis=1)

    # Reshape the DataFrame to the long format
    TurnDfTemp = TurnDf.melt(id_vars=['IntersectionName', 'IntervalStart', 'IntervalEnd'],
                             var_name='Turn',
                             value_name='Count')
    # Sort the DataFrame by IntersectionName and Turn columns
    TurnDfTemp.sort_values(['IntersectionName', 'IntervalStart', 'IntervalEnd'], inplace=True)

    # Reset the index
    TurnDfTemp.reset_index(drop=True, inplace=True)

    # NBT to N and T
    TurnDfTemp['Bound'] = TurnDfTemp['Turn'].str[0]
    TurnDfTemp['Direction'] = TurnDfTemp['Turn'].str[-1]
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

    if not os.path.exists(path_signal):
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

    def __init__(self, config_dict: dict = None):

        self.config_dict = config_dict

        self.Traffic = Traffic()
        self.Network = Network()
        self.Control = Control()
        self.Application = Application()

    def update_AbstractScenario_from_input(self):
        """update values from config dict to specific data object"""

        # TDD check whether the config_dict is not None
        if not self.config_dict:
            warnings.warn("  :config_dict is None, no data to update")
            return

        # update Traffic
        # traffic_dict = self.config_dict.get('Traffic', None)
        if traffic_dict := self.config_dict.get('Traffic'):
            path_volume = traffic_dict.get('Volume', None)
            path_volume_abs = pf.path2linux(os.path.join(self.config_dict.get("input_dir"), path_volume))
            path_turning_ratio = traffic_dict.get('TurningRatio', None)
            path_turning_ratio_abs = pf.path2linux(os.path.join(self.config_dict.get("input_dir"), path_turning_ratio))

            self.Traffic.Volume = load_traffic_volume(path_volume_abs)
            self.Traffic.TurningRatio = load_traffic_turning_ratio(path_turning_ratio_abs)

        # update Network
        # network_dict = self.config_dict.get('Network', None)
        if network_dict := self.config_dict.get('Network'):
            self.Network.NetworkName = network_dict.get('NetworkName', "network")
            self.Network.NetworkVertices = network_dict.get('NetworkVertices', "")
            self.Network.ElevationMap = network_dict.get('ElevationMap', "No elevation map provided!")

            # update the OpenDriveNetwork output directory
            self.Network._output_dir = self.config_dict.get('output_dir', "RT_Network")
            self.Network.OpenDriveNetwork._output_dir = self.Network._output_dir

            # update and crate OpenDriveNetwork
            self.Network.OpenDriveNetwork._net_name = self.Network.NetworkName
            self.Network.OpenDriveNetwork._net_vertices = self.Network.NetworkVertices
            self.Network.OpenDriveNetwork._ele_map = self.Network.ElevationMap
            self.Network.OpenDriveNetwork.setValue()

        # update Control
        # control_dict = self.config_dict.get('Control', None)
        if control_dict := self.config_dict.get('Control'):
            path_signal = control_dict.get('Signal', None)
            path_signal_abs = pf.path2linux(os.path.join(self.config_dict.get("input_dir"), path_signal))
            self.Control.Signal = load_control_signal(path_signal_abs)

        # update Application

        # self.dataObjDict['Network']['OpenDriveNetwork'].OpenDriveNetwork = [
        #     f'MyNetwork/OpenDrive/{Name}.xodr',
        #     f'MyNetwork/OpenDrive/{Name}_WithElevation.xodr']

    def fillAbstractScenario(self):
        """Fill the AbstractScenario with the data from the config_dict"""
        pass
