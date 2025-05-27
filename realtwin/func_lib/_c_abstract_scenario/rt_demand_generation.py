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
import io
import re
import os
from pathlib import Path
import numpy as np
import pandas as pd


def process_signal_from_utdf(file_utdf: object) -> dict[str, pd.DataFrame]:
    """Process the signal data and return a dictionary of DataFrames.

    Args:
        file_lines (object): An iterable containing lines of the signal data from Synchro Signal file.

    Returns:
        dict[str, pd.DataFrame]: A dictionary where keys are table names and values
        are DataFrames containing the data from those tables.
    """

    SignalDict = {}
    current_table = None
    current_table_data = []

    with open(file_utdf, 'r') as f:
        file_lines = f.readlines()

    removal_flag = 0
    for line in file_lines:
        line = line.strip()

        # Check if it's a line to be skipped
        if removal_flag == 1:
            removal_flag = 0
            continue

        # Check if it's a table name (indicated by square brackets)
        if line.startswith("["):
            removal_flag = 1
            if current_table is None:
                # Remove square brackets and ]
                current_table = line[1:-1].split(',')[0].rstrip(']')
            else:
                # Store the previous table data in the dictionary
                if current_table_data:  # Check if there's data to store
                    # Read data as strings
                    df = pd.read_csv(io.StringIO(
                        '\n'.join(current_table_data)), dtype=str)
                    SignalDict[current_table] = df

                # Start a new table
                # Remove the square brackets and extra commas
                current_table = line[1:-1].split(',')[0].rstrip(']')
                current_table_data = []
        else:
            # Accumulate table data
            current_table_data.append(line)

    # Store the last table in the dictionary
    if current_table_data:
        SignalDict[current_table] = pd.read_csv(
            io.StringIO('\n'.join(current_table_data)), dtype=str)

    return SignalDict


# Helper function for sorting: extract the alphabetic base and numeric suffix
def sort_key(movement: str) -> tuple[int, int]:
    """Extracts the base and suffix from a movement string for sorting."""

    base_order = [
        'NBR', 'NBT', 'NBL', 'NER', 'NET', 'NEL',
        'EBR', 'EBT', 'EBL', 'SER', 'SET', 'SEL',
        'SBR', 'SBT', 'SBL', 'SWR', 'SWT', 'SWL',
        'WBR', 'WBT', 'WBL', 'NWR', 'NWT', 'NWL'
    ]
    match = re.match(r"([A-Z]+)(\d*)", movement)
    if match:
        base = match.group(1)
        suffix_str = match.group(2)
        suffix = int(suffix_str) if suffix_str else 0
        try:
            base_index = base_order.index(base)
        except ValueError:
            base_index = len(base_order)
        return (base_index, suffix)
    return (len(base_order), 0)


# Helper function: determine if a value is missing or equivalent to 0 (as a number or string).
def is_missing_or_zero(val) -> bool:
    """Check if a value is missing or equivalent to zero."""

    if pd.isna(val):
        return True
    val_str = str(val).strip()
    if val_str == "" or val_str == "0":
        return True

    try:
        if float(val_str) == 0:
            return True
    except ValueError:
        pass
    return False


def update_matchup_table(path_matchup_table: str, control_dir: str = "", traffic_dir: str = "") -> bool:
    """ Update the match table with data from user prepared demands and Synchro UTDF file.

    Args:
        path_matchup_table (str): the match table that contains the user input.
        control_dir (str, optional): the directory where the Synchro UTDF files are located.
        Defaults to "", which means the current directory.

    Returns:
        bool: True if the matchup table is updated successfully, False otherwise.
    """

    # read the lookup table
    MatchupTable_UserInput = pd.read_excel(path_matchup_table, skiprows=1, dtype=str)

    # Forward fill missing values in merged columns
    merged_columns = ["JunctionID_OpenDrive", "File_Synchro"]
    MatchupTable_UserInput[merged_columns] = MatchupTable_UserInput[merged_columns].ffill()
    MatchupTable_UserInput["Need calibration?"] = "Y"

    movement_order = ["NBR", "NBT", "NBL", "NBU", "EBR", "EBT", "EBL", "EBU",
                      "SBR", "SBT", "SBL", "SBU", "WBR", "WBT", "WBL", "WBU"]

    # Process each unique JunctionID_OpenDrive where File_GridSmart has input
    for junction_id in MatchupTable_UserInput["JunctionID_OpenDrive"].unique():
        subset = MatchupTable_UserInput[MatchupTable_UserInput["JunctionID_OpenDrive"] == junction_id]

        # Get the file path from File_GridSmart if available
        file_name = subset["File_GridSmart"].dropna().iloc[0] if not subset["File_GridSmart"].isna().all() else None
        if file_name:
            # Set Need calibration? to "N"
            MatchupTable_UserInput.loc[MatchupTable_UserInput["JunctionID_OpenDrive"]
                                       == junction_id, "Need calibration?"] = "N"

            # Load the GridSmart file
            gs_file_path = Path(traffic_dir) / file_name
            gs_data = pd.read_excel(gs_file_path, header=None, dtype=str)

            # Extract IntersectionName_GridSmart
            intersection_row = gs_data[gs_data.iloc[:, 0] == "Intersection"].index
            if not intersection_row.empty:
                intersection_col = gs_data.iloc[intersection_row[0], 1:].first_valid_index()
                if intersection_col is not None:
                    intersection_name = gs_data.iloc[intersection_row[0], intersection_col]
                    MatchupTable_UserInput.loc[MatchupTable_UserInput["JunctionID_OpenDrive"]
                                               == junction_id, "IntersectionName_GridSmart"] = intersection_name

            # Extract Date_GridSmart
            date_row = gs_data[gs_data.iloc[:, 0] == "Date"].index
            if not date_row.empty:
                date_col = gs_data.iloc[date_row[0], 1:].first_valid_index()
                if date_col is not None:
                    date_value = gs_data.iloc[date_row[0], date_col]
                    MatchupTable_UserInput.loc[MatchupTable_UserInput["JunctionID_OpenDrive"]
                                               == junction_id, "Date_GridSmart"] = date_value

            # Identify movement columns outside loop
            movement_columns = {}
            for col in gs_data.columns[1:]:
                for movement in ["Right", "Through", "Left"]:
                    if gs_data.iloc[:, col].eq(movement).any():
                        movement_columns[movement] = col

            # Extract Turning Movements
            GridSmartTurns = []

            for direction, prefix in zip(["Northbound", "Eastbound", "Southbound", "Westbound"],
                                         ["NB", "EB", "SB", "WB"]):
                direction_row = gs_data[gs_data.iloc[:, 0] == direction].index
                if not direction_row.empty:
                    movement_row = direction_row[0]

                    # Check if movements exist and add to list
                    if "Right" in movement_columns and pd.notna(gs_data.iloc[movement_row,
                                                                             movement_columns["Right"]]):
                        GridSmartTurns.append(f"{prefix}R")
                    if "Through" in movement_columns and pd.notna(gs_data.iloc[movement_row,
                                                                               movement_columns["Through"]]):
                        GridSmartTurns.append(f"{prefix}T")
                    if "Left" in movement_columns and pd.notna(gs_data.iloc[movement_row,
                                                                            movement_columns["Left"]]):
                        GridSmartTurns.append(f"{prefix}L")
                        GridSmartTurns.append(f"{prefix}U")

            # Sort GridSmartTurns
            GridSmartTurns = sorted(
                GridSmartTurns,
                key=lambda x: movement_order.index(x) if x in movement_order else len(movement_order))

            # Fill Turn_GridSmart column
            rows_to_fill = subset.index.tolist()
            for i, turn in enumerate(GridSmartTurns):
                if i < len(rows_to_fill):
                    MatchupTable_UserInput.at[rows_to_fill[i], "Turn_GridSmart"] = turn
                else:
                    print(f'  :There are more turning movements in {file_name} '
                          f'than OpenDrive junction: {junction_id}.')

    # Initialize a cache for Synchro files to avoid re-reading files
    synchro_cache = {}

    # Process each unique JunctionID_OpenDrive for File_Synchro
    for junction_id in MatchupTable_UserInput["JunctionID_OpenDrive"].unique():
        subset = MatchupTable_UserInput[MatchupTable_UserInput["JunctionID_OpenDrive"] == junction_id]

        # Get the File_Synchro value (if provided)
        file_synchro_name = subset["File_Synchro"].dropna(
        ).iloc[0] if not subset["File_Synchro"].isna().all() else None
        if subset["IntersectionID_Synchro"].dropna().empty:
            continue
        else:
            intersection_id_synchro = subset["IntersectionID_Synchro"].dropna().iloc[0]
            # Use the cache if this file was already read
            if file_synchro_name in synchro_cache:
                signal_dict = synchro_cache[file_synchro_name]
            else:
                synchro_file_path = f"{os.path.join(control_dir, file_synchro_name)}"
                signal_dict = process_signal_from_utdf(synchro_file_path)
                synchro_cache[file_synchro_name] = signal_dict

            # Ensure the 'Lanes' table exists
            lanes_df = signal_dict.get('Lanes')
            if lanes_df is None:
                print(f'No "Lanes" table found in "{file_synchro_name}".')
                continue

            # 2.1: Subset rows where INTID equals IntersectionID_Synchro and RECORDNAME is in the allowed list.
            allowed_recordnames = ["Lanes", "Shared", "Phase1",
                                   "PermPhase1", "Phase2", "PermPhase2", "Phase3", "PermPhase3"]
            subset_lanes = lanes_df[
                (lanes_df['INTID'].astype(str) == str(intersection_id_synchro)) & (
                    lanes_df['RECORDNAME'].astype(str).isin(allowed_recordnames))
            ]

            if subset_lanes.empty:
                print(f'No matching records in Lanes for IntersectionID_Synchro '
                      f'{intersection_id_synchro} in file {file_synchro_name}.')
                continue

            # 2.2: Reindex the row numbering (reset the index)
            subset_lanes.reset_index(drop=True, inplace=True)

            # Drop columns based on the condition applied to row 1 (first row, index 0)
            cols_to_drop = []
            for col in subset_lanes.columns:
                # Get the value in the first row (row 1)
                val_first = subset_lanes.at[0, col]
                if is_missing_or_zero(val_first):
                    # Exception 1: if any value from row 3 onward (i.e. index 2 and beyond) is valid, we keep the column.
                    if subset_lanes.shape[0] > 2:
                        subsequent_valid = subset_lanes[col].iloc[2:].apply(
                            lambda x: not is_missing_or_zero(x)).any()
                    else:
                        subsequent_valid = False

                    # Exception 2: for any column ending with 'R', check if the corresponding XYT column meets the criteria.
                    exception2_keep = False
                    if col.endswith("R"):
                        col_t = col[:-1] + "T"
                        if col_t in subset_lanes.columns and subset_lanes.shape[0] > 1:
                            try:
                                val_first_t = float(subset_lanes.at[0, col_t])
                                val_second_t = float(
                                    subset_lanes.at[1, col_t]) if subset_lanes.shape[0] > 1 else 0
                                if val_first_t > 0 and val_second_t > 1:
                                    exception2_keep = True
                            except ValueError:
                                pass

                    # Drop the column only if neither exception applies.
                    if not subsequent_valid and not exception2_keep:
                        cols_to_drop.append(col)

            # Drop the columns that do not meet the exceptions
            subset_lanes = subset_lanes.copy()
            subset_lanes.drop(columns=cols_to_drop, inplace=True)

            # Finally, get the column names and save them in the variable 'movements',
            # while excluding the unwanted columns
            movements = [col for col in subset_lanes.columns if col not in [
                'RECORDNAME', 'INTID', 'PED', 'HOLD']]

            # Sort the original movements using the sort_key
            sorted_movements = sorted(movements, key=sort_key)

            # For each sorted movement that ends with 'L', immediately add the variant (replace ending 'L' with 'U')
            enhanced_movements = []
            for mov in sorted_movements:
                enhanced_movements.append(mov)
                if mov.endswith("L"):
                    variant = mov[:-1] + "U"
                    enhanced_movements.append(variant)

            # Fill the Turn_Synchro column for the current JunctionID_OpenDrive
            rows_to_fill = subset.index.tolist()
            for i, movement in enumerate(enhanced_movements):
                if i < len(rows_to_fill):
                    MatchupTable_UserInput.at[rows_to_fill[i], "Turn_Synchro"] = movement
                else:
                    print(f'  :There are more turning movements in {file_synchro_name} '
                          f'than OpenDrive junction {junction_id}.')
    return MatchupTable_UserInput


def generate_turn_demand(*, path_matchup_table: str | pd.DataFrame,
                         control_dir: str,
                         traffic_dir: str, output_dir: str = "") -> list[pd.DataFrame]:
    """ Generate turn demand from user input lookup table and Synchro UTDF files.

    Args:
        path_matchup_table (str): Path to the matchup table with user input.
        control_dir (str): Directory where Synchro UTDF files are located.
            Defaults to "", which means the current directory.
        output_dir (str): Directory to save the output files.
            Defaults to "", which means the current directory.
        demand_dir (str): Directory where demand files are located.
            Defaults to "Traffic".

    See Also:
        demand_dir: check sample demand files in datasets/Traffic directory

    Example:
        >>> path_matchup_table = "./MatchupTable_OpenDrive_with user input.xlsx"
        >>> TurnDf, IDRef = generate_turn_demand(path_matchup_table, signal_dir="",
            output_dir="./Output", demand_dir="Traffic")

    Returns:
        list[pd.DataFrame]: A list containing two DataFrames:
            - TurnDf: DataFrame with turn demand data.
            - IDRef: DataFrame with reference IDs for OpenDrive turns (demand lookup table).
    """

    if isinstance(path_matchup_table, str):
        # get the updated lookup table
        try:
            MatchupTable_UserInput = update_matchup_table(path_matchup_table, control_dir, traffic_dir)
        except Exception as e:
            raise Exception(f"Error loading user updated lookup table: {e}")
    elif isinstance(path_matchup_table, pd.DataFrame):
        # Use the provided DataFrame directly
        MatchupTable_UserInput = path_matchup_table
    else:
        raise ValueError("path_matchup_table must be a string or a DataFrame.")

    merged_columns = ["JunctionID_OpenDrive", "File_Synchro"]
    MatchupTable_UserInput[merged_columns] = MatchupTable_UserInput[merged_columns].ffill()

    turn_values = ["NBR", "NBT", "NBL", "NBU", "EBR", "EBT", "EBL", "EBU",
                   "SBR", "SBT", "SBL", "SBU", "WBR", "WBT", "WBL", "WBU"]
    TurnDf_list = []
    IDRef_list = []

    # Process each unique JunctionID_OpenDrive where File_GridSmart has input
    for junction_id in MatchupTable_UserInput["JunctionID_OpenDrive"].unique():
        subset = MatchupTable_UserInput[MatchupTable_UserInput["JunctionID_OpenDrive"] == junction_id]

        # Get the file path from File_GridSmart if available
        file_name = subset["File_GridSmart"].dropna().iloc[0] if not subset["File_GridSmart"].isna().all() else None

        if file_name:
            # Retrieve the IntersectionName_GridSmart
            intersection_name = subset["IntersectionName_GridSmart"].dropna().iloc[0] if not subset["IntersectionName_GridSmart"].isna().all() else "Unknown"
            # Create df_lookup with predefined Turn values
            df_lookup = pd.DataFrame({"Turn": turn_values})
            # Assign IntersectionName
            df_lookup["IntersectionName"] = intersection_name
            # Initialize OpenDriveFromID and OpenDriveToID as empty strings
            df_lookup["OpenDriveFromID"] = ""
            df_lookup["OpenDriveToID"] = ""
            # Map values from subset based on Turn_GridSmart
            for idx, row in df_lookup.iterrows():
                turn = row["Turn"]
                match = subset[subset["Turn_GridSmart"] == turn]
                if not match.empty:
                    df_lookup.at[idx, "OpenDriveFromID"] = match["FromRoadID_OpenDrive"].values[0] if not match["FromRoadID_OpenDrive"].isna().all() else ""
                    df_lookup.at[idx, "OpenDriveToID"] = match["ToRoadID_OpenDrive"].values[0] if not match["ToRoadID_OpenDrive"].isna().all() else ""

            IDRef_list.append(df_lookup)

        # Get the file path from File_GridSmart if available
        file_name = subset["File_GridSmart"].dropna().iloc[0] if not subset["File_GridSmart"].isna().all() else None

        if file_name:
            gs_file_path = os.path.join(traffic_dir, file_name)
            # gs_file_path = f"RealTwinDemand/{file_name}"

            # Check if the file exists before processing
            if os.path.exists(gs_file_path):
                # Load the Excel file
                df = pd.read_excel(gs_file_path, header=None)  # Read without predefined headers

                # Find the first row that contains a time value in the first column
                time_row_index = df[df[0].astype(str).str.match(r'^\d{1,2}:\d{2}$', na=False)].index.min()

                # Determine the starting row for data extraction
                start_row = time_row_index - 2 if pd.notna(time_row_index) else None

                if start_row is not None:
                    # Read the file again with proper headers starting from the determined row
                    df_data = pd.read_excel(gs_file_path, header=[start_row, start_row + 1])

                    # Fill merged cells in the first row
                    df_data.columns = df_data.columns.to_frame().ffill().agg("".join, axis=1)

                    # Remove spaces from column names
                    df_data.columns = [col.replace(" ", "") for col in df_data.columns]

                    # Rename the first column to "Time"
                    df_data.rename(columns={df_data.columns[0]: "Time"}, inplace=True)

                    # Drop fully empty columns
                    df_data.dropna(axis=1, how='all', inplace=True)

                    # Drop data of "Total"
                    df_data = df_data[df_data["Time"] != "Total"]

                    # Convert data columns to numeric
                    for col in df_data.columns[1:]:  # Excluding 'Time' column
                        df_data[col] = pd.to_numeric(df_data[col], errors='coerce').fillna(0).astype(int)

                    # Remove 'Unassigned' columns
                    df_data = df_data.loc[:, ~df_data.columns.str.contains(r'Unassigned', na=False)]

                    # Standardize column names for directions
                    df_data.columns = [col.replace("Northbound", "NB")
                                       .replace("Southbound", "SB")
                                       .replace("Westbound", "WB")
                                       .replace("Eastbound", "EB") for col in df_data.columns]

                    # Define expected columns and ensure all are present
                    expected_columns = ["IntersectionName", "Time", "NBR", "NBT", "NBL", "NBU",
                                        "EBR", "EBT", "EBL", "EBU", "SBR", "SBT", "SBL", "SBU",
                                        "WBR", "WBT", "WBL", "WBU"]
                    df_data = df_data.reindex(columns=expected_columns, fill_value="")
                    # df_data = df_data.reindex(columns=expected_columns, fill_value=None)

                    # Fill IntersectionName using IntersectionName_GridSmart from MatchupTable_UserInput
                    if not subset["IntersectionName_GridSmart"].isna().all():
                        intersection_name = subset["IntersectionName_GridSmart"].dropna().iloc[0]
                    else:
                        intersection_name = "Unknown"
                    df_data["IntersectionName"] = intersection_name

                    # Append processed data to list
                    TurnDf_list.append(df_data)

    TurnDf = pd.concat(TurnDf_list, ignore_index=True) if TurnDf_list else pd.DataFrame()

    IDRef = pd.concat(IDRef_list, ignore_index=True) if IDRef_list else pd.DataFrame()
    IDRef = IDRef[["IntersectionName", "Turn", "OpenDriveFromID", "OpenDriveToID"]]

    if output_dir:
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)
        # Save the DataFrame to a CSV file
        TurnDf.to_excel(os.path.join(output_dir, "GridSmart_demand.xlsx"), index=False)
        IDRef.to_excel(os.path.join(output_dir, "GridSmart_lookuptable.xlsx"), index=False)

    # replace "" to numpy.nan
    # TurnDf = TurnDf.replace("", np.nan)
    # IDRef = IDRef.replace("", np.nan)
    IDRef = IDRef.dropna(subset=['OpenDriveFromID', 'OpenDriveToID'])

    # drop '' in column OpenDriveFromID and OpenDriveToID
    IDRef = IDRef[IDRef["OpenDriveFromID"].astype(str) != ""]
    IDRef = IDRef[IDRef["OpenDriveToID"].astype(str) != ""]
    return [TurnDf, IDRef]


if __name__ == "__main__":
    path_matchup_table = "./MatchupTable_OpenDrive_with user input.xlsx"
    output_dir = "./"
    TurnDf, IDRef = generate_turn_demand(path_matchup_table, output_dir="./")
