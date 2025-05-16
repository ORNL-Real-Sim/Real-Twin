'''
##############################################################
# Created Date: Friday, May 16th 2025
# Contact Info: luoxiangyong01@gmail.com
# Author/Copyright: Mr. Xiangyong Luo
##############################################################
'''
import os
import pandas as pd


def time_to_seconds(time_str):
    hour, minute = map(int, time_str.split(':'))
    return (hour * 3600) + (minute * 60)


def generate_turn_demand_cali(*, path_matchup_table: str,
                              signal_dir: str,
                              traffic_dir: str) -> list[pd.DataFrame]:
    """ Generate turn demand from user input lookup table and Synchro UTDF files.

    Args:
        path_matchup_table (str): Path to the matchup table with user input.
        signal_dir (str): Directory where Synchro UTDF files are located.
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

    # Load the MatchupTable_OpenDrive_withsignal.xlsx file, skipping the first row for correct headers
    MatchupTable_UserInput = pd.read_excel(path_matchup_table, skiprows=1, dtype=str)

    # Forward fill missing values in merged columns
    merged_columns = ["JunctionID_OpenDrive", "IntersectionName_GridSmart", "File_Synchro", "Need calibration?"]
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
            # Append to the list
            IDRef_list.append(df_lookup)

        # Get the file path from File_GridSmart if available
        file_name = subset["File_GridSmart"].dropna().iloc[0] if not subset["File_GridSmart"].isna().all() else None

        if file_name:
            gs_file_path = os.path.join(traffic_dir, file_name)
            # gs_file_path = f"GridSmart/{file_name}"

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
                    df_data.columns = df_data.columns.to_frame().fillna(method="ffill").agg("".join, axis=1)

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

                    # Fill IntersectionName using IntersectionName_GridSmart from MatchupTable_UserInput
                    intersection_name = subset["IntersectionName_GridSmart"].dropna().iloc[0] if not subset["IntersectionName_GridSmart"].isna().all() else "Unknown"
                    df_data["IntersectionName"] = intersection_name

                    # Append processed data to list
                    TurnDf_list.append(df_data)

    TurnDf = pd.concat(TurnDf_list, ignore_index=True) if TurnDf_list else pd.DataFrame()

    IDRef = pd.concat(IDRef_list, ignore_index=True) if IDRef_list else pd.DataFrame()
    IDRef = IDRef[["IntersectionName", "Turn", "OpenDriveFromID", "OpenDriveToID"]]

    # replace "" to numpy.nan
    # TurnDf = TurnDf.replace("", np.nan)
    # IDRef = IDRef.replace("", np.nan)
    IDRef = IDRef.dropna(subset=['OpenDriveFromID', 'OpenDriveToID'])

    # drop '' in column OpenDriveFromID and OpenDriveToID
    IDRef = IDRef[IDRef["OpenDriveFromID"].astype(str) != ""]
    IDRef = IDRef[IDRef["OpenDriveToID"].astype(str) != ""]

    return [TurnDf, IDRef]


