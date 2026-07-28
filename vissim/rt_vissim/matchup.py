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
"""Generate and read the Vissim MatchupTable.

The MatchupTable is RealTwin's single hand-curated artefact: one row per
(junction, approach, turn), tying three namespaces together.

===========================  ==============================================
Column group                 What it identifies
===========================  ==============================================
``*_Vissim``                 The network: junction, from/to link numbers
``*_GridSmart``              The demand: turn-count file, intersection, turn
``*_Synchro``                The control: UTDF file, INTID, turn
===========================  ==============================================

The layout mirrors
:func:`~realtwin.func_lib._c_abstract_scenario.rt_matchup_table_generation.generate_matchup_table`
-- a merged ``Network`` / ``Demand`` / ``Signal`` banner row, the real header
underneath, and merged cells for the values that are constant within a junction.
The network columns differ: Vissim link numbers replace OpenDRIVE road IDs, and
an extra column records the junction-internal links a movement traverses, which
signal-head placement needs later.

As in the SUMO flow, the demand and signal columns are written **blank** for the
user to fill in.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment

#: Network columns, sourced from :func:`rt_vissim.network.build_movement_table`.
NETWORK_COLUMNS = ["JunctionID_Vissim", "Bearing", "Numbering", "FromLinkNo_Vissim",
                   "ToLinkNo_Vissim", "InternalLinks_Vissim", "Turn"]

#: Demand columns, filled in by the user.
DEMAND_COLUMNS = ["File_GridSmart", "Date_GridSmart", "IntersectionName_GridSmart",
                  "Turn_GridSmart"]

#: Signal columns, filled in by the user.
SIGNAL_COLUMNS = ["File_Synchro", "IntersectionID_Synchro", "Turn_Synchro"]

#: Trailing columns that belong to no group.
OTHER_COLUMNS = ["Need calibration?"]

ALL_COLUMNS = NETWORK_COLUMNS + DEMAND_COLUMNS + SIGNAL_COLUMNS + OTHER_COLUMNS

#: Columns merged vertically across the rows of one junction, and therefore
#: forward-filled on read.
MERGED_COLUMNS = ["JunctionID_Vissim", "File_GridSmart", "Date_GridSmart",
                  "IntersectionName_GridSmart", "File_Synchro",
                  "IntersectionID_Synchro", "Need calibration?"]

#: Column widths, keyed by letter, matching the SUMO table's readability.
COLUMN_WIDTHS = {"A": 20, "B": 12, "C": 12, "D": 20, "E": 20, "F": 24, "G": 12,
                 "H": 22, "I": 16, "J": 32, "K": 16, "L": 20, "M": 24, "N": 14,
                 "O": 18}


def generate_matchup_table(movements: pd.DataFrame,
                           path_output: str | Path = "MatchupTable.xlsx") -> Path:
    """Write a blank-to-fill Vissim MatchupTable from a movement table.

    Args:
        movements: Output of
            :func:`rt_vissim.network.build_movement_table`.  Must carry the
            columns in :data:`NETWORK_COLUMNS`; anything else is ignored.
        path_output: Destination ``.xlsx`` path.

    Returns:
        The written path.

    Raises:
        ValueError: If ``movements`` is empty or missing network columns.
    """
    if movements is None or movements.empty:
        raise ValueError("No movements to write; check the network extraction stage.")
    missing = [c for c in NETWORK_COLUMNS if c not in movements.columns]
    if missing:
        raise ValueError(f"Movement table is missing columns: {missing}")

    path_output = Path(path_output)
    path_output.parent.mkdir(parents=True, exist_ok=True)

    df = movements[NETWORK_COLUMNS].copy()
    for col in DEMAND_COLUMNS + SIGNAL_COLUMNS + OTHER_COLUMNS:
        df[col] = None
    df = df[ALL_COLUMNS]

    wb = Workbook()
    ws = wb.active
    ws.title = "MatchupTable"

    # Row 1: the merged group banner.
    ws.append(["Network"] * len(NETWORK_COLUMNS)
              + ["Demand"] * len(DEMAND_COLUMNS)
              + ["Signal"] * len(SIGNAL_COLUMNS)
              + [""] * len(OTHER_COLUMNS))
    n_net, n_dem, n_sig = len(NETWORK_COLUMNS), len(DEMAND_COLUMNS), len(SIGNAL_COLUMNS)
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=n_net)
    ws.merge_cells(start_row=1, start_column=n_net + 1, end_row=1,
                   end_column=n_net + n_dem)
    ws.merge_cells(start_row=1, start_column=n_net + n_dem + 1, end_row=1,
                   end_column=n_net + n_dem + n_sig)

    # Row 2: the real header.  Data starts at row 3.
    ws.append(ALL_COLUMNS)
    for row in df.itertuples(index=False):
        ws.append([None if pd.isna(v) else v for v in row])

    _merge_junction_blocks(ws, df)

    for row in ws.iter_rows():
        for cell in row:
            cell.alignment = Alignment(horizontal="center", vertical="center")
    for col, width in COLUMN_WIDTHS.items():
        ws.column_dimensions[col].width = width
    ws.freeze_panes = "A3"

    wb.save(path_output)
    return path_output


def _merge_junction_blocks(ws, df: pd.DataFrame) -> None:
    """Vertically merge the cells that are constant within a junction.

    Args:
        ws: The openpyxl worksheet, already populated.
        df: The dataframe that was written, used for the junction boundaries.
    """
    merge_cols = [ALL_COLUMNS.index(c) + 1 for c in MERGED_COLUMNS]
    junction_ids = df["JunctionID_Vissim"].tolist()

    start = 3  # data starts at row 3
    for i, junction_id in enumerate(junction_ids):
        row = i + 3
        is_last = i == len(junction_ids) - 1
        next_id = None if is_last else junction_ids[i + 1]
        if is_last or next_id != junction_id:
            if row > start:  # only merge genuine multi-row blocks
                for col in merge_cols:
                    ws.merge_cells(start_row=start, start_column=col,
                                   end_row=row, end_column=col)
            start = row + 1


# ---------------------------------------------------------------------- #
# Reading it back
# ---------------------------------------------------------------------- #
class MatchupTable:
    """A user-filled Vissim MatchupTable, normalised for downstream use.

    Args:
        path: Path to ``MatchupTable.xlsx``.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If ``path`` is not ``.xlsx`` or lacks the network columns.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"MatchupTable not found: {self.path}")
        if self.path.suffix.lower() != ".xlsx":
            raise ValueError(f"MatchupTable must be .xlsx, got: {self.path}")

        # Row 0 is the group banner; the real header is row 1.
        df = pd.read_excel(self.path, skiprows=1, dtype=str)

        missing = [c for c in NETWORK_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"MatchupTable {self.path.name} is missing columns: {missing}")

        for col in MERGED_COLUMNS:
            if col not in df.columns:
                df[col] = pd.NA
        df[MERGED_COLUMNS] = df[MERGED_COLUMNS].ffill()

        # Hand-edited spreadsheets are full of stray whitespace.
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.strip().replace({"nan": None, "": None})

        self.df = df

    # -- signal ------------------------------------------------------- #
    def synchro_file(self) -> str:
        """Return the Synchro UTDF filename referenced by the table.

        Raises:
            ValueError: If no ``File_Synchro`` value is present.
        """
        vals = [v for v in self.df["File_Synchro"].dropna().unique() if str(v).strip()]
        if not vals:
            raise ValueError(f"No File_Synchro value in {self.path.name}")
        if len(vals) > 1:
            print(f"  :NOTE: multiple File_Synchro values {vals}; using '{vals[0]}'.")
        return str(vals[0])

    def signalized_junctions(self) -> dict[str, str]:
        """Return ``{Vissim junction ID: Synchro INTID}`` for signalised junctions.

        A junction counts as signalised when at least one of its rows carries a
        ``Turn_Synchro`` code -- the rule RealTwin's ``parse_SUMO_TLS_ID`` uses
        to decide which junctions get a signal.
        """
        rows = self.df[self.df["Turn_Synchro"].notna()]
        out: dict[str, str] = {}
        for junction_id, group in rows.groupby("JunctionID_Vissim"):
            intids = [v for v in group["IntersectionID_Synchro"].dropna().unique() if str(v).strip()]
            if not intids:
                print(f"  :WARNING: junction {junction_id} has Turn_Synchro codes but no "
                      "IntersectionID_Synchro; skipping its signal.")
                continue
            out[str(junction_id)] = str(intids[0])
        return out

    # -- demand ------------------------------------------------------- #
    def gridsmart_files(self) -> dict[str, str]:
        """Return ``{Vissim junction ID: GridSmart turn-count filename}``."""
        out: dict[str, str] = {}
        for junction_id, group in self.df.groupby("JunctionID_Vissim"):
            files = [v for v in group["File_GridSmart"].dropna().unique() if str(v).strip()]
            if files:
                out[str(junction_id)] = str(files[0])
        return out

    def turn_lookup(self) -> pd.DataFrame:
        """Return the demand join table: GridSmart turn code to Vissim links.

        Returns:
            Columns ``IntersectionName``, ``Turn``, ``JunctionID_Vissim``,
            ``FromLinkNo_Vissim``, ``ToLinkNo_Vissim`` -- one row per movement
            that carries a GridSmart turn code.  This is the Vissim analogue of
            the ``IDRef`` frame RealTwin builds in ``generate_turn_demand_cali``.
        """
        rows = self.df[self.df["Turn_GridSmart"].notna()].copy()
        out = rows[["IntersectionName_GridSmart", "Turn_GridSmart", "JunctionID_Vissim",
                    "FromLinkNo_Vissim", "ToLinkNo_Vissim"]].copy()
        out.columns = ["IntersectionName", "Turn", "JunctionID_Vissim",
                       "FromLinkNo_Vissim", "ToLinkNo_Vissim"]
        out = out.dropna(subset=["FromLinkNo_Vissim", "ToLinkNo_Vissim"])
        for col in ("FromLinkNo_Vissim", "ToLinkNo_Vissim"):
            out[col] = out[col].astype(float).astype(int)
        return out.reset_index(drop=True)

    def link_numbers(self) -> set[int]:
        """Return every Vissim link number referenced by the table."""
        nos: set[int] = set()
        for col in ("FromLinkNo_Vissim", "ToLinkNo_Vissim"):
            for v in self.df[col].dropna():
                try:
                    nos.add(int(float(v)))
                except (TypeError, ValueError):
                    continue
        return nos

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"MatchupTable(path={self.path.name!r}, rows={len(self.df)}, "
                f"junctions={self.df['JunctionID_Vissim'].nunique()})")
