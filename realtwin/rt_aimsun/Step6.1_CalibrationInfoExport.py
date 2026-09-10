import sys

from PyANGBasic import *
from PyANGKernel import *
from PyANGConsole import *

def main( argv ):
    # Start a Console
    console = ANGConsole()
    # Load a network
    if console.open( argv[1] ):
        model = console.getModel()

        import os
        import json

        # External site-packages providing pandas, xlrd and openpyxl.
        try:
            _SITEPACKAGES = r"C:\Users\xh8\AppData\Local\Programs\Python\Python310\Lib\site-packages"
        except Exception:
            input_config = json.loads(argv[-1])
            _SITEPACKAGES = input_config["AIMSUN"]["site_packages"]

        if _SITEPACKAGES not in sys.path:
            sys.path.append(_SITEPACKAGES)

        import pandas as pd

        # Folder of the loaded .ang; holds MatchupTable.xlsx, Traffic and the output json
        INPUT_DIR = os.path.dirname(str(model.getDocumentFileName()))

        MATCHUP_FILE = "MatchupTable.xlsx"
        TRAFFIC_SUBDIR = "Traffic"
        OUTPUT_JSON = "calibration_info.json"

        # Simulation window (seconds from midnight) and bin size
        sim_start_time = 28800
        sim_end_time = 32400
        INTERVAL_SECONDS = 15 * 60

        # Name of the replication to run during calibration
        REPLICATION_NAME = "RealTwin Replication"

        # GridSmart movement columns
        TURN_VALUES = ["NBR", "NBT", "NBL", "NBU", "EBR", "EBT", "EBL", "EBU",
                       "SBR", "SBT", "SBL", "SBU", "WBR", "WBT", "WBL", "WBU"]


        def time_to_seconds(time_str):
            hour, minute = [int(x) for x in str(time_str).split(":")]
            return hour * 3600 + minute * 60


        def read_matchup_table(path_matchup_table):
            matchup = pd.read_excel(path_matchup_table, skiprows=1, dtype=str)
            legacy = {"JunctionID_OpenDrive": "JunctionID_Aimsun",
                      "FromRoadID_OpenDrive": "FromRoadID_Aimsun",
                      "ToRoadID_OpenDrive": "ToRoadID_Aimsun"}
            rename = {old: new for old, new in legacy.items()
                      if old in matchup.columns and new not in matchup.columns}
            if rename:
                matchup = matchup.rename(columns=rename)
            merged_columns = ["JunctionID_Aimsun", "IntersectionName_GridSmart",
                              "File_Synchro", "Need calibration?"]
            for col in merged_columns:
                if col not in matchup.columns:
                    matchup[col] = pd.NA
            matchup[merged_columns] = matchup[merged_columns].ffill()
            return matchup


        def generate_turn_demand(matchup, traffic_dir):
            turn_df_list = []
            id_ref_list = []
            for junction_id in matchup["JunctionID_Aimsun"].dropna().unique():
                subset = matchup[matchup["JunctionID_Aimsun"] == junction_id]
                if subset["File_GridSmart"].isna().all():
                    continue
                file_name = subset["File_GridSmart"].dropna().iloc[0]
                intersection_name = (subset["IntersectionName_GridSmart"].dropna().iloc[0]
                                     if not subset["IntersectionName_GridSmart"].isna().all()
                                     else "Unknown")
                df_lookup = pd.DataFrame({"Turn": TURN_VALUES})
                df_lookup["IntersectionName"] = intersection_name
                df_lookup["JunctionID"] = str(junction_id)
                df_lookup["FromID"] = ""
                df_lookup["ToID"] = ""
                for idx, row in df_lookup.iterrows():
                    match = subset[subset["Turn_GridSmart"] == row["Turn"]]
                    if not match.empty:
                        if not match["FromRoadID_Aimsun"].isna().all():
                            df_lookup.at[idx, "FromID"] = match["FromRoadID_Aimsun"].values[0]
                        if not match["ToRoadID_Aimsun"].isna().all():
                            df_lookup.at[idx, "ToID"] = match["ToRoadID_Aimsun"].values[0]
                id_ref_list.append(df_lookup)

                gs_file_path = os.path.join(traffic_dir, file_name)
                if not os.path.exists(gs_file_path):
                    print(f"  :GridSmart file not found, skipping: {gs_file_path}")
                    continue
                df = pd.read_excel(gs_file_path, header=None)
                time_mask = df[0].astype(str).str.match(r"^\d{1,2}:\d{2}$", na=False)
                time_row_index = df[time_mask].index.min()
                if pd.isna(time_row_index):
                    print(f"  :No time rows found in {gs_file_path}, skipping.")
                    continue
                start_row = time_row_index - 2
                df_data = pd.read_excel(gs_file_path, header=[start_row, start_row + 1])
                df_data.columns = df_data.columns.to_frame().ffill().agg("".join, axis=1)
                df_data.columns = [c.replace(" ", "") for c in df_data.columns]
                df_data.rename(columns={df_data.columns[0]: "Time"}, inplace=True)
                df_data.dropna(axis=1, how="all", inplace=True)
                df_data = df_data[df_data["Time"] != "Total"]
                for col in df_data.columns[1:]:
                    df_data[col] = pd.to_numeric(df_data[col], errors="coerce").fillna(0).astype(int)
                df_data = df_data.loc[:, ~df_data.columns.str.contains(r"Unassigned", na=False)]
                df_data.columns = [c.replace("Northbound", "NB").replace("Southbound", "SB")
                                    .replace("Westbound", "WB").replace("Eastbound", "EB")
                                   for c in df_data.columns]
                expected_columns = ["IntersectionName", "Time"] + TURN_VALUES
                df_data = df_data.reindex(columns=expected_columns, fill_value="")
                df_data["IntersectionName"] = intersection_name
                turn_df_list.append(df_data)

            turn_df = pd.concat(turn_df_list, ignore_index=True) if turn_df_list else pd.DataFrame()
            id_ref = pd.concat(id_ref_list, ignore_index=True) if id_ref_list else pd.DataFrame()
            if not id_ref.empty:
                id_ref = id_ref[["IntersectionName", "JunctionID", "Turn", "FromID", "ToID"]]
                id_ref = id_ref[(id_ref["FromID"].astype(str) != "")
                                & (id_ref["ToID"].astype(str) != "")]
            return turn_df, id_ref


        def build_movement_counts(turn_df, id_ref, sim_begin, sim_end):
            if turn_df.empty or id_ref.empty:
                return pd.DataFrame(columns=["IntervalStart", "JunctionID", "FromID", "ToID", "Count"])
            df = turn_df.copy()
            df["IntervalStart"] = df["Time"].apply(time_to_seconds)
            df["IntervalEnd"] = df["IntervalStart"] + INTERVAL_SECONDS
            df = df.drop(columns=["Time"])
            long_df = df.melt(id_vars=["IntersectionName", "IntervalStart", "IntervalEnd"],
                              var_name="Turn", value_name="Count")
            long_df["Count"] = pd.to_numeric(long_df["Count"], errors="coerce").fillna(0).astype(int)
            long_df = long_df[(long_df["IntervalStart"] >= sim_begin)
                              & (long_df["IntervalEnd"] <= sim_end)]
            ref = id_ref.astype(str)
            merged = long_df.merge(ref, on=["IntersectionName", "Turn"], how="left")
            merged = merged.dropna(subset=["FromID", "ToID"])
            merged = merged[(merged["FromID"] != "") & (merged["ToID"] != "")]
            movements = (merged.groupby(["IntervalStart", "JunctionID", "FromID", "ToID"],
                                        as_index=False)["Count"]
                         .sum())
            movements = movements[movements["Count"] > 0]
            return movements.reset_index(drop=True)


        def _iter_objects(catalog, gktype):
            try:
                for sub in catalog.getUsedSubTypesFromType(gktype):
                    if sub is None:
                        continue
                    for obj in sub.values():
                        if obj is not None:
                            yield obj
            except Exception:
                d = catalog.getObjectsByType(gktype)
                if d:
                    for obj in d.values():
                        if obj is not None:
                            yield obj


        def _node_turnings(node):
            try:
                ts = node.getTurnings()
            except AttributeError:
                return []
            if ts is None:
                return []
            try:
                return list(ts)
            except TypeError:
                return [ts[i] for i in range(len(ts))]


        def build_topology(model):
            catalog = model.getCatalog()
            node_type = model.getType("GKNode")
            upstream = {}
            origins = set()
            for node in _iter_objects(catalog, node_type):
                for turning in _node_turnings(node):
                    try:
                        origin = turning.getOrigin()
                        destination = turning.getDestination()
                    except Exception:
                        continue
                    if origin is None or destination is None:
                        continue
                    upstream.setdefault(destination.getId(), set()).add(origin.getId())
                    origins.add(origin.getId())
            return upstream, origins


        def trace_to_boundary(from_id, upstream):
            current = from_id
            visited = set()
            while current not in visited:
                visited.add(current)
                ups = upstream.get(current)
                if not ups:
                    return current
                if len(ups) == 1:
                    current = next(iter(ups))
                else:
                    return None
            return None


        def objects_of_type(model, type_name):
            gktype = model.getType(type_name)
            if gktype is None:
                return []
            objs = model.getCatalog().getObjectsByType(gktype)
            if not objs:
                return []
            return [o for o in objs.values() if o is not None]


        # ---- read matchup + GridSmart counts ----
        path_matchup = os.path.join(INPUT_DIR, MATCHUP_FILE)
        traffic_dir = os.path.join(INPUT_DIR, TRAFFIC_SUBDIR)
        print("  :Reading matchup table and GridSmart counts ...")
        matchup = read_matchup_table(path_matchup)
        turn_df, id_ref = generate_turn_demand(matchup, traffic_dir)
        movements = build_movement_counts(turn_df, id_ref, sim_start_time, sim_end_time)
        if movements.empty:
            # print("  :ERROR - no GridSmart demand found in the sim window.")
            raise ValueError("No GridSmart demand found in the sim window.")
            # console.close()
            # return

        # ---- field counts per approach (veh over the whole window) ----
        field = (movements.groupby(["JunctionID", "FromID"], as_index=False)["Count"].sum())
        field_approaches = [{"junction": int(float(r["JunctionID"])),
                             "section": int(float(r["FromID"])),
                             "count": int(r["Count"])}
                            for _, r in field.iterrows()]

        # ---- topology: boundary origins with/without imported flow ----
        print("  :Scanning topology ...")
        upstream, origins = build_topology(model)
        boundary_origins = {s for s in origins if not upstream.get(s)}
        data_boundary = set()
        for _, grp in movements.groupby("IntervalStart"):
            for from_id in grp["FromID"].unique():
                b = trace_to_boundary(int(float(from_id)), upstream)
                if b is not None:
                    data_boundary.add(b)
        calib_inflow_sections = sorted(boundary_origins - data_boundary)

        # ---- turnings of the calibration junctions, grouped by approach ----
        calib_col = matchup["Need calibration?"].astype(str).str.strip().str.upper()
        calib_junctions = sorted(set(int(float(j)) for j in
                                     matchup.loc[calib_col == "Y", "JunctionID_Aimsun"].dropna().unique()))
        catalog = model.getCatalog()
        calib_turns = []
        for jid in calib_junctions:
            node = catalog.find(int(jid))
            if node is None or not node.isA("GKNode"):
                print("  :junction %s not found - skipped." % jid)
                continue
            pairs = set()
            for turning in _node_turnings(node):
                try:
                    origin = turning.getOrigin()
                    destination = turning.getDestination()
                except Exception:
                    continue
                if origin is None or destination is None:
                    continue
                pairs.add((origin.getId(), destination.getId()))
            for f, t in sorted(pairs):
                calib_turns.append({"junction": int(jid), "from": int(f), "to": int(t)})

        # ---- replication id ----
        replication_id = None
        for rep in objects_of_type(model, "GKReplication"):
            if str(rep.getName()) == REPLICATION_NAME:
                replication_id = int(rep.getId())
                break
        if replication_id is None:
            print(f"  :WARNING - replication '{REPLICATION_NAME}' not found; set it manually.")

        state_names = [str(s.getName()) for s in objects_of_type(model, "GKTrafficState")]

        info = {
            "replication_id": replication_id,
            "sim_start_time": sim_start_time,
            "sim_end_time": sim_end_time,
            "interval_seconds": INTERVAL_SECONDS,
            "traffic_states": state_names,
            "field_approaches": field_approaches,
            "calib_inflow_sections": [int(s) for s in calib_inflow_sections],
            "calib_junctions": calib_junctions,
            "calib_turns": calib_turns,
        }
        out_path = os.path.join(INPUT_DIR, OUTPUT_JSON)
        with open(out_path, "w") as fh:
            json.dump(info, fh, indent=2)

        print(f"  :replication id: {replication_id}")
        print("  :%d traffic state(s): %s" % (len(state_names), state_names))
        print("  :%d field approach(es): %s"
              % (len(field_approaches), [a["section"] for a in field_approaches]))
        print("  :%d inflow section(s) to calibrate: %s"
              % (len(calib_inflow_sections), [int(s) for s in calib_inflow_sections]))
        print("  :%d calibration junction(s), %d turning(s) to calibrate."
              % (len(calib_junctions), len(calib_turns)))
        print("  :Wrote %s" % out_path)


        console.save( argv[1])
        console.close()
    else:
        console.getLog().addError( "Cannot load the network" )
        print ("cannot load network")


if __name__ == "__main__":
    sys.exit(main(sys.argv))
