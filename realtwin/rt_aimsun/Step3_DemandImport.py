import sys
import json
from PyANGBasic import *
from PyANGKernel import *
from PyANGConsole import *

def main( argv ):
    # Start a Console
    console = ANGConsole()
    # Load a network
    if console.open( argv[1] ):
        model = console.getModel()

        import sys
        import os

        input_config = json.loads(argv[-1])
        _SITEPACKAGES = input_config["AIMSUN"]["site_packages"]

        # External site-packages providing pandas, xlrd and openpyxl.
        # _SITEPACKAGES = r"C:\Users\ggx\AppData\Local\Programs\Python\Python37\Lib\site-packages"
        # _SITEPACKAGES = r"C:\Users\xh8\AppData\Local\Programs\Python\Python310\Lib\site-packages"

        if _SITEPACKAGES not in sys.path:
            sys.path.append(_SITEPACKAGES)

        import pandas as pd


        def _default_output_dir():
            try:
                doc = str(model.getDocumentFileName())
            except Exception:
                doc = ""
            folder = os.path.dirname(doc) if doc else ""
            if not folder or not os.path.isdir(folder):
                folder = os.path.expanduser("~")
            return folder

        INPUT_DIR = _default_output_dir()
        MATCHUP_FILE = "MatchupTable.xlsx"   # matchup table name
        TRAFFIC_SUBDIR = "Traffic"           # GridSmart files sub-folder

        # Simulation window (seconds from midnight).
        sim_start_time = 28800   # 08:00 AM
        sim_end_time = 32400     # 09:00 AM

        # Time-bin size for each traffic state.
        INTERVAL_SECONDS = 15 * 60

        # Optional prefix for state names.
        STATE_NAME_PREFIX = ""

        # Whether to bundle the states into a GKTrafficDemand.
        CREATE_TRAFFIC_DEMAND = True

        # GKVehicle id override; None resolves 'Car' by name.
        VEHICLE_TYPE_ID = None

        # GridSmart movement columns.
        TURN_VALUES = ["NBR", "NBT", "NBL", "NBU", "EBR", "EBT", "EBL", "EBU",
                       "SBR", "SBT", "SBL", "SBU", "WBR", "WBT", "WBL", "WBU"]


        def time_to_seconds(time_str: str) -> int:
            """Convert a 'HH:MM' string to seconds from midnight."""
            hour, minute = [int(x) for x in str(time_str).split(":")]
            return hour * 3600 + minute * 60


        def read_matchup_table(path_matchup_table: str) -> pd.DataFrame:
            """Load MatchupTable.xlsx and forward-fill its merged-cell columns."""
            matchup = pd.read_excel(path_matchup_table, skiprows=1, dtype=str)

            legacy = {"JunctionID_OpenDrive": "JunctionID_Aimsun",
                      "FromRoadID_OpenDrive": "FromRoadID_Aimsun",
                      "ToRoadID_OpenDrive": "ToRoadID_Aimsun"}
            rename = {old: new for old, new in legacy.items()
                      if old in matchup.columns and new not in matchup.columns}
            if rename:
                matchup = matchup.rename(columns=rename)

            required = ["JunctionID_Aimsun", "FromRoadID_Aimsun", "ToRoadID_Aimsun",
                        "File_GridSmart", "Turn_GridSmart"]
            absent = [c for c in required if c not in matchup.columns]
            if absent:
                raise ValueError(
                    "Matchup table %s is missing the column(s): %s\nFound:\n  %s"
                    % (path_matchup_table, ", ".join(absent),
                       ", ".join(map(str, matchup.columns))))

            merged_columns = ["JunctionID_Aimsun", "IntersectionName_GridSmart",
                              "File_Synchro", "Need calibration?"]
            for col in merged_columns:
                if col not in matchup.columns:
                    matchup[col] = pd.NA
            matchup[merged_columns] = matchup[merged_columns].ffill()
            return matchup


        def generate_turn_demand(matchup: pd.DataFrame, traffic_dir: str):
            """Read the GridSmart count files and build the raw turn-count table."""
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
                    print("  :GridSmart file not found, skipping: %s" % gs_file_path)
                    continue

                df = pd.read_excel(gs_file_path, header=None)

                time_mask = df[0].astype(str).str.match(r"^\d{1,2}:\d{2}$", na=False)
                time_row_index = df[time_mask].index.min()
                if pd.isna(time_row_index):
                    print("  :No time rows found in %s, skipping." % gs_file_path)
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
                id_ref = id_ref[["IntersectionName", "Turn", "FromID", "ToID"]]
                id_ref = id_ref[(id_ref["FromID"].astype(str) != "")
                                & (id_ref["ToID"].astype(str) != "")]
            return turn_df, id_ref


        def build_movement_counts(turn_df: pd.DataFrame, id_ref: pd.DataFrame,
                                  sim_begin: int, sim_end: int) -> pd.DataFrame:
            """Reduce the wide count table to per-movement totals over the sim window."""
            if turn_df.empty or id_ref.empty:
                return pd.DataFrame(columns=["IntervalStart", "FromID", "ToID", "Count"])

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

            movements = (merged.groupby(["IntervalStart", "FromID", "ToID"],
                                        as_index=False)["Count"]
                         .sum())
            movements = movements[movements["Count"] > 0]
            movements = movements.sort_values("IntervalStart").reset_index(drop=True)
            return movements


        def seconds_to_label(seconds: int) -> str:
            """Format seconds-from-midnight as a state label ('8:15')."""
            return "%d:%02d" % (int(seconds) // 3600, (int(seconds) % 3600) // 60)


        def seconds_to_hhmm(seconds: int) -> str:
            """Zero-padded HH:MM, e.g. 28800 -> '08:00'."""
            return "%02d:%02d" % (int(seconds) // 3600, (int(seconds) % 3600) // 60)


        def seconds_to_hhmmss(seconds: int) -> str:
            """Zero-padded HH:MM:SS, e.g. 900 -> '00:15:00'."""
            s = int(seconds)
            return "%02d:%02d:%02d" % (s // 3600, (s % 3600) // 60, s % 60)




        def _iter_objects(catalog, gktype):
            """Yield every catalog object of the given GKType (handling subtypes)."""
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
            """Return the GKTurning objects that belong to ``node`` as a Python list."""
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


        def build_upstream_map(model):
            """Build ``{section_id: set(upstream_section_ids)}`` from all turnings."""
            catalog = model.getCatalog()
            node_type = model.getType("GKNode")
            upstream = {}
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
            return upstream


        def build_approach_turnings(model):
            """Build ``{origin_section_id: [destination_section_id, ...]}`` from all turnings."""
            catalog = model.getCatalog()
            node_type = model.getType("GKNode")
            approaches = {}
            for node in _iter_objects(catalog, node_type):
                for turning in _node_turnings(node):
                    try:
                        origin = turning.getOrigin()
                        destination = turning.getDestination()
                    except Exception:
                        continue
                    if origin is None or destination is None:
                        continue
                    approaches.setdefault(origin.getId(), []).append(destination.getId())
            return approaches


        def even_integer_split(n: int) -> list:
            """Split 100 into ``n`` integer parts that sum to exactly 100."""
            if n <= 0:
                return []
            base = 100 // n
            remainder = 100 - base * n
            return [base + (1 if i < remainder else 0) for i in range(n)]


        def trace_to_boundary(from_id: int, upstream: dict):
            """Follow the unique upstream chain from ``from_id`` to a boundary origin."""
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


        def get_state_folder(model):
            """Return (creating if needed) the folder that holds traffic states."""
            folder_name = "GKModel::trafficStates"
            folder = model.getCreateRootFolder().findFolder(folder_name)
            if folder is None:
                folder = GKSystem.getSystem().createFolder(model.getCreateRootFolder(), folder_name)
            return folder


        def create_state(model, name: str):
            """Create a new GKTrafficState with the given name."""
            state = GKSystem.getSystem().newObject("GKTrafficState", model)
            state.setName(name)
            get_state_folder(model).append(state)
            return state


        def set_state_time(state, start_seconds: int, duration_seconds: int) -> bool:
            """Stamp a state's start time (QTime) and duration (GKTimeDuration)."""
            start_str = seconds_to_hhmmss(start_seconds)
            dur_str = seconds_to_hhmmss(duration_seconds)

            ok = True
            try:
                state.setFrom(QTime.fromString(start_str, Qt.ISODate))
            except Exception:
                try:
                    state.setFrom(int(start_seconds))
                except Exception:
                    ok = False
            try:
                state.setDuration(GKTimeDuration.fromString(dur_str))
            except Exception:
                try:
                    state.setDuration(int(duration_seconds))
                except Exception:
                    ok = False
            return ok


        def create_traffic_demand(model, name: str, states: list, duration_seconds: int):
            """Bundle the per-interval states into a GKTrafficDemand."""
            demand = GKSystem.getSystem().newObject("GKTrafficDemand", model)
            demand.setName(name)

            folder_name = "GKModel::trafficDemand"
            folder = model.getCreateRootFolder().findFolder(folder_name)
            if folder is None:
                folder = GKSystem.getSystem().createFolder(model.getCreateRootFolder(), folder_name)
            folder.append(demand)

            for start_seconds, state in states:
                schedule = GKScheduleDemandItem()
                schedule.setTrafficDemandItem(state)
                schedule.setFrom(int(start_seconds))
                schedule.setDuration(int(duration_seconds))
                demand.addToSchedule(schedule)
            return demand


        def find_section(model, entry):
            """Find a GKSection by its Aimsun id; return None if it is not a section."""
            try:
                section = model.getCatalog().find(int(float(entry)))
            except (TypeError, ValueError):
                return None
            if section is None or not section.isA("GKSection"):
                return None
            return section


        def resolve_vehicle(model):
            """Return the 'Car' GKVehicle so all demand is assigned to it."""
            catalog = model.getCatalog()

            if VEHICLE_TYPE_ID is not None:
                veh = catalog.find(int(VEHICLE_TYPE_ID))
                if veh is not None:
                    return veh
                print("  :VEHICLE_TYPE_ID %s not found; falling back to 'Car'." % VEHICLE_TYPE_ID)

            car_id = None
            try:
                veh_folder = catalog.findByName("Vehicles")
                veh_subfolder = veh_folder.getContents()
                for veh_id in list(veh_subfolder.keys()):
                    car_obj = catalog.find(veh_id)
                    if car_obj is not None and car_obj.getName() == "Car":
                        car_id = veh_id
                        break
            except Exception as e:
                print("  :Vehicles-folder lookup failed (%s); trying findByName('Car')." % e)

            if car_id is not None:
                veh = catalog.find(car_id)
                if veh is not None:
                    return veh

            veh = catalog.findByName("Car")
            if veh is None:
                print("  :Vehicle type 'Car' not found; using default vehicle.")
            return veh



        def build_state_for_interval(model, interval_movements, upstream, vehicle,
                                     start_seconds: int, approach_turnings: dict) -> tuple:
            """Create and populate one GKTrafficState for a single time bin."""
            label = "%s%s" % (STATE_NAME_PREFIX, seconds_to_label(start_seconds))
            state = create_state(model, label)
            set_state_time(state, start_seconds, INTERVAL_SECONDS)

            if vehicle is not None:
                try:
                    state.setVehicle(vehicle)
                except Exception as e:
                    print("  :Could not set vehicle on state '%s' (%s)." % (label, e))

            from_totals = interval_movements.groupby("FromID")["Count"].sum().to_dict()

            turns_set = 0
            data_origins = set()
            for _, row in interval_movements.iterrows():
                total = from_totals.get(row["FromID"], 0)
                if not total:
                    continue
                from_sec = find_section(model, row["FromID"])
                to_sec = find_section(model, row["ToID"])
                if from_sec is None or to_sec is None:
                    continue
                percentage = float(row["Count"]) / float(total) * 100.0
                state.setTurningPercentage(from_sec, to_sec, None, percentage)
                turns_set += 1
                try:
                    data_origins.add(int(float(row["FromID"])))
                except (TypeError, ValueError):
                    pass

            even_approaches = 0
            for origin_id, dest_ids in approach_turnings.items():
                if origin_id in data_origins:
                    continue
                from_sec = find_section(model, origin_id)
                if from_sec is None:
                    continue
                to_secs, seen = [], set()
                for dest_id in dest_ids:
                    if dest_id in seen:
                        continue
                    seen.add(dest_id)
                    to_sec = find_section(model, dest_id)
                    if to_sec is not None:
                        to_secs.append(to_sec)
                if not to_secs:
                    continue
                for to_sec, share in zip(to_secs, even_integer_split(len(to_secs))):
                    state.setTurningPercentage(from_sec, to_sec, None, float(share))
                even_approaches += 1

            entrance_flow = {}
            internal_approaches = 0
            for from_id, count in from_totals.items():
                try:
                    key = int(float(from_id))
                except (TypeError, ValueError):
                    continue
                boundary_id = trace_to_boundary(key, upstream)
                if boundary_id is None:
                    internal_approaches += 1
                    continue
                flow_vph = float(count) / float(INTERVAL_SECONDS) * 3600.0
                entrance_flow[boundary_id] = entrance_flow.get(boundary_id, 0.0) + flow_vph

            flows_set = 0
            for boundary_id, flow_vph in entrance_flow.items():
                section = find_section(model, boundary_id)
                if section is None:
                    continue
                state.setEntranceFlow(section, None, flow_vph)
                flows_set += 1

            return state, turns_set, flows_set, internal_approaches, even_approaches


        path_matchup = os.path.join(INPUT_DIR, MATCHUP_FILE)
        traffic_dir = os.path.join(INPUT_DIR, TRAFFIC_SUBDIR)

        for required in (path_matchup, traffic_dir):
            if not os.path.exists(required):
                print("  :ERROR - path not found: %s" % required)
                print("  :Set INPUT_DIR at the top of the script and re-run.")
                return

        if sim_end_time <= sim_start_time:
            print("  :ERROR - sim_end_time must be greater than sim_start_time.")
            return

        print("  :Reading matchup table and GridSmart counts ...")
        matchup = read_matchup_table(path_matchup)
        turn_df, id_ref = generate_turn_demand(matchup, traffic_dir)
        movements = build_movement_counts(turn_df, id_ref, sim_start_time, sim_end_time)

        if movements.empty:
            print("  :No demand found in [%s - %s]. Nothing to import."
                  % (seconds_to_label(sim_start_time), seconds_to_label(sim_end_time)))
            return

        intervals = sorted(movements["IntervalStart"].unique())
        print("  :%s - %s -> %d interval(s) of %d min: %s"
              % (seconds_to_label(sim_start_time), seconds_to_label(sim_end_time),
                 len(intervals), INTERVAL_SECONDS // 60,
                 ", ".join(seconds_to_label(t) for t in intervals)))

        print("  :Scanning Aimsun topology for boundary origin sections ...")
        upstream = build_upstream_map(model)
        approach_turnings = build_approach_turnings(model)
        vehicle = resolve_vehicle(model)

        created = []
        for start_seconds in intervals:
            interval_movements = movements[movements["IntervalStart"] == start_seconds]
            state, turns_set, flows_set, internal, even = build_state_for_interval(
                model, interval_movements, upstream, vehicle, int(start_seconds),
                approach_turnings)
            created.append((int(start_seconds), state))
            print("  :State '%s' - %d turning percentages (+%d approach(es) even-split), "
                  "%d entrance flows (veh/h); %d approaches internally fed."
                  % (state.getName(), turns_set, even, flows_set, internal))

        if CREATE_TRAFFIC_DEMAND and created:
            demand_start = intervals[0]
            demand_end = intervals[-1] + INTERVAL_SECONDS
            demand_name = "Traffic Demand %s to %s" % (
                seconds_to_hhmm(demand_start), seconds_to_hhmm(demand_end))
            try:
                demand = create_traffic_demand(model, demand_name, created, INTERVAL_SECONDS)
                print("  :Traffic demand '%s' scheduled with %d states."
                      % (demand.getName(), len(created)))
            except Exception as e:
                print("  :Could not build the traffic demand (%s). The %d states were "
                      "still created - add them to a demand manually." % (e, len(created)))

        model.getCommander().addCommand(None)
        print("  :Demand import complete - %d traffic state(s) created." % len(created))

        console.save( argv[1])
        console.close()
    else:
        console.getLog().addError( "Cannot load the network" )
        print ("cannot load network")


if __name__ == "__main__":
    sys.exit(main(sys.argv))
