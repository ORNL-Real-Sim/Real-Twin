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
        import io

        input_config = json.loads(argv[-1])
        _SITEPACKAGES = input_config["AIMSUN"]["site_packages"]

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

        MATCHUP_FILE = "MatchupTable.xlsx"
        CONTROL_SUBDIR = "Control"

        # Delete detectors previously created by this script before creating new ones.
        DELETE_EXISTING = True
        DETECTOR_PREFIX = "DET_"

        # Default detector lengths when Synchro gives none.
        DEFAULT_LEN_THROUGH_FT = 50.0    # through and right movements
        DEFAULT_LEN_LEFT_FT = 20.0       # left and U-turn movements

        # Synchro lengths below this are treated as not given.
        MIN_SYNCHRO_LEN_FT = 10.0

        # Place detectors at the stop bar, ignoring Synchro's DetectPos1 setback.
        FORCE_STOP_BAR_DETECTORS = True

        FT_TO_M = 0.3048

        # Detector length and clearance limits (m).
        LANE_LENGTH_TOLERANCE_M = 3.0
        MIN_DETECTOR_LEN_M = 1.0
        UPSTREAM_CLEARANCE_M = 0.5

        # Print the resolved API surface at startup.
        PROBE_API = True



        def process_signal_data(path_signal: str) -> dict:
            """Split a Synchro UTDF csv into its [Lanes]/[Timeplans]/[Phases] tables."""
            with open(path_signal, "r", encoding="utf-8", errors="replace") as fh:
                signal_data = fh.readlines()

            signal_dict = {}
            current_table = None
            current_rows = []
            skip_next = 0

            for line in signal_data:
                line = line.strip()
                if skip_next == 1:
                    skip_next = 0
                    continue
                if line.startswith("["):
                    skip_next = 1
                    if current_table is not None and current_rows:
                        signal_dict[current_table] = pd.read_csv(
                            io.StringIO("\n".join(current_rows)), dtype=str)
                        current_rows = []
                    current_table = line[1:-1].split(",")[0].rstrip("]")
                else:
                    current_rows.append(line)

            if current_table is not None and current_rows:
                signal_dict[current_table] = pd.read_csv(
                    io.StringIO("\n".join(current_rows)), dtype=str)
            return signal_dict


        def read_matchup_table(path_matchup_table: str) -> pd.DataFrame:
            """Load MatchupTable.xlsx, normalising either id-column naming family."""
            matchup = pd.read_excel(path_matchup_table, skiprows=1, dtype=str)

            renames = {}
            for base in ("JunctionID", "FromRoadID", "ToRoadID"):
                if base + "_Aimsun" not in matchup.columns and base + "_OpenDrive" in matchup.columns:
                    renames[base + "_OpenDrive"] = base + "_Aimsun"
            if renames:
                matchup = matchup.rename(columns=renames)

            merged_columns = ["JunctionID_Aimsun", "IntersectionName_GridSmart",
                              "File_Synchro", "IntersectionID_Synchro", "Need calibration?"]
            for col in merged_columns:
                if col not in matchup.columns:
                    matchup[col] = pd.NA
            matchup[merged_columns] = matchup[merged_columns].ffill()
            return matchup


        def build_junction_movements(matchup: pd.DataFrame) -> dict:
            """Return {node_id: {"synchro": INTID, "movements": [(code, from, to), ...]}}."""
            out = {}
            sub = matchup.dropna(subset=["Turn_Synchro"])
            for node_id, grp in sub.groupby("JunctionID_Aimsun", sort=False):
                intid = grp["IntersectionID_Synchro"].dropna()
                if intid.empty:
                    continue
                movements = []
                for _, row in grp.iterrows():
                    code = str(row["Turn_Synchro"]).strip()
                    frm = str(row["FromRoadID_Aimsun"]).strip()
                    to = str(row["ToRoadID_Aimsun"]).strip()
                    if not code or not frm or not to or frm == "nan" or to == "nan":
                        continue
                    movements.append((code, int(float(frm)), int(float(to))))
                if movements:
                    out[int(float(node_id))] = {"synchro": str(intid.iloc[0]).strip(),
                                                "movements": movements}
            return out


        def lanes_value(signal_dict: dict, intid: str, record: str, movement: str):
            """Read one cell of the [Lanes] table, as float, or None."""
            lanes = signal_dict.get("Lanes")
            if lanes is None or movement not in lanes.columns:
                return None
            hit = lanes[(lanes["INTID"] == intid) & (lanes["RECORDNAME"] == record)]
            if hit.empty:
                return None
            val = pd.to_numeric(hit[movement].iloc[0], errors="coerce")
            return None if pd.isna(val) else float(val)


        def lanes_available_length(section, lane_from: int, lane_to: int,
                                   section_length: float = None):
            """Shortest usable length among the lanes a detector covers, or None."""
            lengths = []
            for index in range(int(lane_from), int(lane_to) + 1):
                try:
                    lengths.append(float(section.getLaneLength2D(index)))
                except Exception:
                    continue
            if not lengths:
                return None
            shortest = min(lengths)
            if section_length is not None and (section_length - shortest) <= LANE_LENGTH_TOLERANCE_M:
                return None
            return shortest


        def detector_geometry(signal_dict: dict, intid: str, code: str,
                              section_length: float, lane_limit: float = None):
            """Return ``(position, length, source)`` in metres for one movement."""
            size_ft = lanes_value(signal_dict, intid, "DetectSize1", code)
            pos_ft = lanes_value(signal_dict, intid, "DetectPos1", code)

            if size_ft and size_ft >= MIN_SYNCHRO_LEN_FT:
                source = "Synchro %gft" % size_ft
            else:
                was = size_ft
                is_left = code.endswith("L") or code.endswith("U")
                size_ft = DEFAULT_LEN_LEFT_FT if is_left else DEFAULT_LEN_THROUGH_FT
                source = ("default %gft" % size_ft if not was
                          else "default %gft (Synchro %gft too short)" % (size_ft, was))

            length = size_ft * FT_TO_M
            setback = 0.0 if FORCE_STOP_BAR_DETECTORS else (pos_ft or 0.0) * FT_TO_M
            if FORCE_STOP_BAR_DETECTORS and pos_ft:
                source += ", stop bar (Synchro setback %gft ignored)" % pos_ft

            available = section_length
            if lane_limit is not None and lane_limit < available:
                available = lane_limit
                limited_by_lane = True
            else:
                limited_by_lane = False

            usable = available - setback - UPSTREAM_CLEARANCE_M
            if usable < MIN_DETECTOR_LEN_M:
                setback = 0.0
                usable = available - UPSTREAM_CLEARANCE_M
            if length > usable:
                length = max(MIN_DETECTOR_LEN_M, usable)
                source += " -> trimmed to %s" % ("lane" if limited_by_lane else "section")

            position = section_length - setback - length
            if position < 0.0:
                position = 0.0
            return position, length, source




        def find_node(model, node_id):
            obj = model.getCatalog().find(int(node_id))
            if obj is None or not obj.isA("GKNode"):
                return None
            return obj


        def build_turning_lookup(node) -> dict:
            lookup = {}
            for turning in (node.getTurnings() or []):
                origin = turning.getOrigin()
                dest = turning.getDestination()
                if origin is None or dest is None:
                    continue
                lookup[(origin.getId(), dest.getId())] = turning
            return lookup


        def iter_objects(catalog, gktype):
            try:
                for sub in catalog.getUsedSubTypesFromType(gktype):
                    if sub:
                        for obj in sub.values():
                            if obj is not None:
                                yield obj
            except Exception:
                d = catalog.getObjectsByType(gktype)
                if d:
                    for obj in d.values():
                        if obj is not None:
                            yield obj


        def delete_existing_detectors(model) -> int:
            """Delete every detector this script created (name starts with the prefix)."""
            catalog = model.getCatalog()
            commander = model.getCommander()
            victims = [d for d in iter_objects(catalog, model.getType("GKDetector"))
                       if str(d.getName()).startswith(DETECTOR_PREFIX)]
            for det in victims:
                try:
                    commander.addCommand(det.getDelCmd())
                except Exception as exc:
                    print("  :could not delete detector '%s': %s" % (det.getName(), exc))
            return len(victims)


        def turning_lanes(turning):
            """Lane range the turning uses on its ORIGIN section, as ``(from, to)``."""
            try:
                return int(turning.getOriginFromLane()), int(turning.getOriginToLane())
            except Exception:
                return None


        def create_detector(model, section, name, lane_from, lane_to, position, length):
            """Create one GKDetector on a section."""
            det = GKSystem.getSystem().newObject("GKDetector", model)
            det.setName(name)
            det.setLanes(lane_from, lane_to)
            det.setLength(length)
            det.setPosition(position)
            section.addTopObject(det)
            model.getGeoModel().add(section.getLayer(), det)
            return det


        def build_detectors_for_node(model, node, info, signal_dict):
            """Create the detectors for one node, returning {movement_code: GKDetector}."""
            intid = info["synchro"]
            turnings = build_turning_lookup(node)

            groups = {}
            for code, from_id, to_id in info["movements"]:
                turning = turnings.get((from_id, to_id))
                if turning is None:
                    print("    :no turning for %s (%s->%s)" % (code, from_id, to_id))
                    continue
                section = turning.getOrigin()
                lanes = turning_lanes(turning)
                if section is None or lanes is None:
                    print("    :no origin lanes for %s" % code)
                    continue
                groups.setdefault((section.getId(), lanes[0], lanes[1]),
                                  {"section": section, "lanes": lanes, "codes": []}
                                  )["codes"].append(code)

            detectors = {}
            for key in sorted(groups):
                grp = groups[key]
                section = grp["section"]
                lane_from, lane_to = grp["lanes"]
                codes = grp["codes"]

                try:
                    section_length = float(section.length2D())
                except Exception:
                    print("    :cannot read length of section %s" % section.getId())
                    continue

                lane_limit = lanes_available_length(section, lane_from, lane_to, section_length)

                best = None
                for code in codes:
                    pos, length, source = detector_geometry(
                        signal_dict, intid, code, section_length, lane_limit)
                    if best is None or length > best[1]:
                        best = (pos, length, source, code)
                position, length, source, driver = best

                name = "%s%d_%s" % (DETECTOR_PREFIX, node.getId(), "_".join(sorted(codes)))
                try:
                    det = create_detector(model, section, name,
                                          lane_from, lane_to, position, length)
                except Exception as exc:
                    print("    :FAILED to create %s: %s" % (name, exc))
                    continue

                for code in codes:
                    detectors[code] = det
                print("    :%-24s sec %-5s lanes %d-%d  pos %6.1f len %5.1f m  "
                      "(%s via %s; section %.1f m, shortest lane %s)"
                      % (name, section.getId(), lane_from, lane_to, position, length,
                         source, driver, section_length,
                         "%.1f m" % lane_limit if lane_limit is not None else "n/a"))
            return detectors


        def probe_api():
            """Print whether the methods this script relies on are actually bound."""
            checks = [
                ("GKDetector", ["setLanes", "setLength", "setPosition", "setPositionFromEnd"]),
                ("GKTurning", ["getOriginFromLane", "getOriginToLane", "getOrigin"]),
                ("GKSection", ["length2D", "addTopObject", "getLayer"]),
            ]
            for cls_name, methods in checks:
                cls = globals().get(cls_name)
                if cls is None:
                    print("  :%s not in scope" % cls_name)
                    continue
                have = [m for m in methods if hasattr(cls, m)]
                missing = [m for m in methods if not hasattr(cls, m)]
                print("  :%-12s ok=%s%s" % (cls_name, have,
                                            ("  MISSING=%s" % missing) if missing else ""))




        path_matchup = os.path.join(INPUT_DIR, MATCHUP_FILE)
        control_dir = os.path.join(INPUT_DIR, CONTROL_SUBDIR)
        for required in (path_matchup, control_dir):
            if not os.path.exists(required):
                print("  :ERROR - path not found: %s" % required)
                return

        if PROBE_API:
            probe_api()

        matchup = read_matchup_table(path_matchup)
        junctions = build_junction_movements(matchup)
        if not junctions:
            print("  :No signalised junctions in the matchup table.")
            return

        synchro_file = matchup["File_Synchro"].dropna()
        path_signal = os.path.join(control_dir, str(synchro_file.iloc[0]).strip())
        if not os.path.exists(path_signal):
            print("  :ERROR - Synchro file not found: %s" % path_signal)
            return
        signal_dict = process_signal_data(path_signal)

        if DELETE_EXISTING:
            n = delete_existing_detectors(model)
            print("  :Deleted %d previously created detector(s)." % n)

        total = 0
        for node_id in sorted(junctions):
            info = junctions[node_id]
            node = find_node(model, node_id)
            if node is None:
                print("  :node %s not found (or not a GKNode) - skipped." % node_id)
                continue
            print("  :node %s (Synchro %s)" % (node_id, info["synchro"]))
            detectors = build_detectors_for_node(model, node, info, signal_dict)
            total += len(set(id(d) for d in detectors.values()))

        try:
            GKGUISystem.getGUISystem().getActiveGui().invalidateViews()
        except Exception:
            pass
        model.getCommander().addCommand(None)
        print("  :Done - %d detector(s) created across %d junction(s)."
              % (total, len(junctions)))

        console.save( argv[1])
        console.close()
    else:
        console.getLog().addError( "Cannot load the network" )
        print ("cannot load network")


if __name__ == "__main__":
    sys.exit(main(sys.argv))
