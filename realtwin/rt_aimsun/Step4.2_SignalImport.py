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



        # Folder holding MatchupTable.xlsx and the Control sub-folder.
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

        # Name given to the created GKControlPlan.
        CONTROL_PLAN_NAME = "RealTwin Synchro Actuated"

        # Junction control type: "auto", "actuated", or "fixed".
        CONTROL_JUNCTION_TYPE = "auto"

        # True: rebuild signal groups; False: reuse existing ones by name.
        REPLACE_EXISTING_SIGNAL_GROUPS = True

        # Apply Synchro Recall codes to the phases.
        APPLY_RECALL = True

        # Floor for Passage Time when Synchro VehExt is 0.
        DEFAULT_PASSAGE_TIME_S = 1.0

        # Indication for permitted movements (GKControlPhaseSignal value).
        PERMITTED_SIGNAL_INDICATION = "eFlashingYellow"

        # Mark permitted movements' turnings as give-way.
        SET_GIVEWAY_ON_PERMITTED = True

        # NEMA multi-ring entry mode: "dual" or "single".
        PHASE_ENTRY_MODE = "dual"

        # Fill an empty ring-barrier by mirroring the other ring's phases.
        MIRROR_EMPTY_RING_BARRIER = True

        # Mark one default phase per (ring, barrier) under dual entry.
        MARK_DEFAULT_PHASES = True

        # Detectors
        CREATE_DETECTORS = True
        LINK_DETECTORS_TO_PHASES = True

        DETECTOR_PREFIX = "DET_"
        DEFAULT_LEN_THROUGH_FT = 50.0    # through and right movements
        DEFAULT_LEN_LEFT_FT = 20.0       # left and U-turn movements
        MIN_SYNCHRO_LEN_FT = 10.0        # ignore Synchro detector lengths below this
        FORCE_STOP_BAR_DETECTORS = True  # place detectors at the stop bar
        FT_TO_M = 0.3048
        LANE_LENGTH_TOLERANCE_M = 3.0    # lane-vs-section difference before trimming
        MIN_DETECTOR_LEN_M = 1.0
        UPSTREAM_CLEARANCE_M = 0.5




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


        def _timeplan_value(signal_dict: dict, intid: str, record: str, default=None):
            """Read one scalar from the ``[Timeplans]`` table."""
            tp = signal_dict.get("Timeplans")
            if tp is None:
                return default
            hit = tp[(tp["INTID"] == intid) & (tp["RECORDNAME"] == record)]
            if hit.empty or "DATA" not in hit.columns:
                return default
            val = hit["DATA"].iloc[0]
            return default if pd.isna(val) else val


        def build_phase_table(signal_dict: dict, intid: str) -> pd.DataFrame:
            """Decode [Phases] for one intersection into a per-phase table."""
            phases = signal_dict["Phases"]
            phases = phases[phases["INTID"] == intid]
            tab = phases.set_index("RECORDNAME").transpose().reset_index()
            tab = tab.rename(columns={"index": "Phase"})
            tab = tab[tab["Phase"] != "INTID"]
            if "BRP" not in tab.columns:
                return pd.DataFrame()
            tab = tab.dropna(subset=["BRP"])

            numeric = ["MinGreen", "MaxGreen", "Yellow", "AllRed", "VehExt", "MinGap",
                       "TimeBeforeReduce", "TimeToReduce", "Recall"]
            for col in numeric:
                if col not in tab.columns:
                    tab[col] = 0
                tab[col] = pd.to_numeric(tab[col], errors="coerce").fillna(0)

            # 'D1' -> '1'
            tab["Phase"] = tab["Phase"].astype(str).str.replace("D", "", regex=False)

            # Unused phases carry no green at all.
            tab = tab[~((tab["MinGreen"] == 0) & (tab["MaxGreen"] == 0))]
            if tab.empty:
                return tab

            brp = tab["BRP"].astype(str)
            tab["Barrier"] = brp.str[0].astype(int)
            tab["Ring"] = brp.str[1].astype(int)
            tab["Position"] = brp.str[2].astype(int)

            # A phase occupies its whole split: green + yellow + all-red.
            tab["Split"] = tab["MaxGreen"] + tab["Yellow"] + tab["AllRed"]
            return tab.reset_index(drop=True)


        def schedule_phases(phase_tab: pd.DataFrame) -> tuple:
            """Lay the phases out on the cycle."""
            tab = phase_tab.copy()
            tab["From"] = 0.0

            barrier_starts = {}
            clock = 0.0
            for barrier in sorted(tab["Barrier"].unique()):
                barrier_starts[barrier] = clock
                in_barrier = tab[tab["Barrier"] == barrier]
                ring_totals = in_barrier.groupby("Ring")["Split"].sum()
                for ring in sorted(in_barrier["Ring"].unique()):
                    acc = clock
                    rows = in_barrier[in_barrier["Ring"] == ring].sort_values("Position")
                    for idx in rows.index:
                        tab.at[idx, "From"] = acc
                        acc += float(tab.at[idx, "Split"])
                clock += float(ring_totals.max())

            return tab, barrier_starts, clock


        def build_phase_movements(signal_dict: dict, intid: str, movements: list) -> tuple:
            """Map each phase number to its protected / permitted movement codes."""
            lanes = signal_dict["Lanes"]
            lanes = lanes[lanes["INTID"] == intid]
            move_cols = [c for c in lanes.columns if c not in ("RECORDNAME", "INTID")]

            protected, permitted = {}, {}
            groups = ([("Phase%d" % i, protected) for i in range(1, 5)]
                      + [("PermPhase%d" % i, permitted) for i in range(1, 5)])
            for record, target in groups:
                rows = lanes[lanes["RECORDNAME"] == record]
                if rows.empty:
                    continue
                row = rows.iloc[0]
                for col in move_cols:
                    val = pd.to_numeric(row.get(col), errors="coerce")
                    if pd.isna(val) or int(val) == 0:
                        continue
                    key = str(int(val))
                    target.setdefault(key, [])
                    if col not in target[key]:
                        target[key].append(col)

            # Movements with no phase of their own inherit one from the same approach.
            inherit_order = {"U": ("L", "T"), "R": ("T", "L")}

            codes = [m[0] for m in movements]
            already = set()
            for bucket in (protected, permitted):
                for mvs in bucket.values():
                    already.update(mvs)

            inherited = {}
            for code in codes:
                suffix = code[-1:]
                if suffix not in inherit_order or code in already:
                    continue
                for alt_suffix in inherit_order[suffix]:
                    alt = code[:-1] + alt_suffix
                    # Copy the inherited movement into every phase and bucket its host occupies.
                    placed = False
                    for bucket in (protected, permitted):
                        for mvs in bucket.values():
                            if alt in mvs and code not in mvs:
                                mvs.append(code)
                                placed = True
                    if placed:
                        inherited[code] = alt
                        break

            if inherited:
                print("    :movements inheriting a phase: %s"
                      % ", ".join("%s<-%s" % (k, v) for k, v in sorted(inherited.items())))
            return protected, permitted




        def find_node(model, node_id):
            """Find a GKNode by id, returning None if the id is not a node."""
            obj = model.getCatalog().find(int(node_id))
            if obj is None or not obj.isA("GKNode"):
                return None
            return obj


        def build_turning_lookup(node) -> dict:
            """``{(origin_section_id, destination_section_id): GKTurning}`` for a node."""
            lookup = {}
            for turning in (node.getTurnings() or []):
                origin = turning.getOrigin()
                dest = turning.getDestination()
                if origin is None or dest is None:
                    continue
                lookup[(origin.getId(), dest.getId())] = turning
            return lookup


        def create_signal_groups(model, node, movements: list) -> dict:
            """Create one GKControlPlanSignal per movement and attach it to the node."""
            existing = list(node.getSignals() or [])
            if existing:
                if not REPLACE_EXISTING_SIGNAL_GROUPS:
                    by_name = {}
                    for sig in existing:
                        by_name[str(sig.getName())] = sig
                    print("    :node %s already has %d signal groups - reusing."
                          % (node.getId(), len(existing)))
                    return by_name

                # Delete the old groups before rebuilding.
                commander = model.getCommander()
                for sig in existing:
                    try:
                        commander.addCommand(sig.getDelCmd())
                    except Exception as exc:
                        print("    :could not delete signal group '%s': %s"
                              % (sig.getName(), exc))
                left = len(node.getSignals() or [])
                print("    :node %s - removed %d existing signal group(s)%s."
                      % (node.getId(), len(existing),
                         "" if left == 0 else " (%d survived!)" % left))

            turnings = build_turning_lookup(node)
            signals = {}
            missing = []
            for code, from_id, to_id in movements:
                turning = turnings.get((from_id, to_id))
                if turning is None:
                    missing.append((code, from_id, to_id))
                    continue
                signal = GKSystem.getSystem().newObject("GKControlPlanSignal", model)
                signal.setName(code)
                try:
                    signal.setNode(node)
                except Exception:
                    pass
                signal.addTurning(turning)
                node.addSignal(signal)
                signals[code] = signal

            if missing:
                print("    :no turning for %d movement(s): %s" % (len(missing), missing))
            return signals


        def get_control_plan_folder(model):
            """Return (creating if needed) the folder that holds control plans."""
            folder_name = "GKModel::controlPlans"
            folder = model.getCreateRootFolder().findFolder(folder_name)
            if folder is None:
                folder = model.getCreateRootFolder().createFolder("Control Plans", folder_name)
            return folder


        def create_control_plan(model, name: str):
            """Create a new GKControlPlan with the given name."""
            plan = GKSystem.getSystem().newObject("GKControlPlan", model)
            plan.setName(name)
            get_control_plan_folder(model).append(plan)
            return plan


        _RECALL_MAP_CACHE = None


        def get_recall_map(enum_type=None):
            """Map Synchro Recall codes to GKControlPhaseRecall enum members."""
            global _RECALL_MAP_CACHE
            if _RECALL_MAP_CACHE is not None:
                return _RECALL_MAP_CACHE
            if enum_type is None:
                enum_type = GKControlPhase

            names = [m for m in dir(enum_type) if m.startswith("e")]
            members = {m: getattr(enum_type, m) for m in names}

            def find(*subs):
                for name, val in members.items():
                    low = name.lower()
                    if any(s in low for s in subs):
                        return val
                return None

            no_recall = find("norecall", "none")
            minimum = find("minimum", "min")
            maximum = find("maximum", "max")
            coordinated = find("coordinated", "coord")

            _RECALL_MAP_CACHE = {
                0: no_recall,   # no recall
                1: minimum,     # minimum recall
                2: minimum,     # pedestrian recall -> minimum
                3: maximum,     # maximum recall
                4: minimum,     # rest in walk -> minimum
                "coord": coordinated,   # coordinated phase
            }
            if coordinated is None or minimum is None:
                print("  :recall enum members found on %s: %s"
                      % (getattr(enum_type, "__name__", "?"), names))
            print("  :recall enum resolved -> none=%s min=%s max=%s coord=%s"
                  % (no_recall, minimum, maximum, coordinated))
            return _RECALL_MAP_CACHE


        def parse_reference_phase(ref_phase) -> set:
            """Synchro Reference Phase -> set of coordinated phase numbers (as str)."""
            try:
                rp = int(float(ref_phase))
            except (TypeError, ValueError):
                return set()
            if rp <= 0:
                return set()
            if rp <= 99:
                return {str(rp)}
            return {str(rp // 100), str(rp % 100)}


        def apply_coordinated_recall(phase) -> bool:
            """Set a phase's recall to Coordinated (overrides its Synchro recall)."""
            try:
                value = get_recall_map().get("coord")
            except Exception:
                return False
            if value is None:
                return False
            try:
                phase.setRecall(value)
                return True
            except Exception:
                return False


        def apply_recall(phase, synchro_code):
            """Set the phase recall from a Synchro Recall code, mapped to Aimsun's enum."""
            try:
                recall_map = get_recall_map()
            except Exception as exc:
                print("  :GKControlPhaseRecall unavailable (%s) - recall left as-is." % exc)
                return
            value = recall_map.get(int(synchro_code))
            if value is None:
                return
            try:
                phase.setRecall(value)
            except Exception:
                pass


        def apply_actuated_params(phase, row):
            """Push the Synchro actuated parameters onto a GKControlPhase."""
            # Passage Time from VehExt, floored when VehExt is 0.
            passage = float(row["VehExt"])
            if passage <= 0.0:
                passage = DEFAULT_PASSAGE_TIME_S

            setters = [
                ("setMinDuration", float(row["MinGreen"])),
                ("setMaxDuration", float(row["MaxGreen"])),
                ("setPassageTime", passage),
                ("setSecondsActuation", passage),
                ("setMinimumGap", float(row["MinGap"])),
                ("setTimeBeforeReduce", float(row["TimeBeforeReduce"])),
                ("setTimeToReduce", float(row["TimeToReduce"])),
                ("setUsingGapReduction", float(row["TimeToReduce"]) > 0),
                # The green phase carries its yellow; the all-red goes to the interphase.
                ("setYellowTimeDuration", float(row["Yellow"])),
                ("setMaximumInitial", 0.0),
                ("setHold", False),
                ("setForceOff", 0.0),
                ("setPermissivePeriodFrom", 0.0),
                ("setPermissivePeriodTo", 0.0),
            ]
            for name, value in setters:
                try:
                    getattr(phase, name)(value)
                except Exception:
                    pass

            if APPLY_RECALL:
                apply_recall(phase, row["Recall"])


        # Detectors

        def lanes_value(signal_dict: dict, intid: str, record: str, movement: str):
            """Read one cell of the [Lanes] table as a float, or None."""
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
            """Return (position, length, source) in metres for one movement."""
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

            # The detector must fit inside the section and every lane it covers.
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


        def turning_lanes(turning):
            """Lane range the turning uses on its origin section, as (from, to)."""
            try:
                return int(turning.getOriginFromLane()), int(turning.getOriginToLane())
            except Exception:
                return None


        def build_detectors_for_node(model, node, info, signal_dict) -> dict:
            """Create (or reuse) the stop-bar detectors for one node."""
            intid = info["synchro"]
            turnings = build_turning_lookup(node)

            existing = {}
            for det in (model.getCatalog().getObjectsByType(model.getType("GKDetector")) or {}).values():
                if det is not None and str(det.getName()).startswith(DETECTOR_PREFIX):
                    existing[str(det.getName())] = det

            groups = {}
            for code, from_id, to_id in info["movements"]:
                turning = turnings.get((from_id, to_id))
                if turning is None:
                    continue
                section = turning.getOrigin()
                lanes = turning_lanes(turning)
                if section is None or lanes is None:
                    continue
                groups.setdefault((section.getId(), lanes[0], lanes[1]),
                                  {"section": section, "lanes": lanes, "codes": []}
                                  )["codes"].append(code)

            detectors = {}
            n_new = 0
            for key in sorted(groups):
                grp = groups[key]
                section, (lane_from, lane_to), codes = grp["section"], grp["lanes"], grp["codes"]
                name = "%s%d_%s" % (DETECTOR_PREFIX, node.getId(), "_".join(sorted(codes)))

                if name in existing:
                    for code in codes:
                        detectors[code] = existing[name]
                    continue

                try:
                    section_length = float(section.length2D())
                except Exception:
                    continue

                # One loop per lane group: take the longest detector any movement asks for.
                lane_limit = lanes_available_length(section, lane_from, lane_to, section_length)
                best = None
                for code in codes:
                    pos, length, _ = detector_geometry(signal_dict, intid, code,
                                                       section_length, lane_limit)
                    if best is None or length > best[1]:
                        best = (pos, length)
                position, length = best

                try:
                    det = GKSystem.getSystem().newObject("GKDetector", model)
                    det.setName(name)
                    det.setLanes(lane_from, lane_to)
                    det.setLength(length)
                    det.setPosition(position)
                    section.addTopObject(det)
                    model.getGeoModel().add(section.getLayer(), det)
                except Exception as exc:
                    print("    :detector %s failed: %s" % (name, exc))
                    continue

                for code in codes:
                    detectors[code] = det
                n_new += 1

            if n_new:
                print("    :created %d detector(s)" % n_new)
            return detectors


        def link_detectors_to_phases(phase_objs: dict, detectors: dict,
                                     signal_dict: dict, intid: str,
                                     mirror_map: dict = None) -> int:
            """Attach each movement's detector to the phase(s) Synchro says it calls."""
            mirror_map = mirror_map or {}
            linked = set()
            n = 0
            for code, det in sorted(detectors.items()):
                targets = []
                for i in (1, 2, 3, 4):
                    val = lanes_value(signal_dict, intid, "DetectPhase%d" % i, code)
                    if val and int(val) > 0:
                        targets.append(str(int(val)))
                for phase_num in targets:
                    phases = []
                    primary = phase_objs.get(phase_num)
                    if primary is not None:
                        phases.append(primary)
                    phases.extend(mirror_map.get(phase_num, []))
                    for phase in phases:
                        key = (id(phase), id(det))
                        if key in linked:
                            continue
                        linked.add(key)
                        try:
                            control_det = phase.createControlDetector(det)
                            if control_det is None:
                                continue
                            for setter, value in (("setPhaseActivation", True),
                                                  ("setPhaseExtension", True)):
                                try:
                                    getattr(control_det, setter)(value)
                                except Exception:
                                    pass
                            phase.addControlDetector(control_det)
                            n += 1
                        except Exception as exc:
                            print("    :link %s -> phase %s failed: %s" % (code, phase_num, exc))
            return n


        def add_movement_signals(phase, phase_num, protected, permitted, signals,
                                 permitted_indication):
            """Attach a Synchro phase's protected/permitted movement signals to a phase."""
            prot_codes = list(protected.get(phase_num, []))
            perm_codes = [c for c in permitted.get(phase_num, []) if c not in prot_codes]
            used, permitted_out, n_perm = set(), set(), 0
            for code in prot_codes:
                signal = signals.get(code)
                if signal is None:
                    continue
                phase.addSignal(signal)
                used.add(code)
            for code in perm_codes:
                signal = signals.get(code)
                if signal is None:
                    continue
                # Right turns get a plain green; only left/U-turns get the permissive indication.
                indication = None if code.endswith("R") else permitted_indication
                if indication is None:
                    phase.addSignal(signal)
                else:
                    try:
                        phase.addSignal(signal, indication)
                    except Exception:
                        phase.addSignal(signal)
                n_perm += 1
                permitted_out.add(code)
                used.add(code)
            return n_perm, permitted_out, used


        def resolve_permitted_indication():
            """Return the GKControlPhaseSignal value for permitted movements, or None."""
            name = (PERMITTED_SIGNAL_INDICATION or "").strip()
            if not name or name == "eNo":
                return None
            value = getattr(GKControlPhaseSignal, name, None)
            if value is None:
                print("  :PERMITTED_SIGNAL_INDICATION '%s' not found on "
                      "GKControlPhaseSignal - permitted movements get a plain green."
                      % name)
            return value


        def build_control_junction(model, plan, node, info, signal_dict, signals):
            """Create and populate the GKControlJunction for one node.

            Returns ``(n_phases, computed_cycle, synchro_cycle, unassigned, n_permitted,
            n_giveway)``.
            """
            permitted_indication = resolve_permitted_indication()
            n_permitted = 0
            permitted_codes = set()
            intid = info["synchro"]
            phase_tab = build_phase_table(signal_dict, intid)
            if phase_tab.empty:
                raise ValueError("no usable phases in Synchro data for INTID %s" % intid)

            phase_tab, barrier_starts, computed_cycle = schedule_phases(phase_tab)
            protected, permitted = build_phase_movements(signal_dict, intid, info["movements"])

            synchro_cycle = _timeplan_value(signal_dict, intid, "Cycle Length")
            synchro_cycle = float(synchro_cycle) if synchro_cycle is not None else computed_cycle
            offset = _timeplan_value(signal_dict, intid, "Offset", 0)

            # Control Type -> junction type (0 => Fixed, 1/2/3 => Actuated; 3 coordinated).
            try:
                control_type = int(float(_timeplan_value(signal_dict, intid, "Control Type", 3)))
            except (TypeError, ValueError):
                control_type = 3
            is_coordinated = (control_type == 3)

            mode = CONTROL_JUNCTION_TYPE.lower()
            if mode == "fixed" or (mode == "auto" and control_type == 0):
                junction_type = GKControlJunction.eFixedControl
                type_name = "fixed"
            else:
                junction_type = GKControlJunction.eActuated
                type_name = "actuated"

            cp_node = plan.createControlJunction(node.getId())
            cp_node.setControlJunctionType(junction_type)

            cp_node.setCycle(int(round(synchro_cycle)))
            try:
                cp_node.setOffset(int(round(float(offset))))
            except Exception:
                cp_node.setOffset(0)

            # Junction-level yellow: the most common phase yellow.
            try:
                cp_node.setYellowTime(float(phase_tab["Yellow"].mode().iloc[0]))
            except Exception:
                pass
            single_entry = (PHASE_ENTRY_MODE.lower() == "single")
            for name, value in (("setRestInRed", False),
                                ("setSingleEntry", single_entry)):
                try:
                    getattr(cp_node, name)(value)
                except Exception:
                    pass

            # Coordination (Control Type 3): Reference Phase(s) become the coordinated phases.
            coord_phases = set()
            if is_coordinated:
                coord_phases = parse_reference_phase(
                    _timeplan_value(signal_dict, intid, "Reference Phase", ""))
                try:
                    ref_to = int(float(_timeplan_value(signal_dict, intid, "Referenced To", 3)))
                except (TypeError, ValueError):
                    ref_to = 3
                if ref_to == 0:
                    match_end = True                 # End of Phase
                elif ref_to == 3:
                    match_end = False                # Beginning of Phase
                else:
                    print("    :Referenced To %d not supported (only 0/TS1, 3/TS2) - "
                          "using TS2 (Beginning of Phase)." % ref_to)
                    match_end = False
                try:
                    cp_node.setMatchesOffsetWithEndOfPhase(match_end)
                except Exception:
                    pass
                print("    :control type 3 (coordinated) - coord phases %s, offset %s, "
                      "%s" % (sorted(coord_phases), offset,
                              "End of Phase (TS1)" if match_end else "Beginning of Phase (TS2)"))
            else:
                try:
                    cp_node.setMatchesOffsetWithEndOfPhase(False)
                except Exception:
                    pass
                print("    :control type %d (%s)" % (control_type, type_name))

            # Under dual entry, pick one default phase per (ring, barrier): the largest MaxGreen.
            default_phases = set()
            if MARK_DEFAULT_PHASES and not single_entry:
                for _, grp in phase_tab.groupby(["Ring", "Barrier"]):
                    idx = grp["MaxGreen"].astype(float).idxmax()
                    default_phases.add(str(grp.loc[idx, "Phase"]))

            # Barriers are start times.
            for barrier in sorted(barrier_starts):
                try:
                    cp_node.addBarrier(int(round(barrier_starts[barrier])))
                except Exception as exc:
                    print("    :addBarrier(%s) failed: %s" % (barrier_starts[barrier], exc))

            used_movements = set()
            phase_objs = {}
            n_interphases = 0
            for _, row in phase_tab.sort_values(["Ring", "Barrier", "Position"]).iterrows():
                start = float(row["From"])
                green = float(row["MaxGreen"]) + float(row["Yellow"])    # green + yellow
                clearance = float(row["AllRed"])                          # all-red only

                # Green phase: green interval + yellow.
                phase = cp_node.createPhase()
                phase_objs[str(row["Phase"])] = phase
                phase.setFrom(start)
                phase.setDuration(green)
                phase.setInterphase(False)
                try:
                    phase.setIdRing(int(row["Ring"]))
                    phase.setIdBarrier(int(row["Barrier"]))
                except Exception:
                    pass
                apply_actuated_params(phase, row)
                # Default phase for dual entry (served when its ring has no call).
                try:
                    phase.setIsDefault(str(row["Phase"]) in default_phases)
                except Exception:
                    pass
                # Coordinated phases get Coordinated recall.
                if is_coordinated and str(row["Phase"]) in coord_phases:
                    apply_coordinated_recall(phase)

                # Protected movements get a plain green; permitted ones the permissive indication.
                np_, pc_, uc_ = add_movement_signals(phase, str(row["Phase"]), protected,
                                                     permitted, signals, permitted_indication)
                n_permitted += np_
                permitted_codes.update(pc_)
                used_movements.update(uc_)

                # Interphase: all-red clearance only, when the phase has all-red.
                if clearance > 0.0:
                    inter = cp_node.createPhase()
                    inter.setFrom(start + green)
                    inter.setDuration(clearance)
                    inter.setInterphase(True)
                    try:
                        inter.setIdRing(int(row["Ring"]))
                        inter.setIdBarrier(int(row["Barrier"]))
                    except Exception:
                        pass
                    try:
                        inter.setYellowTimeDuration(0.0)
                    except Exception:
                        pass
                    n_interphases += 1

            # Mirror the other ring into an empty ring-barrier.
            n_mirrors = 0
            mirror_map = {}
            if MIRROR_EMPTY_RING_BARRIER:
                rings = sorted(int(r) for r in phase_tab["Ring"].unique())
                present = set((int(r), int(b))
                              for r, b in zip(phase_tab["Ring"], phase_tab["Barrier"]))
                for b in sorted(int(x) for x in phase_tab["Barrier"].unique()):
                    rings_in_b = sorted(set(int(r) for r in
                                            phase_tab[phase_tab["Barrier"] == b]["Ring"]))
                    if not rings_in_b:
                        continue
                    src_ring = rings_in_b[0]
                    src_rows = phase_tab[(phase_tab["Barrier"] == b) &
                                         (phase_tab["Ring"] == src_ring)].sort_values("Position")
                    for rg in rings:
                        if (rg, b) in present:
                            continue
                        for _, row in src_rows.iterrows():
                            start = float(row["From"])
                            green = float(row["MaxGreen"]) + float(row["Yellow"])
                            clearance = float(row["AllRed"])

                            mirror = cp_node.createPhase()
                            mirror.setFrom(start)
                            mirror.setDuration(green)
                            mirror.setInterphase(False)
                            try:
                                mirror.setIdRing(rg)
                                mirror.setIdBarrier(int(b))
                            except Exception:
                                pass
                            apply_actuated_params(mirror, row)
                            try:
                                mirror.setIsDefault(str(row["Phase"]) in default_phases)
                            except Exception:
                                pass
                            if is_coordinated and str(row["Phase"]) in coord_phases:
                                apply_coordinated_recall(mirror)
                            # Exactly the same signals as the source phase.
                            add_movement_signals(mirror, str(row["Phase"]), protected,
                                                 permitted, signals, permitted_indication)
                            mirror_map.setdefault(str(row["Phase"]), []).append(mirror)
                            n_mirrors += 1

                            # Mirror the source phase's all-red interphase too.
                            if clearance > 0.0:
                                m_inter = cp_node.createPhase()
                                m_inter.setFrom(start + green)
                                m_inter.setDuration(clearance)
                                m_inter.setInterphase(True)
                                try:
                                    m_inter.setIdRing(rg)
                                    m_inter.setIdBarrier(int(b))
                                except Exception:
                                    pass
                                try:
                                    m_inter.setYellowTimeDuration(0.0)
                                except Exception:
                                    pass
                if n_mirrors:
                    print("    :mirrored %d phase(s) into empty ring-barrier(s)" % n_mirrors)

            # A permissive movement needs a Yield sign on its turning.
            n_giveway = 0
            if SET_GIVEWAY_ON_PERMITTED and permitted_codes:
                giveway = getattr(GKTurning, "eGiveway", None)
                if giveway is None:
                    print("    :GKTurning.eGiveway not available - give-way not set.")
                else:
                    for code in sorted(permitted_codes):
                        signal = signals.get(code)
                        if signal is None:
                            continue
                        for turning in (signal.getTurnings() or []):
                            try:
                                turning.setWarningIndicator(giveway)
                                n_giveway += 1
                            except Exception as exc:
                                print("    :give-way failed for %s: %s" % (code, exc))

            # Detectors: an actuated controller with no detection never advances.
            n_detectors, n_links = 0, 0
            if CREATE_DETECTORS:
                detectors = build_detectors_for_node(model, node, info, signal_dict)
                n_detectors = len(set(id(d) for d in detectors.values()))
                if LINK_DETECTORS_TO_PHASES and detectors:
                    n_links = link_detectors_to_phases(phase_objs, detectors,
                                                       signal_dict, intid, mirror_map)

            # Force-offs and permissive periods for coordination.
            if is_coordinated:
                try:
                    cp_node.calcActuatedForceOff()
                except Exception as exc:
                    print("    :calcActuatedForceOff failed: %s" % exc)

            unassigned = [c for c in signals if c not in used_movements]
            return (len(phase_tab), computed_cycle, synchro_cycle, unassigned,
                    n_permitted, n_giveway, n_detectors, n_links, n_interphases)




        path_matchup = os.path.join(INPUT_DIR, MATCHUP_FILE)
        control_dir = os.path.join(INPUT_DIR, CONTROL_SUBDIR)
        for required in (path_matchup, control_dir):
            if not os.path.exists(required):
                print("  :ERROR - path not found: %s" % required)
                return

        print("  :Reading matchup table ...")
        matchup = read_matchup_table(path_matchup)
        junctions = build_junction_movements(matchup)
        if not junctions:
            print("  :No signalised junctions (no Turn_Synchro) in the matchup table.")
            return

        synchro_file = matchup["File_Synchro"].dropna()
        if synchro_file.empty:
            print("  :ERROR - File_Synchro is empty in the matchup table.")
            return
        path_signal = os.path.join(control_dir, str(synchro_file.iloc[0]).strip())
        if not os.path.exists(path_signal):
            print("  :ERROR - Synchro file not found: %s" % path_signal)
            return

        print("  :Reading Synchro timing from %s ..." % os.path.basename(path_signal))
        signal_dict = process_signal_data(path_signal)

        plan = create_control_plan(model, CONTROL_PLAN_NAME)
        print("  :Created control plan '%s' (junction type: %s)."
              % (CONTROL_PLAN_NAME, CONTROL_JUNCTION_TYPE))

        n_ok = 0
        for node_id in sorted(junctions):
            info = junctions[node_id]
            node = find_node(model, node_id)
            if node is None:
                print("  :node %s not found (or not a GKNode) - skipped." % node_id)
                continue
            try:
                signals = create_signal_groups(model, node, info["movements"])
                (n_phases, computed, synchro_cycle, unassigned, n_permitted,
                 n_giveway, n_detectors, n_links, n_interphases) = build_control_junction(
                    model, plan, node, info, signal_dict, signals)

                flag = "" if abs(computed - synchro_cycle) < 0.5 else \
                       "  <== cycle mismatch (computed %g)" % computed
                print("  :node %s (Synchro %s): %d green + %d interphase(s), "
                      "cycle %g, offset %s, %d permitted, %d give-way, "
                      "%d detector(s) with %d phase link(s)%s"
                      % (node_id, info["synchro"], n_phases, n_interphases,
                         synchro_cycle,
                         _timeplan_value(signal_dict, info["synchro"], "Offset", 0),
                         n_permitted, n_giveway, n_detectors, n_links, flag))
                if unassigned:
                    print("    :movements with no phase (never green): %s" % unassigned)
                if CREATE_DETECTORS and n_links == 0:
                    print("    :WARNING no detector linked to any phase - an "
                          "actuated plan will not advance.")
                n_ok += 1
            except Exception as exc:
                print("  :node %s FAILED: %s" % (node_id, exc))

        try:
            plan.setStatus(GKObject.eModified)
        except Exception:
            pass
        try:
            GKGUISystem.getGUISystem().getActiveGui().invalidateViews()
        except Exception:
            pass

        model.getCommander().addCommand(None)
        print("  :Signal import complete - %d/%d junctions." % (n_ok, len(junctions)))

        console.save( argv[1])
        console.close()
    else:
        console.getLog().addError( "Cannot load the network" )
        print ("cannot load network")


if __name__ == "__main__":
    sys.exit(main(sys.argv))
