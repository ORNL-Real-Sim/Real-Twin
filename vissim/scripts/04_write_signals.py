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
"""Stage 4: write the Synchro signal control into the Vissim network.

Reads the Synchro UTDF export the MatchupTable names, writes a Ring Barrier
Controller file per signalised junction, then creates the controllers, signal
groups, signal heads and detectors over COM and saves a new ``.inpx``.

The timings stay actuated: the ``.prbc`` files hold the recalls, extensions and
minimum greens, and Vissim's RBC runs them.  RealTwin's SUMO path has to flatten
the same data to fixed time because SUMO has no equivalent.

Usage::

    python vissim/scripts/04_write_signals.py
    python vissim/scripts/04_write_signals.py --inpx vissim/work/chattanooga/chatt_demand.inpx
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(1, str(Path(__file__).resolve().parents[2]))

from rt_vissim.com import VissimSession  # noqa: E402
from rt_vissim.conflicts import (degenerate_uturn_paths,  # noqa: E402
                                 link_turns, plan_conflicts)
from rt_vissim.conflicts import summarise as summarise_conflicts  # noqa: E402
from rt_vissim.heads import (build_detectors, build_signal_heads,  # noqa: E402
                             rtor_allowed, summarise as summarise_placement)
from rt_vissim.matchup import MatchupTable  # noqa: E402
from rt_vissim.network import read_links_csv  # noqa: E402
from rt_vissim.rbc import write_controllers  # noqa: E402
from rt_vissim.signal import (build_signal_plans, read_synchro,  # noqa: E402
                              summarise as summarise_plans)
from rt_vissim.writer import (clear_signal_control, read_conflict_areas,  # noqa: E402
                              write_conflict_areas, write_detectors,
                              write_rtor_stop_signs, write_signal_controllers,
                              write_signal_heads)
import pandas as pd  # noqa: E402


def open_in_gui(inpx_path: Path) -> bool:
    """Open a saved network in a standalone Vissim GUI, as stage 1 does."""
    spec = importlib.util.spec_from_file_location(
        "stage01", Path(__file__).with_name("01_import_opendrive.py"))
    stage01 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stage01)
    return stage01.open_in_gui(inpx_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--inpx", default="vissim/work/chattanooga/chatt_demand.inpx",
                        help="Network to add signals to (stage 3's output)")
    parser.add_argument("--matchup", default="vissim/work/chattanooga/MatchupTable.xlsx",
                        help="MatchupTable written by stage 2")
    parser.add_argument("--links", default="vissim/work/chattanooga/chatt_links.csv",
                        help="Link table written by stage 1")
    parser.add_argument("--control-dir", default="datasets/example1/Control",
                        help="Directory holding the Synchro UTDF file")
    parser.add_argument("--out", default=None,
                        help="Output .inpx (default: <name>_signals.inpx)")
    parser.add_argument("--progid", default=None, help="Pin a Vissim COM ProgID")
    parser.add_argument("--visible", action="store_true", help="Show the Vissim GUI")
    parser.add_argument("--open-gui", action="store_true",
                        help="Open the saved network in a standalone GUI")
    parser.add_argument("--movements", default="vissim/work/chattanooga/chatt_movements.csv",
                        help="Movement table written by stage 1, for conflict areas")
    parser.add_argument("--no-conflicts", action="store_true",
                        help="Leave the conflict areas as Vissim detected them, so "
                             "permissive turns do not yield")
    parser.add_argument("--no-rtor", action="store_true",
                        help="Skip the right-turn-on-red stop signs")
    parser.add_argument("--no-heads", action="store_true",
                        help="Write the controllers only, skip heads and detectors")
    parser.add_argument("--keep-opendrive-signals", action="store_true",
                        help="Keep the signal controllers and heads Vissim built "
                             "from the OpenDRIVE file instead of replacing them "
                             "with the Synchro timings. Those controllers carry a "
                             "single signal group each, so they cannot express "
                             "NEMA phasing, and their numbering can collide with "
                             "the Synchro INTIDs")
    args = parser.parse_args(argv)

    inpx_path = Path(args.inpx).resolve()
    for label, path in (("Network", inpx_path), ("MatchupTable", Path(args.matchup)),
                        ("Link table", Path(args.links))):
        if not path.exists():
            print(f"  :{label} not found: {path}")
            return 1

    out_path = (Path(args.out).resolve() if args.out
                else inpx_path.with_name(f"{inpx_path.stem}_signals.inpx"))

    # Everything up to the COM calls needs no licence, so a bad Synchro file
    # fails here rather than after a slow start and network load.
    matchup = MatchupTable(args.matchup)
    links = read_links_csv(args.links)
    junctions = matchup.signalized_junctions()
    if not junctions:
        print("  :No signalised junctions in the MatchupTable "
              "(no Turn_Synchro codes); nothing to do.")
        return 1

    utdf = Path(args.control_dir) / matchup.synchro_file()
    if not utdf.exists():
        print(f"  :Synchro file not found: {utdf}")
        return 1
    synchro = read_synchro(utdf)

    # Label each controller with the junction and intersection it serves.  The
    # controller number is the Synchro INTID, and those cross over badly against
    # the OpenDRIVE junction numbers -- junction 14 is INTID 4 while junction 4
    # is INTID 19 -- so without a name there is nothing in the model tying a
    # controller to the intersection you are looking at.
    frame = matchup.df.copy()
    frame["J"] = frame["JunctionID_OpenDrive"].ffill()
    labels = {}
    for junction_id, rows in frame.groupby("J"):
        found = [v for v in rows["IntersectionName_GridSmart"].dropna().unique() if str(v).strip()]
        if found:
            labels[str(junction_id)] = f"J{junction_id} {found[0]}"
    plans, plan_warnings = build_signal_plans(synchro, junctions, names=labels)
    print(f"  :{len(junctions)} signalised junctions from {utdf.name}")
    print(f"  :Plans: {summarise_plans(plans)}")
    for warning in plan_warnings:
        print(f"  :WARNING: {warning}")
    if not plans:
        print("  :No usable signal plans.")
        return 1

    prbc_dir = out_path.parent
    written, rbc_warnings = write_controllers(plans, prbc_dir)
    for warning in rbc_warnings:
        print(f"  :WARNING: {warning}")
    print(f"  :Wrote {len(written)} .prbc controller files into {prbc_dir}")

    heads, detectors = [], []
    if not args.no_heads:
        heads, head_warnings = build_signal_heads(matchup, links, synchro, plans)
        detectors, det_warnings = build_detectors(matchup, links, synchro, plans)
        print(f"  :Placement: {summarise_placement(heads, detectors)}")
        for warning in head_warnings + det_warnings:
            print(f"  :WARNING: {warning}")

    with VissimSession(args.progid, visible=args.visible) as session:
        print(f"  :Loading {inpx_path.name} ...")
        session.load_net(inpx_path)

        if not args.keep_opendrive_signals:
            _, _, warnings = clear_signal_control(session)
            for warning in warnings:
                print(f"  :{warning}")
        else:
            print("  :Keeping the OpenDRIVE signal control; Synchro controllers "
                  "are added alongside it and the numbering may clash.")

        made, warnings = write_signal_controllers(session, plans, prbc_dir)
        for warning in warnings:
            print(f"  :WARNING: {warning}")
        print(f"  :Created {made} signal controllers")

        if heads:
            made, warnings = write_signal_heads(session, heads)
            for warning in warnings:
                print(f"  :{warning}")
            print(f"  :Created {made} signal heads")

        if detectors:
            made, warnings = write_detectors(session, detectors)
            for warning in warnings:
                print(f"  :{warning}")
            print(f"  :Created {made} detectors")

        if not args.no_conflicts and Path(args.movements).exists():
            pairs = read_conflict_areas(session)
            owners = link_turns(pd.read_csv(args.movements), links)
            moves = pd.read_csv(args.movements)
            decisions, notes = plan_conflicts(
                pairs, owners, degenerate_uturn_paths(links, moves))
            print(f"  :Conflicts: {summarise_conflicts(decisions, pairs)}")
            for note in notes:
                print(f"  :{note}")
            made, warnings = write_conflict_areas(session, decisions)
            for warning in warnings:
                print(f"  :{warning}")

        if heads and not args.no_rtor:
            allowed = rtor_allowed(matchup, synchro, plans)
            made, warnings = write_rtor_stop_signs(session, heads, allowed)
            for warning in warnings:
                print(f"  :{warning}")
            print(f"  :Created {made} right-turn-on-red stop signs")

        session.save_net_as(out_path)
        print(f"  :Wrote {out_path}")

    if args.open_gui:
        open_in_gui(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
