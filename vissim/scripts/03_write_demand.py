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
"""Stage 3: write the GridSmart demand into the Vissim network.

Loads the network stage 1 imported, reads the MatchupTable stage 2 filled,
builds the demand IR and creates the Vissim vehicle inputs, then saves a new
``.inpx`` so the result can be opened and inspected.

Only vehicle inputs are written today; routing decisions follow.

Usage::

    python vissim/scripts/03_write_demand.py --open-gui
    python vissim/scripts/03_write_demand.py --start 07:00 --end 09:00
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(1, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402

from rt_vissim.com import VissimSession  # noqa: E402
from rt_vissim.demand import build_demand, build_turn_counts  # noqa: E402
from rt_vissim.matchup import MatchupTable  # noqa: E402
from rt_vissim.network import read_links_csv  # noqa: E402
from rt_vissim.routes import build_integrated_decisions, summarise  # noqa: E402
from rt_vissim.writer import write_routing_decisions, write_vehicle_inputs  # noqa: E402


def clock_to_seconds(text: str) -> float:
    """Convert ``"HH:MM"`` to seconds after midnight."""
    hour, minute = (int(part) for part in text.split(":"))
    return hour * 3600 + minute * 60


def open_in_gui(inpx_path: Path) -> bool:
    """Open a saved network in a standalone Vissim GUI.

    Shares stage 1's implementation so both stages behave identically.
    """
    spec = importlib.util.spec_from_file_location(
        "stage01", Path(__file__).with_name("01_import_opendrive.py"))
    stage01 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stage01)
    return stage01.open_in_gui(inpx_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--inpx", default="vissim/work/chattanooga/chatt.inpx",
                        help="Network written by stage 1")
    parser.add_argument("--matchup", default="vissim/work/chattanooga/MatchupTable.xlsx",
                        help="MatchupTable written by stage 2")
    parser.add_argument("--links", default="vissim/work/chattanooga/chatt_links.csv",
                        help="Link table written by stage 1")
    parser.add_argument("--traffic-dir", default="datasets/chattanooga/Traffic",
                        help="Directory holding the GridSmart files")
    parser.add_argument("--out", default=None,
                        help="Output .inpx (default: <inpx stem>_demand.inpx)")
    parser.add_argument("--start", default="07:00", help="Scenario start, HH:MM")
    parser.add_argument("--end", default="09:00", help="Scenario end, HH:MM")
    parser.add_argument("--progid", default=None, help="Pin a Vissim COM ProgID")
    parser.add_argument("--visible", action="store_true", help="Show the Vissim GUI")
    parser.add_argument("--open-gui", action="store_true",
                        help="Open the saved network in a standalone GUI")
    parser.add_argument("--movements", default="vissim/work/chattanooga/chatt_movements.csv",
                        help="Movement table written by stage 1, for the routing decisions")
    parser.add_argument("--no-routes", action="store_true",
                        help="Write vehicle inputs only, skip routing decisions")
    args = parser.parse_args(argv)

    inpx_path = Path(args.inpx).resolve()
    for label, path in (("Network", inpx_path), ("MatchupTable", Path(args.matchup)),
                        ("Link table", Path(args.links))):
        if not path.exists():
            print(f"  :{label} not found: {path}")
            return 1

    start_time, end_time = clock_to_seconds(args.start), clock_to_seconds(args.end)
    out_path = (Path(args.out).resolve() if args.out
                else inpx_path.with_name(f"{inpx_path.stem}_demand.inpx"))

    # Build the demand before starting Vissim: it needs no licence, so a bad
    # MatchupTable fails fast instead of after a slow COM start and network load.
    links = read_links_csv(args.links)
    matchup = MatchupTable(args.matchup)
    inputs, decisions, warnings = build_demand(
        matchup, links, args.traffic_dir,
        sim_start_time=start_time, sim_end_time=end_time)

    print(f"  :{len(links)} links, {matchup.df['JunctionID_Vissim'].nunique()} junctions")
    print(f"  :Demand {args.start}-{args.end}: {len(inputs)} vehicle inputs, "
          f"{len(decisions)} routing decisions")
    for warning in warnings:
        print(f"  :WARNING: {warning}")
    if not inputs:
        print("  :Nothing to write.")
        return 1

    total = sum(vi.count for vi in inputs)
    print(f"  :{total} vehicles across {len({vi.link_no for vi in inputs})} origin links, "
          f"{len({(vi.interval_start, vi.interval_end) for vi in inputs})} time intervals")

    integrated: list = []
    if not args.no_routes:
        movements = pd.read_csv(args.movements)
        turn_counts = build_turn_counts(matchup, args.traffic_dir)
        turn_counts = turn_counts[(turn_counts["IntervalStart"] >= start_time)
                                  & (turn_counts["IntervalEnd"] <= end_time)]
        integrated, route_warnings = build_integrated_decisions(movements, turn_counts)
        for warning in route_warnings:
            print(f"  :WARNING: {warning}")
        print(f"  :Routing: {summarise(integrated)}")

    with VissimSession(args.progid, visible=args.visible) as session:
        print(f"  :Loading {inpx_path.name} ...")
        session.load_net(inpx_path)
        session.configure_simulation(start_time=start_time, end_time=end_time)

        created, write_warnings = write_vehicle_inputs(session, inputs, start_time)
        for warning in write_warnings:
            print(f"  :WARNING: {warning}")
        print(f"  :Created {created} vehicle inputs")

        if integrated:
            n_dec, n_routes, route_warnings = write_routing_decisions(
                session, integrated, start_time, links)
            for warning in route_warnings:
                print(f"  :{warning}")
            print(f"  :Created {n_dec} routing decisions, {n_routes} routes")

        session.save_net_as(out_path)
        print(f"  :Wrote {out_path}")

    if args.open_gui:
        open_in_gui(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
