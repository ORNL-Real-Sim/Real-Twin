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
"""Apply the conflict-area rules offline and say what they would change.

Reads the conflict areas Vissim detected (from stage 3's ``.inpx``, before
stage 4 rewrites them) and runs :func:`rt_vissim.conflicts.plan_conflicts` on
them exactly as stage 4 does, without a Vissim session.  Prints the tally by
rule, every note the planner makes, and -- the part to read on a new site --
the pairs no rule reached, with both movements' turns and junctions, because
each of those is either a geometry the rules have not met yet or a conflict
between two intersections that no rule should touch.

``--list`` prints every status the rules would change, one line per conflict
area, for checking against the network in the Vissim GUI.

Usage::

    python vissim/scripts/diagnostics/conflict_dryrun.py --work vissim/work/chattanooga
    python vissim/scripts/diagnostics/conflict_dryrun.py --work vissim/work/phoenix --list
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd

from _work import elements, find_build  # noqa: E402

from rt_vissim.conflicts import (_junction, degenerate_uturn_paths,  # noqa: E402
                                 link_turns, major_roads, plan_conflicts, summarise)
from rt_vissim.matchup import MatchupTable  # noqa: E402
from rt_vissim.network import read_links_csv  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--work", required=True, help="Build folder, e.g. vissim/work/chattanooga")
    parser.add_argument("--inpx", default=None,
                        help="Network holding Vissim's own conflict areas (default: stage 3's output)")
    parser.add_argument("--list", action="store_true", help="Print every change, one per line")
    args = parser.parse_args(argv)

    build = find_build(args.work)
    inpx = Path(args.inpx) if args.inpx else build.inpx("demand")
    for path in (inpx, build.links, build.movements, build.matchup):
        if not path.exists():
            print(f"  :Not found: {path}")
            return 1

    links = read_links_csv(build.links)
    moves = pd.read_csv(build.movements)
    signalised = MatchupTable(build.matchup).signalized_junctions()
    pairs = [(int(a["no"]), int(a["linkA"]), int(a["linkB"]), a["status"])
             for a in elements(inpx, "conflictArea")]
    owners = link_turns(moves, links)
    decisions, notes = plan_conflicts(
        pairs, owners, degenerate_uturn_paths(links, moves),
        major_roads(moves, links, signalised=signalised), links, signalised=signalised)

    print(f"  :{inpx.name}: {summarise(decisions, pairs)}")
    for note in notes:
        print(f"  :{note}")

    # What no rule reached.  The planner returns only the statuses it would
    # change, so a pair already at the status a rule wants looks the same as a
    # pair no rule covers.  Telling them apart: plan twice more with every
    # status forced to A and then to B -- a pair any rule decides is changed in
    # at least one of those runs; a pair no rule decides is changed in neither.
    reached: set[int] = set()
    for forced in ("AHASRIGHTOFWAY", "BHASRIGHTOFWAY"):
        fake = [(no, a, b, forced) for no, a, b, _ in pairs]
        changed, _ = plan_conflicts(
            fake, owners, degenerate_uturn_paths(links, moves),
            major_roads(moves, links, signalised=signalised), links, signalised=signalised)
        reached |= set(changed)
    controlled = {_junction(j) for j in signalised}
    unreached: Counter = Counter()
    examples: dict = {}
    for no, a, b, status in pairs:
        if no in reached or a not in owners or b not in owners:
            continue
        oa, ob = owners[a], owners[b]
        same = oa.junction == ob.junction
        key = (("same junction, " if same else "two junctions, ")
               + f"{oa.turn} vs {ob.turn}"
               + (" at a signal" if same and oa.junction in controlled else ""))
        unreached[key] += 1
        examples.setdefault(key, f"conflict {no}: J{oa.junction} {oa.turn} on {a} vs J{ob.junction} {ob.turn} on {b} ({status})")
    print(f"\n  :{sum(unreached.values())} conflicts between two movements that no rule decides "
          f"(equal rank at a signal or head-on, or a passive area, are expected; anything else "
          f"within one junction is a geometry the rules have not met):")
    for key, n in unreached.most_common():
        print(f"    {n:5d}  {key:40s} e.g. {examples[key]}")

    if args.list:
        status = {no: st for no, _, _, st in pairs}
        print(f"\n  :{len(decisions)} changes:")
        for no, wanted in sorted(decisions.items()):
            _, a, b, _ = next(p for p in pairs if p[0] == no)
            oa, ob = owners.get(a), owners.get(b)
            print(f"    {no:5d}  {status[no]:16s} -> {wanted:16s} "
                  f"A {a} ({oa.turn if oa else '-'})  B {b} ({ob.turn if ob else '-'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
