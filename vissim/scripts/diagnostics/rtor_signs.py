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
"""Right turn on red: Synchro's raw permission against the stop signs in the model.

Synchro records right-turn-on-red per movement as the ``Allow RTOR`` row of its
``Lanes`` table.  Stage 4 turns each permitted right turn into a stop sign on
the turn's connector, bound to the turn's signal group, which Vissim reads as
"stop only while that group is red".  This prints the raw row for every
signalised intersection, then every right turn at a signalised junction with
whether Synchro permits it and whether the model carries the sign, and flags
each mismatch.

The ``OnlyOnRed`` flag is a *calculated* attribute: Vissim derives it from the
signal-group binding and never writes it to the ``.inpx``.  So a sign with an
``sg`` attribute is a sign that stops on red only.  ``--flags`` opens Vissim
and reads the flag back for every sign, which is the only way to see it.

Usage::

    python vissim/scripts/diagnostics/rtor_signs.py --work vissim/work/chattanooga
    python vissim/scripts/diagnostics/rtor_signs.py --work vissim/work/phoenix --flags
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from _work import elements, find_build  # noqa: E402

from rt_vissim.dataset import data_dir  # noqa: E402
from rt_vissim.heads import rtor_allowed  # noqa: E402
from rt_vissim.matchup import MatchupTable  # noqa: E402
from rt_vissim.signal import build_signal_plans, read_synchro  # noqa: E402


def read_flags(inpx: Path) -> list[tuple]:
    """``(no, name, lane, sg, only_on_red)`` for every stop sign, over COM."""
    from rt_vissim.com import VissimSession
    with VissimSession(visible=False) as session:
        session.load_net(inpx)
        signs = session.net.StopSigns
        columns = [[v[1] for v in signs.GetMultiAttValues(a)]
                   for a in ("No", "Name", "Lane", "SG", "OnlyOnRed")]
    return list(zip(*columns))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--work", required=True, help="Build folder, e.g. vissim/work/chattanooga")
    parser.add_argument("--inpx", default=None, help="Network to check (default: stage 4's output)")
    parser.add_argument("--control-dir", default=None, help="Folder holding the Synchro file")
    parser.add_argument("--flags", action="store_true",
                        help="Open Vissim and read every sign's OnlyOnRed flag")
    args = parser.parse_args(argv)

    build = find_build(args.work)
    inpx = Path(args.inpx) if args.inpx else build.inpx("signals")
    for path in (inpx, build.matchup):
        if not path.exists():
            print(f"  :Not found: {path}")
            return 1
    matchup = MatchupTable(build.matchup)
    control = Path(data_dir("control", args.control_dir, build.work))
    utdf = control / matchup.synchro_file()
    print(f"  :Synchro file: {utdf}")
    synchro = read_synchro(utdf)
    lanes = synchro["Lanes"]
    plans, _ = build_signal_plans(synchro, matchup.signalized_junctions())

    record = next(c for c in lanes.columns if c.upper() == "RECORDNAME")
    print("\nRaw 'Allow RTOR' rows (Synchro Lanes table, 1 = permitted):")
    print(lanes[lanes[record].astype(str).str.strip() == "Allow RTOR"].to_string(index=False))

    allowed = rtor_allowed(matchup, synchro, plans)
    signed = set()
    for sign in elements(inpx, "stopSign"):
        found = re.search(r"RTOR (\w+) \(J(\d+)\)", sign.get("name", ""))
        if found:
            signed.add((found.group(2), found.group(1)))

    print(f"\nRight turns at signalised junctions in {inpx.name}:")
    print(f"  {'junction':10s} {'INTID':6s} {'move':5s} {'Allow RTOR':11s} sign in model")
    mismatches = 0
    for plan in plans:
        for (junction, code), permitted in sorted(allowed.items()):
            if junction != str(plan.junction_id) or not code.endswith("R"):
                continue
            has = (junction, code) in signed
            flag = "" if has == permitted else "   <-- mismatch"
            mismatches += has != permitted
            print(f"  J{junction:9s} {str(plan.synchro_intid):6s} {code:5s} "
                  f"{'1' if permitted else '0':11s} {'yes' if has else 'no'}{flag}")
    permitted = sum(1 for (j, c), ok in allowed.items() if c.endswith("R") and ok)
    print(f"\n{permitted} right turns permitted on red, {len(signed)} with a sign, "
          f"{mismatches} mismatches")

    if args.flags:
        rows = read_flags(inpx)
        print(f"\n{len(rows)} stop signs in Vissim:")
        for no, name, lane, sg, only_red in rows:
            print(f"  {no:3d} {name:18s} lane {str(lane):10s} SG {str(sg):8s} OnlyOnRed={only_red}")
        print(f"OnlyOnRed true on {sum(1 for r in rows if r[4])} of {len(rows)}")
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
