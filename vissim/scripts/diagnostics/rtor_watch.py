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
"""Watch the right-turn-on-red signs during a run: on red, do vehicles stop, then go?

Steps the simulation one step at a time.  Each vehicle entering a connector
that carries an RTOR stop sign is classified by the sign's signal group at that
moment (red, including amber and red-amber, or green) and by whether it came
to a halt -- under 0.5 m/s -- in the 20 s before entering.  The healthy
picture is "on red, after stopping" and "on green, without stopping"; every
"on red, without stopping" entry is printed as it happens so it can be
explained.

Two explanations seen so far that are not faults: a vehicle rolling at a few
km/h on amber that never quite halted, and a vehicle created on a stub entry
link shorter than its own look-ahead, which meets the sign before it has had
time to brake (Chattanooga's link 43).

The resolution must be a multiple of every controller's frequency: Synchro's
ring-barrier controllers run at 10 steps a second, so 5 fails.

Usage::

    python vissim/scripts/diagnostics/rtor_watch.py --work vissim/work/chattanooga
    python vissim/scripts/diagnostics/rtor_watch.py --work vissim/work/phoenix --seconds 600
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from pathlib import Path

from _work import elements, find_build  # noqa: E402

from rt_vissim.com import VissimSession  # noqa: E402

STOPPED = 0.5    # m/s
LOOKBACK = 20    # s
RED = ("RED", "REDAMBER", "AMBER")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--work", required=True, help="Build folder, e.g. vissim/work/chattanooga")
    parser.add_argument("--inpx", default=None, help="Network to run (default: stage 4's output)")
    parser.add_argument("--seconds", type=int, default=900, help="Length of the run")
    parser.add_argument("--start", type=float, default=8 * 3600,
                        help="Start time in seconds after midnight (default 08:00)")
    parser.add_argument("--resolution", type=int, default=10, help="Steps per second")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    build = find_build(args.work)
    inpx = Path(args.inpx) if args.inpx else build.inpx("signals")
    if not inpx.exists():
        print(f"  :Not found: {inpx}")
        return 1

    signs = []   # (name, connector, (controller, group))
    for sign in elements(inpx, "stopSign"):
        if "sg" in sign:
            controller, group = (int(x) for x in sign["sg"].split())
            signs.append((sign["name"], int(sign["lane"].split()[0]), (controller, group)))
    by_link: dict[int, list] = defaultdict(list)
    for name, link, ref in signs:
        by_link[link].append((name, ref))
    print(f"  :{len(signs)} RTOR stop signs on {len(by_link)} connectors in {inpx.name}")
    if not signs:
        return 1

    res = args.resolution
    tally: Counter = Counter()
    per_sign: dict[str, Counter] = defaultdict(Counter)
    with VissimSession(visible=False) as session:
        session.load_net(inpx)
        session.configure_simulation(start_time=args.start, end_time=args.start + args.seconds,
                                     resolution=res, seed=args.seed)
        sim = session.vissim.Simulation
        groups = {ref: session.net.SignalControllers.ItemByKey(ref[0]).SGs.ItemByKey(ref[1])
                  for ref in {ref for _, _, ref in signs}}
        vehicles = session.net.Vehicles
        history: dict = defaultdict(lambda: deque(maxlen=LOOKBACK * res))   # recent speeds, m/s
        where: dict[int, int] = {}                                           # link last seen on
        for step in range(args.seconds * res):
            sim.RunSingleStep()
            nos = [v[1] for v in vehicles.GetMultiAttValues("No")]
            lanes = [v[1] for v in vehicles.GetMultiAttValues("Lane")]
            speeds = [v[1] for v in vehicles.GetMultiAttValues("Speed")]
            states = {ref: g.AttValue("SigState") for ref, g in groups.items()}
            seen = set()
            for no, lane, kmh in zip(nos, lanes, speeds):
                seen.add(no)
                link = int(str(lane).split("-")[0])
                if link in by_link and where.get(no) != link:
                    halted = any(v < STOPPED for v in history[no])
                    for name, ref in by_link[link]:
                        state = states[ref]
                        red = state in RED
                        key = (("on red" if red else "on green")
                               + (", after stopping" if halted else ", without stopping"))
                        tally[key] += 1
                        per_sign[name][key] += 1
                        if red and not halted:
                            low = min(history[no], default=99) * 3.6
                            print(f"  t={step / res:6.1f}s  {name:16s} SG {ref} {state:8s} veh {no} "
                                  f"entered at {kmh:5.1f} km/h, min {low:5.1f} km/h over the last "
                                  f"{len(history[no]) / res:.0f}s", flush=True)
                where[no] = link
                history[no].append(kmh / 3.6)
            for no in [n for n in where if n not in seen]:
                where.pop(no, None)
                history.pop(no, None)
            if step % (60 * res) == 0:
                print(f"  t={step // res:4d}s  {dict(tally)}", flush=True)

    print(f"\nVehicles entering an RTOR connector in {args.seconds}s:")
    for key in sorted(tally):
        print(f"  {tally[key]:5d}  {key}")
    print("\nPer sign:")
    for name in sorted(per_sign):
        c = per_sign[name]
        print(f"  {name:22s} red/stopped {c['on red, after stopping']:3d}  "
              f"red/no stop {c['on red, without stopping']:3d}  "
              f"green/stopped {c['on green, after stopping']:3d}  "
              f"green/no stop {c['on green, without stopping']:3d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
