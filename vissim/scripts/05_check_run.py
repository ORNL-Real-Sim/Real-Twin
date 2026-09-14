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
"""Stage 5: say what Vissim recorded while the model was running.

Vissim writes an ``.err`` file beside every network and notes in it each
vehicle it had to remove, with the link, the route and the reason.  Nothing
reads it, so a model can delete a steady fraction of its traffic and look
perfectly healthy: the phases cycle, no two greens conflict, and the counts
that *do* arrive match.  Chattanooga was losing 101 vehicles of 4,061 that way,
and it took watching the animation to notice.

So run this after a simulation.  It reports:

* **vehicles removed**, split by cause and grouped by link.  A vehicle stranded
  at a link end could not reach the lane its turn leaves from -- usually a
  routing decision too close to the junction for the lane change it implies.
  One removed after waiting knew its route and never found a gap, which is
  congestion rather than information.
* **warnings from when the network was built**, which often predict the first
  kind.  Vissim says outright when a routing decision sits a few metres from
  its first connector.

Usage::

    python vissim/scripts/05_check_run.py
    python vissim/scripts/05_check_run.py --inpx vissim/work/chattanooga/chatt_demand_signals.inpx
    python vissim/scripts/05_check_run.py --run 900      # simulate first
"""

from __future__ import annotations

import argparse
import collections
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(1, str(Path(__file__).resolve().parents[2]))

#: A vehicle that reached the end of a link and could not enter the connector
#: its route needs, because it is in the wrong lane and has no room to move.
STRANDED = re.compile(
    r'Vehicle (\d+) \(on Static Vehicle Route "([\d ]+-[\d ]+)"\) reached the '
    r'end of link "(\d+):[^"]*" without finding the subsequent link \("(\d+)"\)')

#: A vehicle that waited for a gap to change lanes until Vissim gave up on it.
WAITED = re.compile(
    r'After ([\d.]+) seconds of waiting for lane change the vehicle (\d+)'
    r'(?: \(on Static Vehicle Route "([\d ]+-[\d ]+)"\))?')

#: Vissim's own warning that a decision leaves no room for the lane change.
TOO_CLOSE = re.compile(
    r'Static Vehicle Routing Decision (\d+): ([^\t]*?) is located only '
    r'([\d.]+) m upstream of the first connector')


def read_messages(paths: list[Path]) -> list[str]:
    """Return every message line from the error files, newest file last."""
    messages: list[str] = []
    for path in sorted(paths, key=lambda p: p.stat().st_mtime):
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            fields = [f.strip() for f in line.split("\t") if f.strip()]
            if len(fields) >= 2:
                messages.append(fields[1])
    return messages


def summarise(messages: list[str]) -> dict:
    """Group the messages into the things worth acting on."""
    stranded: collections.Counter = collections.Counter()
    stranded_detail: dict = {}
    waited: collections.Counter = collections.Counter()
    too_close: list[tuple[str, str, float]] = []
    other: collections.Counter = collections.Counter()
    other_example: dict[str, str] = {}

    for message in messages:
        found = STRANDED.search(message)
        if found:
            link = found.group(3)
            stranded[link] += 1
            stranded_detail.setdefault(link, (found.group(2), found.group(4)))
            continue
        found = WAITED.search(message)
        if found:
            waited[found.group(3) or "unknown route"] += 1
            continue
        found = TOO_CLOSE.search(message)
        if found:
            too_close.append((found.group(1), found.group(2),
                              float(found.group(3))))
            continue
        # Anything else, collapsed so a thousand identical lines read as one --
        # but keep a real example of each, because the numbers are the point:
        # "SG N of controller N" says nothing about which signal to go and look
        # at.
        shape = re.sub(r"\d+(\.\d+)?", "N", message)[:110]
        other[shape] += 1
        other_example.setdefault(shape, message)

    return {"stranded": stranded, "detail": stranded_detail, "waited": waited,
            "too_close": too_close, "other": other, "example": other_example}


def report(found: dict, released: int | None) -> int:
    """Print the summary.  Returns the number of vehicles lost."""
    lost = sum(found["stranded"].values()) + sum(found["waited"].values())

    if found["too_close"]:
        print(f"  :{len(found['too_close'])} routing decisions sit close to their "
              "first connector, which is where vehicles get stranded:")
        for number, name, metres in sorted(found["too_close"],
                                           key=lambda t: t[2]):
            print(f"  :   decision {number}: {metres:.2f} m -- {name}")

    if not lost:
        print("  :No vehicles were removed during the run.")
    else:
        share = f" ({lost / released:.2%} of {released:,})" if released else ""
        print(f"  :{lost} vehicles were removed during the run{share}.")

    if found["stranded"]:
        total = sum(found["stranded"].values())
        print(f"  :   {total} reached the end of a link and could not enter the "
              "connector their route needs. They were in the wrong lane with no "
              "room left to move over:")
        for link, count in found["stranded"].most_common():
            route, connector = found["detail"][link]
            print(f"  :      link {link}: {count} vehicles, route {route} "
                  f"into connector {connector}")
        print("  :   Consider combining the routing decisions so vehicles learn "
              "the turn earlier (stage 3 does this by default), or a longer "
              "approach where there is no upstream decision to combine with.")

    if found["waited"]:
        total = sum(found["waited"].values())
        print(f"  :   {total} waited for a lane change until Vissim removed them. "
              "They knew the route and never found a gap, so this is congestion "
              "or geometry rather than information:")
        for route, count in found["waited"].most_common(5):
            print(f"  :      route {route}: {count} vehicles")

    if found["other"]:
        print(f"  :{sum(found['other'].values())} other messages:")
        for shape, count in found["other"].most_common(5):
            example = found["example"].get(shape, shape)
            print(f"  :   x{count} {example}")
            if count > 1:
                print(f"  :        (and {count - 1} like it)")

    return lost


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--inpx",
                        default="vissim/work/chattanooga/chatt_demand_signals.inpx",
                        help="The model whose error files should be read")
    parser.add_argument("--run", type=int, default=0, metavar="SECONDS",
                        help="Simulate for this many seconds first, discarding "
                             "any earlier error files")
    parser.add_argument("--progid", default=None, help="Pin a Vissim COM ProgID")
    parser.add_argument("--fail-over", type=int, default=None, metavar="N",
                        help="Exit non-zero if more than N vehicles were removed")
    args = parser.parse_args(argv)

    net = Path(args.inpx).resolve()
    if not net.exists():
        print(f"  :Network not found: {net}")
        return 1

    released = None
    if args.run:
        from rt_vissim.com import VissimSession  # noqa: PLC0415 - needs a licence

        for stale in net.parent.glob(f"{net.stem}*.err"):
            stale.unlink()
        with VissimSession(args.progid, visible=False) as session:
            session.load_net(net)
            simulation = session.net.Simulation
            simulation.SetAttValue("SimPeriod", args.run)
            simulation.SetAttValue("SimBreakAt", args.run)
            print(f"  :Running {net.name} for {args.run}s ...")
            simulation.RunContinuous()

    errors = sorted(net.parent.glob(f"{net.stem}*.err"))
    if not errors:
        print(f"  :No error file beside {net.name}. Run the simulation first, "
              "in the GUI or with --run.")
        return 0

    print(f"  :Reading {len(errors)} error file(s) for {net.stem}")
    found = summarise(read_messages(errors))
    lost = report(found, released)

    if args.fail_over is not None and lost > args.fail_over:
        print(f"  :{lost} removed, more than the {args.fail_over} allowed.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
