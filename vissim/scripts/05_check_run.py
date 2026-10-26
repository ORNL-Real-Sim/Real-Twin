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
* with ``--compare-counts``, **what was ingested against what the run
  delivered**: every counted movement's volume and every approach's turn
  shares, split into approaches fed straight from a counted entry (where they
  must agree) and approaches fed from upstream junctions (where only the
  shares can agree until stage 6 estimates the missing demand).  See
  :mod:`rt_vissim.verify`.

Usage::

    python vissim/scripts/05_check_run.py
    python vissim/scripts/05_check_run.py --inpx vissim/work/chattanooga/chatt_demand_signals.inpx
    python vissim/scripts/05_check_run.py --run 900      # simulate first
    python vissim/scripts/05_check_run.py --compare-counts \
        --matchup vissim/work/mynet/MatchupTable.xlsx \
        --links vissim/work/mynet/chatt_links.csv --traffic-dir datasets/chattanooga/Traffic
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


def _seconds(clock: str) -> int:
    hours, minutes = clock.split(":")
    return int(hours) * 3600 + int(minutes) * 60


def prepare_comparison(args) -> dict:
    """The counts for the period and where to count them -- no Vissim needed."""
    from rt_vissim.calibrate import counting_sites  # noqa: PLC0415
    from rt_vissim.demand import build_turn_counts  # noqa: PLC0415
    from rt_vissim.matchup import MatchupTable  # noqa: PLC0415
    from rt_vissim.network import read_links_csv  # noqa: PLC0415
    from rt_vissim.verify import approach_feeds  # noqa: PLC0415

    start, end = _seconds(args.start), _seconds(args.end)
    counts = build_turn_counts(MatchupTable(args.matchup), Path(args.traffic_dir))
    counts = counts[(counts.IntervalStart >= start) & (counts.IntervalEnd <= end)]
    grouped = counts.groupby(["FromLinkNo_Vissim", "ToLinkNo_Vissim"]).Count.sum()
    counted = {(int(a), int(b)): float(v) for (a, b), v in grouped.items() if v > 0}
    links = read_links_csv(args.links)
    sites, notes = counting_sites(links, counted)
    for note in notes:
        print(f"  :NOTE: {note}")
    return {"counted": counted, "sites": sites, "period": end - start,
            "feeds": approach_feeds(links, {a for a, _b in counted})}


def place_comparison_counters(session, comparison: dict, period: int) -> None:
    """Data collection on every counted movement and across every counted entry."""
    from rt_vissim.calibrate import place_counters  # noqa: PLC0415

    net = session.net
    comparison["measures"] = place_counters(session, comparison["sites"])
    comparison["entries"] = {}
    for vehicle_input in net.VehicleInputs.GetAll():
        volumes = []
        for interval in range(1, 100):
            try:
                volumes.append(float(vehicle_input.AttValue(f"Volume({interval})")))
            except Exception:  # noqa: BLE001 - past the last interval
                break
        if not sum(volumes):
            continue
        link = vehicle_input.Link
        position = min(2.0, float(link.AttValue("Length2D")) / 2)
        points = [str(net.DataCollectionPoints.AddDataCollectionPoint(0, lane, position)
                      .AttValue("No")) for lane in link.Lanes.GetAll()]
        measure = net.DataCollectionMeasurements.AddDataCollectionMeasurement(0)
        measure.SetAttValue("DataCollectionPoints", ",".join(points))
        comparison["entries"][int(vehicle_input.AttValue("No"))] = (
            int(link.AttValue("No")), sum(volumes) / len(volumes) * period / 3600, measure)
    evaluation = net.Evaluation
    for key, value in (("DataCollCollectData", True), ("DataCollFromTime", 0),
                       ("DataCollToTime", period), ("DataCollInterval", period)):
        evaluation.SetAttValue(key, value)


def read_comparison(session, comparison: dict) -> None:
    from rt_vissim.calibrate import read_counters  # noqa: PLC0415

    comparison["simulated"] = read_counters(comparison["measures"])
    comparison["entered"] = [
        (no, link, want, float(m.AttValue("Vehs(Current, Last, All)") or 0))
        for no, (link, want, m) in sorted(comparison["entries"].items())]


def print_comparison(comparison: dict, net: Path) -> None:
    import pandas as pd  # noqa: PLC0415

    from rt_vissim.calibrate import geh  # noqa: PLC0415
    from rt_vissim.verify import compare_counts, format_report  # noqa: PLC0415

    counted, simulated = comparison["counted"], comparison.get("simulated", {})
    print("  :Ingested against simulated:")
    for line in format_report(compare_counts(counted, simulated, comparison["feeds"]),
                              comparison.get("entered")):
        print(f"  :{line}")
    out = net.with_name(f"{net.stem}_ingested_vs_simulated.csv")
    pd.DataFrame([{"from": a, "to": b, "feed": comparison["feeds"].get(a),
                   "ingested": c, "simulated": simulated.get((a, b), 0.0),
                   "GEH": round(geh(simulated.get((a, b), 0.0), c), 2)}
                  for (a, b), c in sorted(counted.items())]).to_csv(out, index=False)
    print(f"  :Wrote {out.name}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--inpx",
                        default="vissim/work/mynet/chatt_demand_signals.inpx",
                        help="The model whose error files should be read")
    parser.add_argument("--run", type=int, default=0, metavar="SECONDS",
                        help="Simulate for this many seconds first, discarding "
                             "any earlier error files")
    parser.add_argument("--progid", default=None, help="Pin a Vissim COM ProgID")
    parser.add_argument("--fail-over", type=int, default=None, metavar="N",
                        help="Exit non-zero if more than N vehicles were removed")
    parser.add_argument("--compare-counts", action="store_true",
                        help="Simulate the count period and compare what was ingested "
                             "with what the run delivered (needs --matchup and --links)")
    parser.add_argument("--matchup", default=None, help="The MatchupTable stage 3 used")
    parser.add_argument("--links", default=None, help="Link table written by stage 1")
    parser.add_argument("--traffic-dir", default=None, help="Directory of GridSmart files")
    parser.add_argument("--start", default="08:00", help="Count period start, HH:MM")
    parser.add_argument("--end", default="09:00", help="Count period end, HH:MM")
    args = parser.parse_args(argv)

    net = Path(args.inpx).resolve()
    if not net.exists():
        print(f"  :Network not found: {net}")
        return 1

    released = None
    comparison = None
    if args.compare_counts:
        from rt_vissim.dataset import data_dir  # noqa: PLC0415
        args.traffic_dir = data_dir("traffic", args.traffic_dir, net.parent)
        missing = [f"--{n}" for n in ("matchup", "links") if not getattr(args, n)]
        if missing:
            print(f"  :--compare-counts needs {', '.join(missing)}")
            return 1
        comparison = prepare_comparison(args)
        if not args.run:
            args.run = comparison["period"]
    if args.run:
        from rt_vissim.com import VissimSession  # noqa: PLC0415 - needs a licence

        for stale in net.parent.glob(f"{net.stem}*.err"):
            stale.unlink()
        with VissimSession(args.progid, visible=False) as session:
            session.load_net(net)
            simulation = session.net.Simulation
            simulation.SetAttValue("SimPeriod", args.run)
            simulation.SetAttValue("SimBreakAt", args.run)
            if comparison:
                place_comparison_counters(session, comparison, args.run)
            print(f"  :Running {net.name} for {args.run}s ...")
            simulation.RunContinuous()
            if comparison:
                read_comparison(session, comparison)

    errors = sorted(net.parent.glob(f"{net.stem}*.err"))
    if not errors:
        print(f"  :No error file beside {net.name}. Run the simulation first, "
              "in the GUI or with --run.")
        return 0

    print(f"  :Reading {len(errors)} error file(s) for {net.stem}")
    found = summarise(read_messages(errors))
    lost = report(found, released)

    if comparison:
        print_comparison(comparison, net)

    if args.fail_over is not None and lost > args.fail_over:
        print(f"  :{lost} removed, more than the {args.fail_over} allowed.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
