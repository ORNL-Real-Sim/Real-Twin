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
"""Stage 6: calibrate the inflows and turn splits nobody measured.

Junctions without a camera give the pipeline nothing to write, so stage 3 left
their entry links at zero and split their approaches evenly.  This searches
for the inflows and splits that make the *counted* movements downstream come
out right, scored by GEH -- the method of RealTwin's SUMO ``TurnInflowCali``,
over the same variables:

* an **inflow** for each vehicle input stage 3 wrote at zero, and
* a **split** for each routing decision on an approach with no counts,

restricted to junctions the MatchupTable marks ``Need calibration? = Y``, as
SUMO does.  Nothing is listed by hand; the knobs are read off the network.

The search is one of SUMO's three mealpy algorithms -- genetic algorithm
(``ga``, the default), simulated annealing (``sa``) or tabu search (``ts``) --
with its settings from ``realtwin_config.yaml``.  They cost very different
numbers of simulations at the same epoch, so ``--budget`` sets each one's epoch
to spend about the same, for a fair comparison.

One Vissim session serves every evaluation, with the random seed fixed so the
search sees the parameters rather than the dice.  Each evaluation is a full
simulation of the demand period -- about a minute on Chattanooga -- and is
logged as it finishes, so a long run can be watched and a stopped one still
says what it found.

Writes ``<name>_calibrated_<algo>.inpx`` with the best demand found, a CSV of
every evaluation, and a JSON summary of the values and the before/after scores.

Usage::

    python vissim/scripts/06_calibrate_turn_inflow.py --dry-run
    python vissim/scripts/06_calibrate_turn_inflow.py --epoch 1
    python vissim/scripts/06_calibrate_turn_inflow.py --algo sa --budget 110
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(1, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402

from rt_vissim.calibrate import (ALGORITHMS, MAX_INFLOW, OBJECTIVES,  # noqa: E402
                                 TARGET_GEH,
                                 TARGET_SHARE, Evaluator, apply_solution,
                                 build_problem, counting_sites, epoch_for, make_optimiser,
                                 plot_comparison, plot_history, simulations)
from rt_vissim.demand import build_turn_counts  # noqa: E402
from rt_vissim.matchup import MatchupTable  # noqa: E402
from rt_vissim.network import read_links_csv  # noqa: E402


def clock_to_seconds(text: str) -> int:
    hours, minutes = text.split(":")
    return int(hours) * 3600 + int(minutes) * 60


def config_defaults(config_path: Path) -> dict:
    """Algorithm settings from ``realtwin_config.yaml``, so both pipelines agree.

    Returns ``{"ga": {...}, "sa": {...}, "ts": {...}, "max_inflow": float}``,
    from the ``ga_config``, ``sa_config`` and ``ts_config`` blocks SUMO reads.
    """
    settings = {"ga": {"model_selection": "BaseGA", "epoch": 10, "pop_size": 10,
                       "pc": 0.75, "pm": 0.1},
                "sa": {"model_selection": "OriginalSA", "epoch": 10, "temp_init": 100,
                       "cooling_rate": 0.891, "step_size": 0.1, "scale": 0.1},
                "ts": {"epoch": 10, "tabu_size": 10, "neighbour_size": 10,
                       "perturbation_scale": 0.05},
                "max_inflow": MAX_INFLOW}
    try:
        import yaml  # noqa: PLC0415
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        calibration = config.get("Calibration", {}) or {}
        for algo in ALGORITHMS:
            settings[algo].update(calibration.get(f"{algo}_config", {}) or {})
        # The same key SUMO's turn and inflow calibration reads, in the same
        # unit -- vehicles per hour -- so both search the same space.
        inflow = (calibration.get("turn_inflow", {}) or {}).get("max_inflow")
        if inflow is not None:
            settings["max_inflow"] = float(inflow)
    except Exception:  # noqa: BLE001 - no config is fine, the defaults stand
        pass
    return settings


def copy_supply_files(net_path: Path, out_path: Path) -> list[str]:
    """Copy the files the model names relative to its own folder.

    Vissim stores each signal controller's timing file as ``#data#<name>`` --
    "next to the network file".  A model saved into another folder still names
    them that way, and without them Vissim refuses to start the run ("The
    simulation run could not be initialized").
    """
    if net_path.parent == out_path.parent:
        return []
    copied = []
    text = net_path.read_text(encoding="utf-8", errors="ignore")
    for name in sorted(set(re.findall(r'"#data#([^"]+)"', text))):
        source = net_path.parent / name
        if source.exists():
            shutil.copy2(source, out_path.parent / name)
            copied.append(name)
    return copied


def chart_reference(before: dict | None, history: Path) -> tuple[float, float] | None:
    """The "as built" line, at the level the chart plots.

    The charts plot per approach whenever the history logged it, so the
    reference must be the per-approach score too -- the per-movement one sits
    far lower and makes every run look worse than where it started.
    """
    if not before:
        return None
    header = history.read_text(encoding="utf-8").splitlines()[0].split(",")
    if "approach_mean_geh" in header and "approach_mean" in before:
        return before["approach_mean"], before["approach_share"]
    return before["mean"], before["share"]


def draw_charts(out_path: Path, algo: str, before: dict | None = None) -> list[Path]:
    """Chart a run's progress, and compare it with the other algorithms' runs.

    ``<name>_progress.png`` plots mean GEH and the GEH<5 share against
    evaluation.  When ``<name>`` ends in the algorithm (as the default output
    name does) and a sibling run of another algorithm sits beside it,
    ``<base>_comparison.png`` puts their best-so-far curves together.
    """
    stem = out_path.stem
    history = out_path.with_name(f"{stem}_history.csv")
    if not history.exists():
        return []
    if before is None:
        summary = out_path.with_name(f"{stem}_summary.json")
        if summary.exists():
            before = json.loads(summary.read_text(encoding="utf-8")).get("before")
    reference = chart_reference(before, history)
    written = [plot_history(history, out_path.with_name(f"{stem}_progress.png"),
                            f"{algo.upper()} -- {stem}", reference)]
    if stem.endswith(f"_{algo}"):
        base = stem[: -len(algo) - 1]
        runs = {a: out_path.with_name(f"{base}_{a}_history.csv") for a in ALGORITHMS}
        runs = {a: path for a, path in runs.items() if path.exists()}
        if len(runs) > 1:
            written.append(plot_comparison(runs, out_path.with_name(f"{base}_comparison.png"),
                                           f"GA, SA and TS -- {base}", reference))
    return written


def approach_table(before: dict, after: dict, counts: pd.DataFrame) -> pd.DataFrame:
    """One row per approach, laid out like SUMO's: junction, bound, counted, modelled, GEH."""
    names = (counts.assign(Bound=counts.Turn.str[0])
             .groupby("FromLinkNo_Vissim")[["IntersectionName", "Bound"]].first())
    rows = []
    for (link, want, got_before, geh_before), (_l, _w, got_after, geh_after) in zip(
            before["by_approach"], after["by_approach"]):
        rows.append({"IntersectionName": names.IntersectionName.get(link, "?"),
                     "Bound": names.Bound.get(link, "?"), "approach_link": link,
                     "counted": round(want), "modelled_before": round(got_before),
                     "GEH_before": round(geh_before, 2), "modelled_after": round(got_after),
                     "GEH_after": round(geh_after, 2)})
    return pd.DataFrame(rows).sort_values(["IntersectionName", "Bound"])


def report(label: str, result: dict) -> None:
    """The target is judged per approach, as SUMO judges it; movements are shown too."""
    met = "meets" if result["approach_share"] >= TARGET_SHARE else "does not meet"
    print(f"  :{label}: per approach (SUMO's test): GEH<{TARGET_GEH:.0f} on "
          f"{result['approach_share']:.1%} of {result['approaches']} approaches, "
          f"mean GEH {result['approach_mean']:.2f} -- {met} the {TARGET_SHARE:.0%} target")
    print(f"  :{' ' * len(label)}  per movement: GEH<{TARGET_GEH:.0f} on {result['share']:.1%} of "
          f"{result['matched']} movements, mean GEH {result['mean']:.2f}; modelled "
          f"{result['modelled_total']:.0f} of {result['counted_total']:.0f} counted")


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[2]
    work = "vissim/work/chattanooga"
    defaults = config_defaults(root / "realtwin_config.yaml")

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--inpx", default=f"{work}/chatt_demand_signals.inpx",
                        help="The model stage 4 wrote")
    parser.add_argument("--matchup", default=f"{work}/MatchupTable.xlsx")
    parser.add_argument("--links", default=f"{work}/chatt_links.csv")
    parser.add_argument("--movements", default=f"{work}/chatt_movements.csv")
    parser.add_argument("--traffic-dir", default="datasets/chattanooga/Traffic")
    parser.add_argument("--start", default="08:00", help="Demand start, HH:MM")
    parser.add_argument("--end", default="09:00", help="Demand end, HH:MM")
    parser.add_argument("--algo", choices=ALGORITHMS, default="ga",
                        help="Genetic algorithm, simulated annealing or tabu search, "
                             "as SUMO's calibration offers")
    parser.add_argument("--epoch", type=int, default=None,
                        help="Override the algorithm's epoch from the config")
    parser.add_argument("--pop-size", type=int, default=None,
                        help="Override the population size (GA needs at least 10)")
    parser.add_argument("--objective", choices=OBJECTIVES, default="approach",
                        help="What the search minimises: mean GEH per approach, as "
                             "SUMO's calibration does (default), or per movement")
    parser.add_argument("--budget", type=int, default=None,
                        help="Set the epoch to spend about this many simulations, "
                             "to compare algorithms on equal terms")
    parser.add_argument("--max-inflow", type=float, default=defaults["max_inflow"],
                        help="Upper bound on an unmeasured inflow, vehicles per hour "
                             "(default: max_inflow in realtwin_config.yaml, as SUMO)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Vissim random seed, fixed for every evaluation")
    parser.add_argument("--ga-seed", type=int, default=812,
                        help="The optimiser's own seed (any algorithm), as in SUMO")
    parser.add_argument("--out", default=None,
                        help="Output .inpx (default: <name>_calibrated_<algo>.inpx)")
    parser.add_argument("--progid", default=None, help="Pin a Vissim COM ProgID")
    parser.add_argument("--dry-run", action="store_true",
                        help="List the variables and stop; needs no licence")
    parser.add_argument("--plot-only", action="store_true",
                        help="Redraw the charts of a finished run (--out, --algo) "
                             "and stop; needs no licence")
    args = parser.parse_args(argv)

    if args.plot_only:
        net = Path(args.inpx).resolve()
        out_path = (Path(args.out).resolve() if args.out
                    else net.with_name(f"{net.stem}_calibrated_{args.algo}.inpx"))
        written = draw_charts(out_path, args.algo)
        print(f"  :Wrote {', '.join(p.name for p in written)}" if written
              else f"  :No history beside {out_path.name}")
        return 0 if written else 1

    net_path = Path(args.inpx).resolve()
    for label, path in (("Network", net_path), ("MatchupTable", Path(args.matchup)),
                        ("Link table", Path(args.links)),
                        ("Movement table", Path(args.movements))):
        if not path.exists():
            print(f"  :{label} not found: {path}")
            return 1

    start, end = clock_to_seconds(args.start), clock_to_seconds(args.end)
    try:
        matchup = MatchupTable(args.matchup)
    except PermissionError:
        print(f"  :{args.matchup} is open in another program; close it and re-run.")
        return 1

    counts = build_turn_counts(matchup, Path(args.traffic_dir))
    counts = counts[(counts.IntervalStart >= start) & (counts.IntervalEnd <= end)]
    grouped = counts.groupby(["FromLinkNo_Vissim", "ToLinkNo_Vissim"]).Count.sum()
    counted = {(int(a), int(b)): float(v) for (a, b), v in grouped.items()}
    counted_approaches = {int(a) for a, _b in counted}

    table = matchup.df.copy()
    table["J"] = table["JunctionID_OpenDrive"].ffill().astype(str).str.removesuffix(".0")
    flagged = set(table.loc[table["Need calibration?"].astype(str).str.upper() == "Y", "J"])

    links = read_links_csv(args.links)
    problem, notes = build_problem(net_path, links,
                                   pd.read_csv(args.movements), counted_approaches,
                                   counted, args.max_inflow, flagged)
    sites, site_notes = counting_sites(links, counted)
    notes += site_notes
    print(f"  :{len(counted)} counted movements, {counts.Count.sum():.0f} vehicles, "
          f"{args.start}-{args.end}")
    print(f"  :Junctions marked for calibration: {', '.join(sorted(flagged, key=int))}")
    print(f"  :Counted by data collection: {len(sites)} movements over "
          f"{sum(len(s) for s in sites.values())} junction links, every lane")
    print(f"  :{problem.size} variables -- {len(problem.inflows)} inflows, "
          f"{len(problem.splits)} splits:")
    for line in problem.describe():
        print(f"  :   {line}")
    for note in notes:
        print(f"  :NOTE: {note}")
    if not problem.size:
        print("  :Nothing to calibrate.")
        return 0

    settings = dict(defaults[args.algo])
    if args.pop_size is not None:
        settings["pop_size"] = args.pop_size
    if args.epoch is not None:
        settings["epoch"] = args.epoch
    elif args.budget is not None:
        settings["epoch"] = epoch_for(args.algo, settings, args.budget)
    shown = ", ".join(f"{k} {v}" for k, v in settings.items())
    print(f"  :{args.algo.upper()}: {shown} -- about "
          f"{simulations(args.algo, settings)} simulations")
    print(f"  :Bounds: inflows 0-{args.max_inflow:.0f} veh/h, splits 0-1")
    if args.dry_run:
        return 0

    from mealpy import FloatVar  # noqa: PLC0415

    from rt_vissim.com import VissimSession  # noqa: PLC0415

    out_path = (Path(args.out).resolve() if args.out
                else net_path.with_name(f"{net_path.stem}_calibrated_{args.algo}.inpx"))
    history = out_path.with_name(f"{out_path.stem}_history.csv")
    summary = out_path.with_name(f"{out_path.stem}_summary.json")
    period = end - start
    began = time.time()

    with VissimSession(args.progid, visible=False) as session:
        session.load_net(net_path)
        evaluator = Evaluator(session, problem, period, sites, seed=args.seed,
                              history=history, objective=args.objective)

        print("  :Scoring the model as it stands ...", flush=True)
        before = evaluator.run(None)
        report("Before", before)

        lower, upper = problem.bounds()
        optimiser = make_optimiser(args.algo, settings)
        optimiser.solve({"obj_func": evaluator, "minmax": "min", "log_to": None,
                         "bounds": FloatVar(lb=lower, ub=upper)},
                        termination={"max_epoch": settings["epoch"]}, seed=args.ga_seed)

        best_mean, best_solution, _ = evaluator.best
        print("  :Writing the best demand found and scoring it once more ...",
              flush=True)
        apply_solution(session, problem, best_solution)
        after = evaluator.run(None)
        session.save_net_as(out_path)
    copied = copy_supply_files(net_path, out_path)
    if copied:
        print(f"  :Copied beside it, as the model needs them: {', '.join(copied)}")

    report("Before", before)
    report("After ", after)
    approaches = approach_table(before, after, counts)
    approaches.to_csv(out_path.with_name(f"{out_path.stem}_approaches.csv"), index=False)
    print("  :Per approach, as SUMO scores it (vehicles in the hour):")
    for line in approaches.to_string(index=False).splitlines():
        print(f"  :   {line}")
    print(f"  :{evaluator.count} simulations in {(time.time() - began) / 60:.0f} min")
    values = problem.decode(best_solution)
    for label, value in values:
        print(f"  :   {label}: {value}")

    # The ceiling is SUMO's default rather than anything measured here, so say
    # when it decided the answer: an inflow that finishes at the bound is one
    # the search wanted to push higher.
    pinned = [label for (label, value), knob in zip(values, problem.inflows)
              if isinstance(value, float) and value >= 0.95 * args.max_inflow]
    if pinned:
        print(f"  :WARNING: {len(pinned)} inflows finished at the "
              f"{args.max_inflow:.0f} veh/h ceiling, so the bound -- not the "
              "counts -- set them. Re-run with a higher --max-inflow:")
        for label in pinned:
            print(f"  :   {label}")

    summary.write_text(json.dumps({
        "network": net_path.name, "calibrated": out_path.name,
        "period": [args.start, args.end],
        "algorithm": args.algo, "objective": args.objective, "settings": settings,
        "optimiser_seed": args.ga_seed,
        "vissim_seed": args.seed, "evaluations": evaluator.count,
        "before": {k: before[k] for k in ("mean", "share", "matched",
                                          "modelled_total", "counted_total",
                   "approach_mean", "approach_share", "approaches")},
        "after": {k: after[k] for k in ("mean", "share", "matched",
                                        "modelled_total", "counted_total",
                   "approach_mean", "approach_share", "approaches")},
        "values": [{"knob": label, "value": value} for label, value in values],
        "solution": problem.physical(best_solution),
    }, indent=2, default=str), encoding="utf-8")
    print(f"  :Wrote {out_path.name}, {history.name}, {summary.name}")
    try:
        charts = draw_charts(out_path, args.algo, before)
        print(f"  :Charts: {', '.join(p.name for p in charts)}")
    except Exception as exc:  # noqa: BLE001 - a chart must never cost a finished run
        print(f"  :Charts not drawn ({exc}); re-run with --plot-only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
