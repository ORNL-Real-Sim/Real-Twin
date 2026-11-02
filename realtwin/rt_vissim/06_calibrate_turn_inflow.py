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
"""Stage 6: calibrate Vissim demand or driving behavior with shared optimizers.

Junctions without a camera give the pipeline nothing to write, so stage 3 left
their entry links at zero and split their approaches evenly.  This searches
for the inflows and splits that make the *counted* movements downstream come
out right, scored by GEH -- the method of RealTwin's SUMO ``TurnInflowCali``,
over the same variables:

* an **inflow** for each vehicle input stage 3 wrote at zero, and
* a **split** for each routing decision on an approach with no counts,

restricted to junctions the MatchupTable marks ``Need calibration? = Y``, as
SUMO does.  Nothing is listed by hand; the knobs are read off the network.

The search uses the package's genetic algorithm
(``ga``, the default), simulated annealing (``sa``), tabu search (``ts``), or
shared Bayesian optimization (``bo``) --
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

    python -m realtwin.rt_vissim.06_calibrate_turn_inflow --help
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import time
from pathlib import Path

import pandas as pd

from realtwin.rt_vissim.calibrate import (ALGORITHMS, MAX_INFLOW, OBJECTIVES,
                                 TARGET_GEH,
                                 TARGET_SHARE, Evaluator,
                                 build_problem, calibration_termination, counting_sites, epoch_for, make_optimiser,
                                 plot_comparison, plot_history, simulations)
from realtwin.rt_vissim.calibrate_vissim import cali_vissim, validate_bo_settings
from realtwin.rt_vissim.cali_behavior import (
    BehaviorEvaluator, export_behavior_info, resolve_behavior_config)
from realtwin.rt_vissim.demand import build_turn_counts
from realtwin.rt_vissim.matchup import MatchupTable
from realtwin.rt_vissim.network import read_links_csv


def clock_to_seconds(text: str) -> int:
    """Accept a clock time or whole seconds after midnight without truncation."""
    text = str(text)
    if ":" in text:
        parts = [int(part) for part in text.split(":")]
        if len(parts) not in (2, 3) or not 0 <= parts[1] < 60:
            raise ValueError(f"Invalid clock time: {text!r}")
        seconds = parts[2] if len(parts) == 3 else 0
        if not 0 <= seconds < 60:
            raise ValueError(f"Invalid clock time: {text!r}")
        value = float(parts[0] * 3600 + parts[1] * 60 + seconds)
    else:
        value = float(text)
    if not math.isfinite(value) or not value.is_integer() or not 0 <= value <= 86400:
        raise ValueError(f"Time must be whole seconds between 0 and 86400: {text!r}")
    return int(value)


def load_config(config_path: Path | None) -> dict:
    """Read an explicit YAML/JSON configuration; invalid files must fail visibly."""
    if config_path is None:
        return {}
    import yaml

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Calibration configuration must be a mapping: {config_path}")
    return config


def config_defaults(config_path: Path | dict | None = None) -> dict:
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
                "bo": {"kernel_type": "RBF", "tolerance": 3, "target": 0,
                       "random_points": 4000, "max_evaluations": 100,
                       "total_run": 1, "seed": 812},
                "max_inflow": MAX_INFLOW}
    config = config_path if isinstance(config_path, dict) else load_config(config_path)
    calibration = config.get("Calibration", {}) or {}
    for algo in ALGORITHMS:
        settings[algo].update(calibration.get(f"{algo}_config", {}) or {})
    inflow = (calibration.get("turn_inflow", {}) or {}).get("max_inflow")
    if inflow is not None:
        settings["max_inflow"] = float(inflow)
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
    target_share = TARGET_SHARE
    summary = out_path.with_name(f"{stem}_summary.json")
    if summary.exists():
        saved = json.loads(summary.read_text(encoding="utf-8"))
        if before is None:
            before = saved.get("before")
        target_share = saved.get("calibration_target", {}).get("GEHPercent", TARGET_SHARE)
    reference = chart_reference(before, history)
    written = [plot_history(history, out_path.with_name(f"{stem}_progress.png"),
                            f"{algo.upper()} -- {stem}", reference, target_share=target_share)]
    if stem.endswith(f"_{algo}"):
        base = stem[: -len(algo) - 1]
        runs = {a: out_path.with_name(f"{base}_{a}_history.csv") for a in ALGORITHMS}
        runs = {a: path for a, path in runs.items() if path.exists()}
        if len(runs) > 1:
            written.append(plot_comparison(runs, out_path.with_name(f"{base}_comparison.png"),
                                           f"Calibration comparison -- {base}", reference,
                                           target_share=target_share))
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


def report(label: str, result: dict, target_geh: float = TARGET_GEH,
           target_share: float = TARGET_SHARE) -> None:
    """The target is judged per approach, as SUMO judges it; movements are shown too."""
    met = "meets" if result["approach_share"] >= target_share else "does not meet"
    print(f"  :{label}: per approach (SUMO's test): GEH<{target_geh:g} on "
          f"{result['approach_share']:.1%} of {result['approaches']} approaches, "
          f"mean GEH {result['approach_mean']:.2f} -- {met} the {target_share:.0%} target")
    print(f"  :{' ' * len(label)}  per movement: GEH<{target_geh:g} on {result['share']:.1%} of "
          f"{result['matched']} movements, mean GEH {result['mean']:.2f}; modelled "
          f"{result['modelled_total']:.0f} of {result['counted_total']:.0f} counted")


def algorithm_settings(args, defaults: dict, parser: argparse.ArgumentParser,
                       calibration: dict) -> dict:
    """Apply CLI overrides and validate settings before opening Vissim."""
    settings = dict(defaults[args.algo])
    if args.algo == "bo" and (args.epoch is not None or args.pop_size is not None):
        parser.error("BO uses bo_config.max_evaluations and total_run, rather than --epoch/--pop-size")
    if args.pop_size is not None:
        settings["pop_size"] = args.pop_size
    if args.epoch is not None:
        settings["epoch"] = args.epoch
    elif args.budget is not None:
        if args.budget < 1:
            parser.error("budget must be positive")
        if args.algo == "bo":
            runs = settings["total_run"]
            if not isinstance(runs, int) or isinstance(runs, bool) or runs < 1:
                parser.error("bo_config.total_run must be a positive integer")
            if args.budget < runs:
                parser.error("BO budget must allow at least one evaluation per run")
            settings["max_evaluations"] = args.budget // runs
        else:
            settings["epoch"] = epoch_for(args.algo, settings, args.budget)
    if args.algo == "bo":
        settings["seed"] = args.ga_seed
        validate_bo_settings(settings)
    else:
        make_optimiser(args.algo, settings)
        calibration_termination(calibration, args.stage, settings)
    return settings


def run_behavior_stage(args, calibration: dict, defaults: dict, net_path: Path,
                       start: int, end: int, resolution: int,
                       parser: argparse.ArgumentParser) -> int:
    """Calibrate native driving behavior and save parameters and route scores."""
    from realtwin.rt_vissim.com import VissimSession

    settings = algorithm_settings(args, defaults, parser, calibration)
    options = calibration.get("behavior", {})
    stage_config = {**calibration, f"{args.algo}_config": settings,
                    "behavior": {**options, "optimizer_seed": args.ga_seed}}
    effective = {"VISSIM": {"behavior": stage_config}}
    info = export_behavior_info(effective, net_path)
    print(f"  :{args.algo.upper()} behavior: {info['problem'].size} native parameters, "
          f"{len(info['routes'])} travel-time routes; bounds {info['problem'].ranges}")
    if args.dry_run:
        return 0
    out_path = (Path(args.out).resolve() if args.out
                else net_path.with_name(f"{net_path.stem}_behavior_{args.algo}.inpx"))
    if out_path == net_path:
        parser.error("--out must differ from the input network; calibration preserves the source")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    history = out_path.with_name(f"{out_path.stem}_history.csv")
    with VissimSession(args.progid, visible=False) as session:
        session.load_net(net_path)
        evaluator = BehaviorEvaluator(session, info, start=start, end=end,
                                      seed=args.seed, resolution=resolution, history=history)
        before = evaluator.run()
        model, _, after = cali_vissim(input_config=effective, evaluator=evaluator,
                                     sel_algo={"behavior": args.algo}, stage="behavior")
        if after["missing_routes"]:
            raise RuntimeError("Best behavior candidate has routes without completed vehicles; "
                               "check demand, route links and the simulation period")
        session.vissim.Simulation.Stop()
        session.save_net_as(out_path)
    copy_supply_files(net_path, out_path)
    best_solution = evaluator.best[1]
    values = dict(zip(info["problem"].names, info["problem"].physical(best_solution)))
    summary = {"network": str(net_path), "calibrated": str(out_path), "stage": "behavior",
               "algorithm": args.algo, "objective": "travel_time_mae_seconds", "settings": settings,
               "period": [start, end], "vissim_seed": args.seed, "optimiser_seed": args.ga_seed,
               "resolution": resolution, "evaluations": evaluator.count,
               "route_positions": evaluator.route_positions,
               "reference_source": info["reference_source"],
               "driving_behavior_ids": info["problem"].driving_behavior_ids,
               "params_ranges": options["params_ranges"], "values": values,
               "solution": list(values.values()), "before": before, "after": after,
               "missing_route_penalty": info["missing_route_penalty"]}
    if args.algo != "bo":
        summary["termination"] = calibration_termination(stage_config, "behavior", settings)
    out_path.with_name(f"{out_path.stem}_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(after["routes"]).to_csv(
        out_path.with_name(f"{out_path.stem}_routes.csv"), index=False)
    if args.algo == "bo":
        model.run_vis(str(out_path.with_name(f"{out_path.stem}_bo_result")))
    if args.result:
        args.result.write_text(json.dumps({"network": str(out_path)}), encoding="utf-8")
    print(f"  :Behavior MAE: {before['mae_seconds']:.2f} -> {after['mae_seconds']:.2f} s; "
          f"wrote {out_path.name}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, help="Effective calibration settings, YAML or JSON")
    parser.add_argument("--stage", choices=("turn_inflow", "behavior"), default="turn_inflow",
                        help="Independent calibration stage (default: turn_inflow)")
    parser.add_argument("--result", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--inpx",
                        help="The model stage 4 wrote")
    parser.add_argument("--matchup")
    parser.add_argument("--links")
    parser.add_argument("--movements")
    parser.add_argument("--traffic-dir")
    parser.add_argument("--start", help="Demand start, HH:MM[:SS] or seconds after midnight")
    parser.add_argument("--end", help="Demand end, HH:MM[:SS] or seconds after midnight")
    parser.add_argument("--algo", choices=ALGORITHMS, default="ga",
                        help="GA (default), SA, TS or shared Bayesian optimization")
    parser.add_argument("--epoch", type=int, default=None,
                        help="Override the algorithm's epoch from the config")
    parser.add_argument("--pop-size", type=int, default=None,
                        help="Override the population size (GA needs at least 10)")
    parser.add_argument("--objective", choices=OBJECTIVES,
                        help="What the search minimises: mean GEH per approach, as "
                             "SUMO's calibration does (default), or per movement")
    parser.add_argument("--budget", type=int, default=None,
                        help="Set the epoch to spend about this many simulations, "
                             "to compare algorithms on equal terms")
    parser.add_argument("--max-inflow", type=float,
                        help="Upper bound on an unmeasured inflow, vehicles per hour "
                             "(default: max_inflow in realtwin_config.yaml, as SUMO)")
    parser.add_argument("--seed", type=int,
                        help="Vissim random seed, fixed for every evaluation")
    parser.add_argument("--ga-seed", type=int,
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

    config = load_config(args.config)
    defaults = config_defaults(config)
    calibration = config.get("Calibration", {}) or {}
    if args.stage == "behavior":
        calibration = resolve_behavior_config(calibration)
    scenario = calibration.get("scenario_config", {}) or {}
    turn_inflow = calibration.get("turn_inflow", {}) or {}
    stage_options = calibration.get(args.stage, {}) or {}
    # Explicit CLI options take precedence over the live configuration snapshot.
    for name, fallback in (("start", scenario.get("sim_start_time", 28800)),
                           ("end", scenario.get("sim_end_time", 32400)),
                           ("max_inflow", defaults["max_inflow"]),
                           ("objective", turn_inflow.get("objective", "approach")),
                           ("seed", stage_options.get("seed", 42)),
                           ("ga_seed", defaults["bo"]["seed"] if args.algo == "bo"
                            else stage_options.get("optimizer_seed", 812)),
                           ("budget", stage_options.get("budget")),
                           ("out", stage_options.get("out")),
                           ("progid", (config.get("VISSIM") or {}).get("progid"))):
        if getattr(args, name) is None:
            setattr(args, name, fallback)
    if not args.inpx and not (args.plot_only and args.out):
        parser.error("--inpx must name the prepared demand/signals network")

    if args.plot_only:
        if args.stage == "behavior":
            parser.error("--plot-only currently plots turn/inflow GEH history")
        net = Path(args.inpx).resolve() if args.inpx else Path(args.out).resolve()
        out_path = (Path(args.out).resolve() if args.out
                    else net.with_name(f"{net.stem}_calibrated_{args.algo}.inpx"))
        written = draw_charts(out_path, args.algo)
        print(f"  :Wrote {', '.join(p.name for p in written)}" if written
              else f"  :No history beside {out_path.name}")
        return 0 if written else 1

    net_path = Path(args.inpx).resolve()
    if not net_path.is_file():
        print(f"  :Network not found: {net_path}")
        return 1

    start, end = clock_to_seconds(args.start), clock_to_seconds(args.end)
    if end <= start:
        parser.error("Demand end must be after demand start")
    target = scenario.get("calibration_target", {}) or {}
    target_geh = float(target.get("GEH", TARGET_GEH))
    target_share = float(target.get("GEHPercent", TARGET_SHARE))
    if not math.isfinite(target_geh) or target_geh <= 0 or not 0 <= target_share <= 1:
        parser.error("calibration_target requires finite GEH > 0 and GEHPercent between 0 and 1")
    if not math.isfinite(float(args.max_inflow)) or float(args.max_inflow) <= 0:
        parser.error("max_inflow must be finite and positive (vehicles per hour)")
    resolution = 10
    if "calibration_time_step" in stage_options or "calibration_time_step" in scenario:
        step = float(stage_options.get(
            "calibration_time_step", scenario.get("calibration_time_step")))
        if not math.isfinite(step) or step <= 0:
            parser.error("calibration_time_step must be positive seconds")
        reciprocal = 1 / step
        if not reciprocal.is_integer() or not 1 <= reciprocal <= 20:
            parser.error("calibration_time_step must correspond to 1-20 Vissim steps per second")
        resolution = int(reciprocal)
    if args.stage == "behavior":
        return run_behavior_stage(args, calibration, defaults, net_path,
                                  start, end, resolution, parser)
    for name in ("matchup", "links", "movements", "traffic_dir"):
        if not getattr(args, name):
            parser.error(f"--{name.replace('_', '-')} is required for calibration")
        if not Path(getattr(args, name)).exists():
            print(f"  :{name} not found: {getattr(args, name)}")
            return 1
    if args.objective not in OBJECTIVES:
        parser.error(f"objective must be one of {OBJECTIVES}")
    try:
        matchup = MatchupTable(args.matchup)
    except PermissionError:
        print(f"  :{args.matchup} is open in another program; close it and re-run.")
        return 1

    counts = build_turn_counts(matchup, Path(args.traffic_dir))
    counts = counts[(counts.IntervalStart >= start) & (counts.IntervalEnd <= end)]
    if counts.empty:
        print("  :No observed traffic counts in the configured demand period.")
        return 1
    for movement, observations in counts.groupby(["FromLinkNo_Vissim", "ToLinkNo_Vissim"]):
        intervals = (observations[["IntervalStart", "IntervalEnd"]].drop_duplicates()
                     .sort_values("IntervalStart"))
        starts, ends = intervals.IntervalStart.tolist(), intervals.IntervalEnd.tolist()
        if (starts[0] != start or ends[-1] != end
                or starts[1:] != ends[:-1] or any(last <= first for first, last in zip(starts, ends))):
            print(f"  :Observed counts for movement {movement} must cover the entire "
                  f"{start}-{end} simulation window with contiguous complete intervals.")
            return 1
    grouped = counts.groupby(["FromLinkNo_Vissim", "ToLinkNo_Vissim"]).Count.sum()
    counted = {(int(a), int(b)): float(v) for (a, b), v in grouped.items()}
    counted_approaches = {int(a) for a, _b in counted}

    table = matchup.df.copy()
    table["J"] = table["JunctionID_OpenDrive"].ffill().astype(str).str.removesuffix(".0")
    flagged = set(table.loc[table["Need calibration?"].astype(str).str.upper() == "Y", "J"])

    links = read_links_csv(args.links)
    movements = pd.read_csv(args.movements)
    # CSV link names can omit OpenDRIVE junction metadata. The movement table
    # explicitly records internal links and is authoritative for these IDs.
    for movement in movements.itertuples(index=False):
        for number in re.findall(r"\d+", str(movement.InternalLinks_Vissim)):
            if int(number) in links:
                links[int(number)].junction_key = str(movement.JunctionID_OpenDrive).removesuffix(".0")
    problem, notes = build_problem(net_path, links,
                                   movements, counted_approaches,
                                   counted, args.max_inflow, flagged)
    sites, site_notes = counting_sites(links, counted)
    notes += site_notes
    print(f"  :{len(counted)} counted movements, {counts.Count.sum():.0f} vehicles, "
          f"{args.start}-{args.end}")
    print(f"  :Junctions marked for calibration: {', '.join(sorted(flagged))}")
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
        if args.result:
            args.result.write_text(json.dumps({"network": str(net_path)}), encoding="utf-8")
        return 0
    if set(sites) != set(counted):
        print("  :Cannot score all observed movements. Check the matchup and network paths.")
        return 1

    settings = algorithm_settings(args, defaults, parser, calibration)
    shown = ", ".join(f"{k} {v}" for k, v in settings.items())
    print(f"  :{args.algo.upper()}: {shown} -- about "
          f"{simulations(args.algo, settings)} simulations")
    print(f"  :Bounds: inflows 0-{args.max_inflow:.0f} veh/h, splits 0-1")
    if args.dry_run:
        return 0

    from realtwin.rt_vissim.com import VissimSession  # noqa: PLC0415

    out_path = (Path(args.out).resolve() if args.out
                else net_path.with_name(f"{net_path.stem}_calibrated_{args.algo}.inpx"))
    if out_path == net_path:
        parser.error("--out must differ from the input network; calibration preserves the source")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    history = out_path.with_name(f"{out_path.stem}_history.csv")
    summary = out_path.with_name(f"{out_path.stem}_summary.json")
    period = end - start
    began = time.time()

    with VissimSession(args.progid, visible=False) as session:
        session.load_net(net_path)
        evaluator = Evaluator(session, problem, period, sites, seed=args.seed,
                              resolution=resolution, start_time=start, target_geh=target_geh,
                              history=history, objective=args.objective)

        print("  :Scoring the model as it stands ...", flush=True)
        before = evaluator.run(None)
        report("Before", before, target_geh, target_share)

        stage_config = {**calibration, f"{args.algo}_config": settings,
                        "turn_inflow": {**turn_inflow, "optimizer_seed": args.ga_seed}}
        model, _, after = cali_vissim(
            input_config={"VISSIM": {"turn_inflow": stage_config}},
            evaluator=evaluator, sel_algo={"turn_inflow": args.algo})
        if evaluator.best is None:
            raise RuntimeError("The optimizer did not evaluate any calibration candidates")
        _, best_solution, _ = evaluator.best
        print("  :Writing the best demand found ...", flush=True)
        session.vissim.Simulation.Stop()
        session.save_net_as(out_path)
    copied = copy_supply_files(net_path, out_path)
    if copied:
        print(f"  :Copied beside it, as the model needs them: {', '.join(copied)}")

    report("Before", before, target_geh, target_share)
    report("After ", after, target_geh, target_share)
    approaches = approach_table(before, after, counts)
    approaches.to_csv(out_path.with_name(f"{out_path.stem}_approaches.csv"), index=False)
    print("  :Per approach (vehicles in the configured demand period):")
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
        "termination": (calibration_termination(stage_config, "turn_inflow", settings)
                        if args.algo != "bo" else None),
        "optimiser_seed": args.ga_seed,
        "vissim_seed": args.seed, "evaluations": evaluator.count,
        "resolution": resolution, "max_inflow": args.max_inflow,
        "calibration_target": {"GEH": target_geh, "GEHPercent": target_share},
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
    if args.algo == "bo":
        model.run_vis(str(out_path.with_name(f"{out_path.stem}_bo_result")))
    if args.result:
        args.result.write_text(json.dumps({"network": str(out_path)}), encoding="utf-8")
    try:
        charts = draw_charts(out_path, args.algo, before)
        print(f"  :Charts: {', '.join(p.name for p in charts)}")
    except Exception as exc:  # noqa: BLE001 - a chart must never cost a finished run
        print(f"  :Charts not drawn ({exc}); re-run with --plot-only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
