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
"""Estimate the demand nobody measured, from the demand somebody did.

Not every junction is surveyed.  Where one is not, the pipeline has nothing to
write: its entry links are left at zero volume and its approaches get an equal
share across their exits.  That is a placeholder, and it shows -- on
Chattanooga the model puts 6,543 vehicles through the counted movements against
13,786 counted, because the traffic that should feed them enters through the
junctions nobody measured.

The unmeasured demand can still be inferred, because it has a measured effect:
whatever enters at an unsurveyed junction has to leave through movements that
*were* counted.  So the unknowns are the inflows and turn splits at the
unsurveyed junctions, and the evidence is GEH against the counted movements.
That is the problem RealTwin's SUMO flow solves in ``TurnInflowCali``, and on
Chattanooga the two arrive at the same twelve variables:

===========================  =====  =====
                             SUMO   here
===========================  =====  =====
inflow volumes                   4      4
turn splits                      8      8
===========================  =====  =====

Nothing about the variables is written in by hand.  :func:`build_problem` reads
them off the built network: an inflow is any vehicle input stage 3 left at zero
because no counted approach traces to it, and a split is any routing decision
governing an approach that has no counts.  That keeps it standalone -- a
network nobody has seen before gets the right knobs without anyone listing
them.

A split with ``n`` usable exits needs ``n - 1`` variables, not ``n``: the
shares have to sum to one, so the first takes ``v1``, the next takes
``(1 - v1) * v2``, and the last takes what is left.  With two exits that is the
familiar ``x`` and ``1 - x``.  It is the parameterisation the earlier ORNL
notebooks used (``second_right = (1 - first_right) * x``), and it lets every
variable range over ``[0, 1]`` independently.  U-turns are held at zero
throughout -- the demand stage does not model them, and freeing them lets an
optimiser send traffic round a turnaround to close a gap somewhere else.

Each counted movement is measured by data collection points, one per lane on
the junction link that carries that movement and nothing else
(:func:`counting_sites`, :func:`place_counters`).  The earlier ORNL notebooks
did the same with 68 hand-placed measurements; here they are placed over COM
when the model is opened, from the network's own topology.

Node evaluation looks like the obvious source and is not usable for this:
"If an edge between nodes leads via more than three branchings, it is ignored
during node evaluation" (manual, Evaluating nodes).  Every main-road approach
whose lanes fan out into turn bays before the stop line is such an edge, and
node evaluation reported those approaches -- five of Chattanooga's 22, its
busiest -- as carrying no traffic at all.
"""

from __future__ import annotations

import csv
import math
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

#: RealTwin's target, from ``realtwin_config.yaml``: GEH below 5 on 85% of the
#: movements that have a count to compare against.
TARGET_GEH = 5.0
TARGET_SHARE = 0.85

#: What the search minimises and the target is judged on.  ``"approach"`` is
#: SUMO's: each approach's movements summed, one GEH per approach
#: (``result_analysis_on_EdgeData``).  ``"movement"`` scores every counted turn.
OBJECTIVES = ("approach", "movement")

#: Vehicles per hour, the ceiling on an unmeasured entry.  The same value and
#: unit as SUMO's ``max_inflow`` (``realtwin_config.yaml``, default 200): its
#: ``update_turn_flow_from_solution`` reads the GA's value as vehicles per hour
#: and scales it by 15/60 into each demand interval.
#:
#: It was 2,000 here, ten times SUMO's, on the reasoning that the gap between
#: counted and modelled vehicles had to be made up at these entries.  That gap
#: is summed over movements, and one vehicle passes several counted movements,
#: so it overstated the need badly.  At 2,000 the random candidates put more
#: traffic on four side streets than the main road carries, and the network
#: jammed: throughput through the counted movements fell to half the
#: uncalibrated model's.
MAX_INFLOW = 200.0

#: How often to retry an evaluation when COM refuses a call.  Vissim answers
#: "Call was rejected by callee" now and then when it is busy; it is transient.
COM_RETRIES = 3


def geh(modelled: float, counted: float) -> float:
    """The GEH statistic, which both pipelines use to score a demand.

    Gentler than a percentage on small flows and stricter on large ones, which
    is what a corridor mixing 6 vehicles an hour with 900 needs.

    Args:
        modelled: Vehicles the simulation put through the movement.
        counted: Vehicles the survey counted.

    Returns:
        The statistic; zero when both are zero.
    """
    if modelled + counted == 0:
        return 0.0
    return math.sqrt(2 * (modelled - counted) ** 2 / (modelled + counted))


def stick_shares(values) -> list[float]:
    """Turn ``n - 1`` free values in ``[0, 1]`` into ``n`` shares summing to one.

    ``[x]`` gives ``[x, 1 - x]``; ``[a, b]`` gives ``[a, (1 - a) b, (1 - a)(1 - b)]``.

    Args:
        values: The free values, each clipped to ``[0, 1]``.

    Returns:
        One share per route, in order.
    """
    shares: list[float] = []
    left = 1.0
    for value in values:
        value = min(max(float(value), 0.0), 1.0)
        shares.append(left * value)
        left *= 1.0 - value
    shares.append(left)
    return shares


@dataclass
class InflowKnob:
    """An entry link whose volume nobody measured.

    Attributes:
        input_no: Vissim vehicle input number.
        link_no: The entry link it sits on.
        junction: The junction it feeds, for reporting.
        intervals: How many time intervals the input carries.
    """

    input_no: int
    link_no: int
    junction: str = "?"
    intervals: int = 4

    @property
    def size(self) -> int:
        return 1

    def label(self) -> str:
        return f"inflow  input {self.input_no} on link {self.link_no} (J{self.junction})"


@dataclass
class SplitKnob:
    """An approach whose turn split nobody measured.

    Attributes:
        decision_no: Vissim static routing decision number.
        approach: The approach link whose split this governs.  Not always the
            link the decision sits on -- stage 3 moves a decision upstream when
            the approach is too short, so the decision can be a junction back.
        junction: The unsurveyed junction, for reporting.
        free_routes: Route numbers sharing the traffic, in order.
        exits: The exit link each free route reaches, in the same order.
        zero_routes: U-turn route numbers, pinned at zero.
        intervals: How many time intervals the routes carry.
        lane_routes: For an exit the road reaches from more than one lane,
            ``{exit: ((route, share of the exit's flow), ...)}`` -- stage 3
            writes one route per lane path.  The search still sets one share
            per exit; it is divided among these routes.  Absent means the
            exit has the one route in ``free_routes``.
    """

    decision_no: int
    approach: int
    junction: str = "?"
    free_routes: tuple[int, ...] = ()
    exits: tuple[int, ...] = ()
    zero_routes: tuple[int, ...] = ()
    intervals: int = 4
    lane_routes: dict[int, tuple[tuple[int, float], ...]] = field(default_factory=dict)

    @property
    def size(self) -> int:
        return max(0, len(self.free_routes) - 1)

    def routes_of(self, index: int) -> tuple[tuple[int, float], ...]:
        """The routes carrying exit ``index``'s flow, each with its share of it."""
        exit_no = self.exits[index]
        return self.lane_routes.get(exit_no, ((self.free_routes[index], 1.0),))

    def label(self) -> str:
        return (f"split   decision {self.decision_no} for approach {self.approach} "
                f"(J{self.junction}), exits {self.exits}")


@dataclass
class Problem:
    """The knobs to turn and the counts to turn them against."""

    inflows: list[InflowKnob] = field(default_factory=list)
    splits: list[SplitKnob] = field(default_factory=list)
    counted: dict[tuple[int, int], float] = field(default_factory=dict)
    max_inflow: float = MAX_INFLOW

    @property
    def size(self) -> int:
        """How many variables the optimiser searches over."""
        return sum(k.size for k in self.inflows) + sum(k.size for k in self.splits)

    def bounds(self) -> tuple[list[float], list[float]]:
        """Every variable runs over ``[0, 1]``, inflows first then splits.

        An inflow is searched as a fraction of :attr:`max_inflow` and scaled to
        vehicles per hour only when written (:meth:`physical`).  The space is the
        same either way, but the optimisers' step sizes are absolute: SA's
        ``step_size: 0.1`` and TS's ``perturbation_scale: 0.05`` would move a
        split by a tenth and an inflow by a tenth of a vehicle an hour.  On a
        common scale they move every variable by the same fraction of its range.
        GA draws within the bounds, so its search is unchanged by this.
        """
        return [0.0] * self.size, [1.0] * self.size

    def physical(self, solution) -> list[float]:
        """The optimiser's values with inflows in vehicles per hour."""
        values = [float(v) for v in solution]
        for i in range(len(self.inflows)):
            values[i] = min(max(values[i], 0.0), 1.0) * self.max_inflow
        return values

    def describe(self) -> list[str]:
        """One line per knob, so a solution can be read back."""
        return [k.label() for k in self.inflows] + [k.label() for k in self.splits]

    def decode(self, solution) -> list[tuple[str, object]]:
        """Pair each knob with the value a solution gives it, in readable form."""
        solution = self.physical(solution)
        out: list[tuple[str, object]] = []
        cursor = 0
        for knob in self.inflows:
            out.append((knob.label(), round(float(solution[cursor]), 1)))
            cursor += 1
        for knob in self.splits:
            values = solution[cursor:cursor + knob.size]
            cursor += knob.size
            shares = stick_shares(values)
            out.append((knob.label(),
                        {exit_no: round(share, 3)
                         for exit_no, share in zip(knob.exits, shares)}))
        return out


# --------------------------------------------------------------------------- #
# Finding the knobs
# --------------------------------------------------------------------------- #
def _route_sequence(route) -> list[int]:
    seq = route.find("linkSeq")
    if seq is None:
        return []
    out = []
    for ref in seq:
        key = ref.get("key")
        if key is not None:
            try:
                out.append(int(key))
            except ValueError:
                continue
    return out


def _first_rel_flow(route) -> float:
    """A route's relative flow in its first interval, from ``relFlow="2 0:50.5, ..."``."""
    text = route.get("relFlow") or ""
    try:
        return float(text.split(",")[0].split(":")[1])
    except (IndexError, ValueError):
        return 0.0


def _lane_shares(routes: list[tuple[int, float]]) -> tuple[tuple[int, float], ...]:
    """Each lane route's share of its exit, from the flows stage 3 wrote (by lanes)."""
    total = sum(flow for _r, flow in routes)
    if total <= 0:
        return tuple((r, 1.0 / len(routes)) for r, _f in routes)
    return tuple((r, flow / total) for r, flow in routes)


def build_problem(inpx: str | Path, links: dict, movements, counted_approaches: set[int],
                  counted: dict[tuple[int, int], float] | None = None,
                  max_inflow: float = MAX_INFLOW,
                  needs_calibration: set[str] | None = None) -> tuple[Problem, list[str]]:
    """Read the knobs off a built network.

    Nothing is listed by hand.  An inflow knob is any vehicle input stage 3 left
    at zero in every interval -- it does that when no counted approach traces
    to the entry.  A split knob is any static routing decision whose routes
    pass through an approach with no counts.  The ``.inpx`` is read as XML, so
    this needs no Vissim licence.

    ``needs_calibration`` is the MatchupTable's ``Need calibration?`` column,
    which SUMO's ``generate_inflow`` and ``generate_turn_summary`` also gate on:
    a knob is kept only at a junction marked Y.  Stage 2 marks exactly the
    junctions with no counts, so by default this changes nothing -- but setting
    a junction to N by hand takes it out of the search.

    Args:
        inpx: The network stage 4 wrote.
        links: Output of :func:`rt_vissim.network.read_links_csv`.
        movements: The stage 1 movement table, for junctions and turn types.
        counted_approaches: Approach links that have turning-movement counts.
        counted: ``{(from, to): vehicles}`` to score against.
        max_inflow: Upper bound on an inflow, vehicles per hour.
        needs_calibration: Junction ids marked Y, or ``None`` for no gate.

    Returns:
        ``(problem, notes)``.
    """
    root = ET.parse(inpx).getroot()
    notes: list[str] = []

    turn_of = {(int(r.FromLinkNo_Vissim), int(r.ToLinkNo_Vissim)): str(r.Turn)
               for r in movements.itertuples(index=False)}
    junction_of = {int(r.FromLinkNo_Vissim): str(r.JunctionID_OpenDrive).removesuffix(".0")
                   for r in movements.itertuples(index=False)}

    inflows: list[InflowKnob] = []
    for entry in root.find("vehicleInputs") or []:
        vols = entry.find("timeIntVehVols")
        values = [float(v.get("volume") or 0) for v in (vols if vols is not None else [])]
        if values and sum(values) == 0:
            link_no = int(entry.get("link"))
            inflows.append(InflowKnob(int(entry.get("no")), link_no,
                                      junction_of.get(link_no, "?"), len(values)))

    intervals = max((k.intervals for k in inflows), default=4)
    splits: list[SplitKnob] = []
    for decision in root.find("vehicleRoutingDecisionsStatic") or []:
        number = int(decision.get("no"))
        sits_on = int(decision.get("link"))
        routes = []
        for route in decision.find("vehRoutSta") or []:
            seq = _route_sequence(route)
            if not seq:
                continue
            last = links.get(seq[-1])
            exit_no = int(last.to_link) if last is not None and last.is_connector else seq[-1]
            # The approach is the last ordinary link the route passes before its
            # final junction; with none, the decision sits on it already.
            plain = [n for n in seq[:-1]
                     if n in links and not links[n].is_connector and not links[n].is_internal]
            approach = plain[-1] if plain else sits_on
            routes.append((int(route.get("no")), approach, exit_no, _first_rel_flow(route)))
        if not routes:
            continue
        approaches = {a for _r, a, _e, _f in routes}
        if len(approaches) != 1:
            notes.append(f"decision {number} governs approaches {sorted(approaches)}; "
                         "left alone")
            continue
        approach = approaches.pop()
        if approach in counted_approaches:
            continue
        # One variable per exit, however many lane routes stage 3 wrote to it.
        by_exit: dict[int, list[tuple[int, float]]] = {}
        for r, _a, e, flow in routes:
            by_exit.setdefault(e, []).append((r, flow))
        free_exits = [e for e in by_exit if turn_of.get((approach, e)) != "Uturn"]
        zero = tuple(r for e, rs in by_exit.items() if turn_of.get((approach, e)) == "Uturn"
                     for r, _f in rs)
        if len(free_exits) < 2:
            continue                      # nothing to choose between
        lane_routes = {e: _lane_shares(by_exit[e]) for e in free_exits if len(by_exit[e]) > 1}
        splits.append(SplitKnob(number, approach, junction_of.get(approach, "?"),
                                tuple(by_exit[e][0][0] for e in free_exits), tuple(free_exits),
                                zero, intervals, lane_routes))

    if needs_calibration is not None:
        gated = {str(j).removesuffix(".0") for j in needs_calibration}
        dropped = ([k for k in inflows if k.junction not in gated]
                   + [k for k in splits if k.junction not in gated])
        if dropped:
            notes.append(f"{len(dropped)} knobs sit at junctions marked "
                         "'Need calibration? = N' and were left out: "
                         + "; ".join(k.label() for k in dropped))
        inflows = [k for k in inflows if k.junction in gated]
        splits = [k for k in splits if k.junction in gated]

    splits.sort(key=lambda k: (k.junction, k.decision_no))
    return Problem(inflows, splits, dict(counted or {}), max_inflow), notes


# --------------------------------------------------------------------------- #
# Writing a candidate in, and reading the result out
# --------------------------------------------------------------------------- #
def apply_solution(session, problem: Problem, solution) -> None:
    """Write one candidate demand into the loaded network.

    Args:
        session: A started :class:`~rt_vissim.com.VissimSession`.
        problem: The knobs being turned.
        solution: The optimiser's values, in the order :meth:`Problem.bounds`
            returns; inflows are fractions of ``max_inflow``.
    """
    solution = problem.physical(solution)
    net = session.net
    cursor = 0
    for knob in problem.inflows:
        value = max(0.0, float(solution[cursor]))
        cursor += 1
        entry = net.VehicleInputs.ItemByKey(knob.input_no)
        for interval in range(1, knob.intervals + 1):
            entry.SetAttValue(f"Volume({interval})", value)

    for knob in problem.splits:
        shares = stick_shares(solution[cursor:cursor + knob.size])
        cursor += knob.size
        weights = {route_no: share * lane_share
                   for index, share in enumerate(shares)
                   for route_no, lane_share in knob.routes_of(index)}
        weights.update({route_no: 0.0 for route_no in knob.zero_routes})
        decision = net.VehicleRoutingDecisionsStatic.ItemByKey(knob.decision_no)
        for route_no, weight in weights.items():
            route = decision.VehRoutSta.ItemByKey(route_no)
            for interval in range(1, knob.intervals + 1):
                route.SetAttValue(f"RelFlow({interval})", float(weight))


def counting_sites(links: dict, movements) -> tuple[dict[tuple[int, int], list[int]], list[str]]:
    """Where to count each movement: every junction path that carries it alone.

    Read off the network (:func:`rt_vissim.network.junction_paths`) rather
    than the movement table, which lists one junction link per movement: a
    double right turn has two (on Chattanooga, approach 35 turns right into
    link 10 over links 156 and 157), and counting only the listed one misses
    the traffic on the other.  Stage 3 routes over the same paths.

    Args:
        links: Output of :func:`rt_vissim.network.read_links_csv`.
        movements: ``(from link, to link)`` pairs to count.

    Returns:
        ``({(from, to): [link or connector numbers]}, notes)``.
    """
    from .network import junction_paths  # noqa: PLC0415

    sites: dict[tuple[int, int], list[int]] = {}
    notes: list[str] = []
    for a, b in movements:
        paths = junction_paths(links, int(a), int(b))
        if paths:
            sites[(int(a), int(b))] = [site for site, _lanes in paths]
        else:
            notes.append(f"movement {a} -> {b}: no path through the junction; not counted")
    return sites, notes


def place_counters(session, sites: dict[tuple[int, int], list[int]]) -> dict:
    """Put a data collection point on every lane of every site, one measurement per movement.

    Returns:
        ``{(from, to): DataCollectionMeasurement}`` for :func:`read_counters`.
    """
    net = session.net
    measures = {}
    for (a, b), site_links in sites.items():
        points: list[str] = []
        for number in site_links:
            link = net.Links.ItemByKey(number)
            middle = float(link.AttValue("Length2D")) / 2
            for lane in link.Lanes.GetAll():
                point = net.DataCollectionPoints.AddDataCollectionPoint(0, lane, middle)
                points.append(str(point.AttValue("No")))
        measure = net.DataCollectionMeasurements.AddDataCollectionMeasurement(0)
        measure.SetAttValue("DataCollectionPoints", ",".join(points))
        measure.SetAttValue("Name", f"movement {a}->{b}")
        measures[(a, b)] = measure
    return measures


def read_counters(measures: dict) -> dict[tuple[int, int], float]:
    """Vehicles each movement's measurement counted in the run just finished."""
    return {key: float(m.AttValue("Vehs(Current, Last, All)") or 0)
            for key, m in measures.items()}


def score(modelled: dict[tuple[int, int], float],
          counted: dict[tuple[int, int], float]) -> dict:
    """Score a run against the counts.

    Each counted turning movement is scored -- the objective.  The same run is
    also scored the way SUMO's ``result_analysis_on_EdgeData`` scores it: the
    movements summed per approach and one GEH per approach.  That test is
    looser (an approach can carry the right total split the wrong way) and is
    reported only, so the two pipelines' numbers can be set side by side.

    Returns:
        ``mean`` GEH, ``share`` below :data:`TARGET_GEH`, ``matched`` movements,
        totals and the worst few, so a run can be judged rather than just ranked;
        and ``approach_mean``, ``approach_share``, ``approaches`` for SUMO's test.
    """
    values = []
    for key, want in counted.items():
        got = modelled.get(key)
        if got is None:
            continue
        values.append((key, want, got, geh(got, want)))
    if not values:
        return {"mean": float("inf"), "share": 0.0, "matched": 0, "worst": [],
                "modelled_total": 0.0, "counted_total": 0.0,
                "approach_mean": float("inf"), "approach_share": 0.0, "approaches": 0}
    approaches: dict[int, list[float]] = {}
    for (from_link, _to), want, got, _g in values:
        pair = approaches.setdefault(from_link, [0.0, 0.0])
        pair[0] += want
        pair[1] += got
    by_approach = [geh(got, want) for want, got in approaches.values()]
    approach_rows = [(from_link, want, got, geh(got, want))
                     for from_link, (want, got) in sorted(approaches.items())]
    values.sort(key=lambda row: -row[3])
    return {"mean": sum(row[3] for row in values) / len(values),
            "share": sum(1 for row in values if row[3] < TARGET_GEH) / len(values),
            "matched": len(values), "worst": values[:10],
            "modelled_total": sum(row[2] for row in values),
            "counted_total": sum(row[1] for row in values),
            "approach_mean": sum(by_approach) / len(by_approach),
            "approach_share": sum(1 for g in by_approach if g < TARGET_GEH) / len(by_approach),
            "approaches": len(by_approach), "by_approach": approach_rows}


def speed_up(session) -> list[str]:
    """Stop Vissim drawing anything, since nobody is watching a calibration.

    A hidden window is not an idle one: Vissim still moves and colours every
    vehicle in the network editor each step, and refreshes its lists.  Quick
    Mode hides the dynamic objects -- "vehicles, pedestrians, dynamic labels,
    and colors" (manual, Using the Quick Mode) -- and ``SuspendUpdateGUI``
    stops the workspace refreshing at all.  The names are the ones PTV's own
    COM example uses (Examples Training/COM/Basic Commands).  Maximum
    simulation speed and all cores are set as well, though the networks this
    pipeline writes already carry both.

    Each is tried separately: a hidden instance may have no network window,
    and a missing one should not stop the run.

    Args:
        session: A started :class:`~rt_vissim.com.VissimSession`.

    Returns:
        The settings that took, for the log.
    """
    applied: list[str] = []
    steps = (
        ("QuickMode",
         lambda: session.vissim.Graphics.CurrentNetworkWindow.SetAttValue("QuickMode", 1)),
        ("SuspendUpdateGUI", lambda: session.vissim.SuspendUpdateGUI()),
        ("UseMaxSimSpeed",
         lambda: session.net.Simulation.SetAttValue("UseMaxSimSpeed", True)),
        ("UseAllCores", lambda: session.net.Simulation.SetAttValue("UseAllCores", True)),
    )
    for name, step in steps:
        try:
            step()
            applied.append(name)
        except Exception:  # noqa: BLE001 - not available in this instance
            continue
    return applied


# --------------------------------------------------------------------------- #
# Progress charts
# --------------------------------------------------------------------------- #
#: Chart colours: categorical slots 1-3 for GA, SA and TS, a light step of
#: slot 1 for individual evaluations, muted ink for reference lines.
_SERIES = {"ga": "#2a78d6", "sa": "#eb6834", "ts": "#1baf7a"}
_DOTS = "#86b6ef"
_INK, _MUTED, _SURFACE = "#52514e", "#898781", "#fcfcfb"


def read_history(path: str | Path, level: str = "approach") -> dict[str, list[float]]:
    """The evaluation log :class:`Evaluator` writes, as columns, at one level.

    ``level`` is ``"approach"`` (SUMO's, the default) or ``"movement"``; a log
    written before approach scores were recorded falls back to movements.
    Adds ``best`` (lowest mean GEH so far) and ``best_share`` (the GEH<5 share
    of that best candidate).
    """
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if level == "approach" and rows and "approach_mean_geh" in rows[0]:
        mean_col, share_col, used = "approach_mean_geh", "approach_share_below_5", "approach"
    else:
        mean_col, share_col, used = "mean_geh", "share_below_5", "movement"
    out = {"n": [int(r["n"]) for r in rows],
           "mean": [float(r[mean_col]) for r in rows],
           "share": [100 * float(r[share_col]) for r in rows],
           "best": [], "best_share": [], "level": used}
    best, best_share = math.inf, 0.0
    for mean, share in zip(out["mean"], out["share"]):
        if mean < best:
            best, best_share = mean, share
        out["best"].append(best)
        out["best_share"].append(best_share)
    return out


def _style(ax, ylabel: str) -> None:
    ax.set_facecolor(_SURFACE)
    ax.set_ylabel(ylabel, color=_INK)
    ax.grid(axis="y", color="#e6e5e1", linewidth=0.8)
    ax.tick_params(colors=_MUTED, labelcolor=_INK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_MUTED)


def _reference(ax, value: float | None, text: str, last: int) -> None:
    if value is None:
        return
    ax.axhline(value, color=_MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    ax.annotate(text, (last, value), xytext=(4, 3), textcoords="offset points",
                color=_INK, fontsize=8, ha="right", va="bottom")


def plot_history(path: str | Path, out: str | Path, title: str,
                 before: tuple[float, float] | None = None) -> Path:
    """Mean GEH and GEH<5 share against evaluation number, for one run.

    Two panels sharing the x axis rather than one with two scales.  Dots are
    every evaluation; the line is the best found so far -- the candidate the
    run will keep -- and, below, that candidate's share under GEH 5.

    Args:
        path: The ``*_history.csv``.
        out: The ``.png`` to write.
        title: The chart title, e.g. the algorithm and network.
        before: ``(mean GEH, GEH<5 share 0-1)`` of the uncalibrated model.

    Returns:
        ``out``.
    """
    import matplotlib  # noqa: PLC0415 - only needed when charting
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    h = read_history(path)
    last = h["n"][-1] if h["n"] else 1
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(8, 6.4), sharex=True,
                                      facecolor=_SURFACE)
    level = f"per {h['level']}"
    for ax, each, best, label in ((top, h["mean"], h["best"], f"Mean GEH, {level}"),
                                  (bottom, h["share"], h["best_share"], f"GEH < 5 (%), {level}")):
        _style(ax, label)
        ax.scatter(h["n"], each, s=14, color=_DOTS, zorder=2, label="each evaluation")
        ax.step(h["n"], best, where="post", color=_SERIES["ga"], linewidth=2, zorder=3,
                label="best so far")
        if best:
            ax.annotate(f"{best[-1]:.2f}" if ax is top else f"{best[-1]:.1f}%",
                        (h["n"][-1], best[-1]), xytext=(5, 0), textcoords="offset points",
                        color=_INK, fontsize=9, va="center")
    if before:
        _reference(top, before[0], f"as built {before[0]:.2f}", last)
        _reference(bottom, 100 * before[1], f"as built {100 * before[1]:.1f}%", last)
    _reference(bottom, 100 * TARGET_SHARE, f"target {100 * TARGET_SHARE:.0f}%", last)
    top.legend(frameon=False, loc="upper right", fontsize=8, labelcolor=_INK)
    bottom.set_xlabel("Evaluation (one simulation each)", color=_INK)
    top.set_title(title, loc="left", color="#0b0b0b", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=_SURFACE)
    plt.close(fig)
    return Path(out)


def plot_comparison(histories: dict[str, str | Path], out: str | Path, title: str,
                    before: tuple[float, float] | None = None) -> Path:
    """Best-so-far mean GEH and its GEH<5 share for several runs, on one chart.

    Args:
        histories: ``{"ga": path, "sa": path, ...}`` -- the keys pick colours.
        out: The ``.png`` to write.
        title: The chart title.
        before: ``(mean GEH, GEH<5 share 0-1)`` of the uncalibrated model.
    """
    import matplotlib  # noqa: PLC0415
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    runs = {k: read_history(p) for k, p in histories.items() if Path(p).exists()}
    last = max((h["n"][-1] for h in runs.values() if h["n"]), default=1)
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(8, 6.4), sharex=True,
                                      facecolor=_SURFACE)
    levels = {h["level"] for h in runs.values()}
    level = f"per {levels.pop()}" if len(levels) == 1 else "mixed levels"
    _style(top, f"Best mean GEH, {level}")
    _style(bottom, f"GEH < 5 of best (%), {level}")
    for algo, h in runs.items():
        if not h["n"]:
            continue
        colour = _SERIES.get(algo, _INK)
        name = algo.upper()
        for ax, values, fmt in ((top, h["best"], "{:.2f}"), (bottom, h["best_share"], "{:.1f}%")):
            ax.step(h["n"], values, where="post", color=colour, linewidth=2, label=name)
            ax.annotate(f"{name} {fmt.format(values[-1])}", (h["n"][-1], values[-1]),
                        xytext=(5, 0), textcoords="offset points", color=_INK,
                        fontsize=8, va="center")
    if before:
        _reference(top, before[0], f"as built {before[0]:.2f}", last)
        _reference(bottom, 100 * before[1], f"as built {100 * before[1]:.1f}%", last)
    _reference(bottom, 100 * TARGET_SHARE, f"target {100 * TARGET_SHARE:.0f}%", last)
    # Bottom left: the curves start high on the left and finish low on the right.
    top.legend(frameon=False, loc="lower left", fontsize=8, labelcolor=_INK)
    bottom.set_xlabel("Evaluation (one simulation each)", color=_INK)
    top.set_title(title, loc="left", color="#0b0b0b", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=_SURFACE)
    plt.close(fig)
    return Path(out)


# --------------------------------------------------------------------------- #
# The optimisers
# --------------------------------------------------------------------------- #
#: The algorithms SUMO's ``TurnInflowCali`` offers, by the names its
#: ``sel_algo`` takes, each configured by its block in ``realtwin_config.yaml``.
ALGORITHMS = ("ga", "sa", "ts")

GA_MODELS = ("BaseGA", "EliteSingleGA", "EliteMultiGA", "MultiGA", "SingleGA")
SA_MODELS = ("OriginalSA", "GaussianSA", "SwarmSA")


def simulations(algo: str, settings: dict) -> int:
    """How many simulations an algorithm costs at these settings.

    They differ a great deal at the same ``epoch``: GA scores its whole
    population every epoch, SA one neighbour, TS ``neighbour_size`` of them.
    At RealTwin's defaults that is 110, 12 and 102.  SwarmSA's inner loop
    varies, so it is counted as one population per epoch.
    """
    epoch = int(settings["epoch"])
    pop = int(settings.get("pop_size", 2))
    if algo == "ga":
        pop = max(pop, 10)
        return (epoch + 1) * pop
    if algo == "sa":
        if settings.get("model_selection", "OriginalSA") == "SwarmSA":
            return (epoch + 1) * pop
        return pop + epoch
    if algo == "ts":
        return pop + epoch * int(settings.get("neighbour_size", 10))
    raise ValueError(f"unknown algorithm {algo!r}; use one of {ALGORITHMS}")


def epoch_for(algo: str, settings: dict, budget: int) -> int:
    """The epoch that spends about ``budget`` simulations, so algorithms compare fairly."""
    epoch = 1
    while simulations(algo, {**settings, "epoch": epoch + 1}) <= budget:
        epoch += 1
    return epoch


def make_optimiser(algo: str, settings: dict):
    """Build the mealpy optimiser SUMO's ``run_GA``/``run_SA``/``run_TS`` builds.

    The same models, parameters and defaults, read from the same config keys.
    """
    from mealpy import GA, SA, TS  # noqa: PLC0415 - only a real run needs it

    epoch = int(settings["epoch"])
    if algo == "ga":
        model = settings.get("model_selection", "BaseGA")
        model = model if model in GA_MODELS else "BaseGA"
        common = {"epoch": epoch, "pop_size": max(int(settings.get("pop_size", 10)), 10),
                  "pc": settings.get("pc", 0.75), "pm": settings.get("pm", 0.1)}
        if model == "BaseGA":
            return GA.BaseGA(**common)
        extra = {"selection": settings.get("selection", "roulette"),
                 "k_way": settings.get("k_way", 0.2),
                 "crossover": settings.get("crossover", "uniform"),
                 "mutation": settings.get("mutation", "swap")}
        if model.startswith("Elite"):
            extra.update(elite_best=settings.get("elite_best", 0.1),
                         elite_worst=settings.get("elite_worst", 0.3))
        return getattr(GA, model)(**common, **extra)
    if algo == "sa":
        model = settings.get("model_selection", "OriginalSA")
        model = model if model in SA_MODELS else "OriginalSA"
        pop = int(settings.get("pop_size", 2))
        temp = settings.get("temp_init", 100)
        cooling = settings.get("cooling_rate", 0.891)
        if model == "OriginalSA":
            return SA.OriginalSA(epoch=epoch, pop_size=pop, temp_init=temp,
                                 step_size=settings.get("step_size", 0.1))
        if model == "GaussianSA":
            return SA.GaussianSA(epoch=epoch, pop_size=pop, temp_init=temp,
                                 cooling_rate=cooling, scale=settings.get("scale", 0.1))
        return SA.SwarmSA(epoch=epoch, pop_size=pop, max_sub_iter=5, t0=temp, t1=1,
                          move_count=5, mutation_rate=0.1, mutation_step_size=0.1,
                          mutation_step_size_damp=cooling)
    if algo == "ts":
        return TS.OriginalTS(epoch=epoch, pop_size=int(settings.get("pop_size", 2)),
                             tabu_size=settings.get("tabu_size", 10),
                             neighbour_size=settings.get("neighbour_size", 10),
                             perturbation_scale=settings.get("perturbation_scale", 0.05))
    raise ValueError(f"unknown algorithm {algo!r}; use one of {ALGORITHMS}")


class Evaluator:
    """Runs the model for one candidate demand and scores it.

    One Vissim session serves every evaluation: the network is loaded once and
    each candidate is written into it over COM, which is what makes a hundred
    evaluations affordable.  The random seed and resolution are fixed, so the
    search sees the effect of the parameters and not of the dice.

    Every evaluation is appended to ``history`` as it finishes, so a run that
    is stopped part way still says what it found.

    ``sites`` says where to count each movement (:func:`counting_sites`).  The
    counters are placed in the loaded network, so the input file is untouched;
    the calibrated model saved at the end keeps them, for checking in the GUI.
    """

    def __init__(self, session, problem: Problem, period: int,
                 sites: dict[tuple[int, int], list[int]], seed: int = 42,
                 resolution: int = 10, history: str | Path | None = None,
                 objective: str = "approach"):
        if objective not in OBJECTIVES:
            raise ValueError(f"objective must be one of {OBJECTIVES}, not {objective!r}")
        self.session = session
        self.problem = problem
        self.sites = sites
        self.objective = objective
        self.period = period
        self.seed = seed
        self.resolution = resolution
        self.history = Path(history) if history else None
        self.count = 0
        self.best: tuple[float, list[float], dict] | None = None
        self._prepare()
        if self.history is not None:
            with self.history.open("w", newline="") as handle:
                csv.writer(handle).writerow(
                    ["n", "seconds", "mean_geh", "share_below_5", "modelled_total",
                     "approach_mean_geh", "approach_share_below_5"]
                    + [f"x{i}" for i in range(problem.size)])

    def _prepare(self) -> None:
        net = self.session.net
        sim = net.Simulation
        sim.SetAttValue("SimPeriod", self.period)
        sim.SetAttValue("SimBreakAt", self.period)
        sim.SetAttValue("RandSeed", self.seed)
        sim.SetAttValue("SimRes", self.resolution)
        self.speed_settings = speed_up(self.session)
        self.measures = place_counters(self.session, self.sites)
        evaluation = net.Evaluation
        evaluation.SetAttValue("DataCollCollectData", True)
        # Measured from the start of the run, not the clock time it represents.
        evaluation.SetAttValue("DataCollFromTime", 0)
        evaluation.SetAttValue("DataCollToTime", self.period)
        evaluation.SetAttValue("DataCollInterval", self.period)

    def run(self, solution=None) -> dict:
        """Score one candidate, or the network as it stands when ``None``."""
        last_error = None
        for attempt in range(COM_RETRIES):
            try:
                if solution is not None:
                    apply_solution(self.session, self.problem, solution)
                self.session.net.Simulation.RunContinuous()
                return score(read_counters(self.measures), self.problem.counted)
            except Exception as exc:  # noqa: BLE001 - transient COM refusals
                last_error = exc
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"evaluation failed {COM_RETRIES} times: {last_error}")

    def value(self, result: dict) -> float:
        """The number the search minimises: mean GEH at the chosen level."""
        return result["approach_mean"] if self.objective == "approach" else result["mean"]

    def __call__(self, solution) -> float:
        """Objective for the optimiser: mean GEH per approach (SUMO's) or per movement."""
        began = time.time()
        result = self.run(solution)
        self.count += 1
        values = [float(v) for v in solution]
        if self.best is None or self.value(result) < self.best[0]:
            self.best = (self.value(result), values, result)
        if self.history is not None:
            # Inflows in vehicles per hour, so the file reads without the scale.
            with self.history.open("a", newline="") as handle:
                csv.writer(handle).writerow(
                    [self.count, round(time.time() - began, 1), round(result["mean"], 4),
                     round(result["share"], 4), round(result["modelled_total"], 1),
                     round(result["approach_mean"], 4), round(result["approach_share"], 4)]
                    + [round(v, 4) for v in self.problem.physical(values)])
        print(f"  :eval {self.count:>3}  per approach GEH {result['approach_mean']:6.2f}  "
              f"GEH<5 {result['approach_share']:6.1%}  |  per movement GEH "
              f"{result['mean']:5.2f}  GEH<5 {result['share']:6.1%}  |  "
              f"best ({self.objective}) {self.best[0]:6.2f}", flush=True)
        return self.value(result)
