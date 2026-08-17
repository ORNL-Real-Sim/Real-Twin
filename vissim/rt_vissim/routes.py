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
"""Build one integrated routing decision per approach.

RealTwin's SUMO flow hands turn ratios to ``jtrrouter``, which expands them into
routes.  Vissim has no router, so the turns are expressed directly as static
routing decisions -- but *how* they are expressed matters, because a vehicle that
is already on a route ignores every routing decision it passes until it reaches
that route's destination (Vissim manual, 2.16.3.3).

So each approach gets a **single** decision carrying a route to **every** exit it
can reach, not one decision per movement.  Movements with no count are written at
relative flow 0 rather than left out: a movement with no route is not "unknown"
to Vissim, it is forbidden, and vehicles can never make that turn.  A zero is
explicit, visible in the model, and becomes a live movement the moment a number
is put in its place.

Positions follow the manual's ordering rule -- "ensure that all preceding routes
end before the start of a new route" -- structurally rather than by tuning.  A
route ends a short way into its exit link, and any decision on that same link
sits further downstream still, so a decision can never fall inside a live route.
"""

from __future__ import annotations

import pandas as pd

from .ir import RoutingDecision

#: Where a route ends on its destination link, in metres from the link start.
#: Far enough past the junction to commit the turn, short enough to leave the
#: rest of the link as approach for the next decision.
ROUTE_END_OFFSET = 2.0

#: Fallback maximum desired speed in km/h, used when the network's speed
#: distributions cannot be read.  Only sets the minimum decision gap.
FALLBACK_MAX_SPEED_KMH = 120.0


def minimum_gap(max_speed_kmh: float, sim_resolution: int) -> float:
    """Return the smallest workable distance between a route end and a decision.

    The manual requires that "the distance between the decision marker and the
    subsequent link or connector must be defined at least large enough so that
    the length of the path corresponds to the vehicle with the highest possible
    desired speed within a time step", because a decision marker only takes
    effect on the following time step.

    Args:
        max_speed_kmh: Highest desired speed in the network, km/h.
        sim_resolution: Simulation steps per second.

    Returns:
        Distance in metres, never less than 0.5 m.
    """
    per_step = (max_speed_kmh / 3.6) / max(1, sim_resolution)
    return max(0.5, round(per_step, 2))


def decision_position(link_length: float, gap: float) -> tuple[float, list[str]]:
    """Return where a decision sits on its approach link.

    As early on the link as the ordering rule allows, which maximises the
    distance vehicles have to pre-sort into the correct lane, but never past
    mid-link: a decision sitting almost at the link end can be downstream of
    where the outgoing connectors leave, and Vissim then finds no path from it to
    any destination and the routes are silently empty.

    Args:
        link_length: Length of the approach link in metres.
        gap: Minimum gap from :func:`minimum_gap`.

    Returns:
        ``(position, warnings)``.
    """
    wanted = ROUTE_END_OFFSET + gap
    cap = link_length / 2.0
    if wanted <= cap:
        return round(wanted, 2), []

    # Too short to hold the route end and the full gap ahead of mid-link.  Keep
    # the ordering (after the route end) and give up gap instead, since a route
    # with no path is worse than a decision some vehicles miss.
    position = max(ROUTE_END_OFFSET + 0.1, cap)
    return round(position, 2), [
        f"approach is only {link_length:.1f} m: decision at {position:.1f} m "
        f"leaves {position - ROUTE_END_OFFSET:.1f} m of gap, under the "
        f"{gap:.1f} m one time step needs, so some vehicles may miss it"]


def plan_decision_positions(approaches: list[tuple], links: dict,
                            destinations: set[int], gap: float,
                            ) -> tuple[dict[tuple, tuple[int, float]], list[str]]:
    """Choose where each approach's decision sits, as far upstream as is legal.

    Placing a decision on its own approach link wastes every metre between the
    previous junction and that link.  On Chattanooga the decision for junction 4
    sat on link 48 even though the routes feeding it ended several links
    upstream, so vehicles learnt their turn far later than they could have.

    So the walk goes back from the approach through the corridor until it reaches
    the link where the upstream routes end, and the decision goes just after that
    end.  The walk only crosses a link when the connection is unambiguous in both
    directions -- one way in and one way out -- because at a fork some vehicles
    would not be heading for this junction at all, and assigning them a route
    through it would be wrong.

    Args:
        approaches: ``(junction_id, from_link)`` pairs.
        links: Output of :func:`rt_vissim.network.read_links_csv`.
        destinations: Links that routes end on.
        gap: Minimum gap from :func:`minimum_gap`.

    Returns:
        ``({(junction_id, from_link): (link, position)}, warnings)``.
    """
    from .demand import road_predecessors  # local: avoids a circular import

    preds = road_predecessors(links)
    succs: dict[int, set[int]] = {}
    for target, sources in preds.items():
        for source in sources:
            succs.setdefault(source, set()).add(target)

    placements: dict[tuple, tuple[int, float]] = {}
    taken: dict[int, tuple] = {}
    moved = 0
    warnings: list[str] = []

    for junction_id, from_link in sorted(approaches, key=lambda a: (str(a[0]), a[1])):
        # Walk upstream through the unambiguous corridor, nearest first.
        chain, current = [from_link], from_link
        while current not in destinations:
            upstream = preds.get(current, set())
            if len(upstream) != 1:
                break                      # a merge, or the network entry
            nxt = next(iter(upstream))
            if nxt in chain or succs.get(nxt, set()) != {current}:
                break                      # a loop, or a fork: not all traffic
                                           # here is bound for this junction
            chain.append(nxt)
            current = nxt

        # Prefer the furthest upstream link that no other approach has claimed.
        anchor = next((c for c in reversed(chain) if c not in taken), from_link)
        length = links[anchor].length if anchor in links else 0.0
        if anchor in destinations:
            position, notes = decision_position(length, gap)
        else:
            # A network entry: nothing ends here, so sit one gap in.
            position = round(min(gap, max(0.5, length / 2.0)), 2)
            notes = []
        for note in notes:
            warnings.append(f"junction {junction_id}: {note}")

        placements[(junction_id, from_link)] = (anchor, position)
        taken[anchor] = (junction_id, from_link)
        if anchor != from_link:
            moved += 1

    if moved:
        warnings.append(f"{moved} of {len(approaches)} decisions moved upstream of "
                        "their approach, so vehicles are told of the turn earlier.")
    return placements, warnings


def build_integrated_decisions(movements: pd.DataFrame, turn_counts: pd.DataFrame,
                               ) -> tuple[list[RoutingDecision], list[str]]:
    """Build one decision per (approach, interval), covering every movement.

    Args:
        movements: The stage 1 movement table, one row per turn movement.
        turn_counts: Output of :func:`rt_vissim.demand.build_turn_counts`.

    Returns:
        ``(decisions, warnings)``.  Relative flows are raw counts; Vissim
        normalises them, so the numbers stay checkable against GridSmart.
    """
    warnings: list[str] = []
    if movements is None or movements.empty:
        return [], ["No movements; cannot build routing decisions."]

    intervals = sorted({(float(a), float(b)) for a, b
                        in zip(turn_counts["IntervalStart"], turn_counts["IntervalEnd"])}) \
        if not turn_counts.empty else []
    if not intervals:
        return [], ["No demand intervals; cannot build routing decisions."]

    # Counts keyed by the movement they belong to.
    counted: dict[tuple[int, int, float], float] = {}
    for row in turn_counts.itertuples(index=False):
        key = (int(row.FromLinkNo_Vissim), int(row.ToLinkNo_Vissim),
               float(row.IntervalStart))
        counted[key] = counted.get(key, 0.0) + float(row.Count)

    names = {}
    if "IntersectionName" in turn_counts.columns:
        for row in turn_counts.itertuples(index=False):
            names.setdefault(str(row.JunctionID_OpenDrive), str(row.IntersectionName))

    decisions: list[RoutingDecision] = []
    uncounted_approaches = 0
    single_route = []

    for (junction_id, from_link), group in movements.groupby(
            ["JunctionID_OpenDrive", "FromLinkNo_Vissim"], sort=True):
        exits = sorted({int(x) for x in group["ToLinkNo_Vissim"]})
        if not exits:
            continue
        if len(exits) == 1:
            # An approach with one exit offers no choice, so a decision here
            # decides nothing -- and it is worse than merely redundant.  A
            # vehicle already on a route ignores every decision it passes until
            # that route ends (manual, 2.16.3.3), so a one-route decision blinds
            # vehicles to the *next* one and eats the room it needed.  Leaving it
            # out lets the downstream decision move further upstream instead.
            single_route.append((junction_id, int(from_link)))
            continue
        bound = str(group["Bound"].iloc[0]) if "Bound" in group.columns else ""
        label = names.get(str(junction_id), f"Junction {junction_id}")
        has_any = False

        for start, end in intervals:
            routes = {e: counted.get((int(from_link), e, start), 0.0) for e in exits}
            if sum(routes.values()) > 0:
                has_any = True
            decisions.append(RoutingDecision(
                junction_id=junction_id,
                from_link_no=int(from_link),
                interval_start=start,
                interval_end=end,
                routes=routes,
                name=f"{label} {bound}".strip(),
            ))
        if not has_any:
            uncounted_approaches += 1

    if single_route:
        where = ", ".join(f"J{j} link {f}" for j, f in single_route)
        warnings.append(
            f"{len(single_route)} approaches have a single exit and were given no "
            f"routing decision ({where}); vehicles have no choice to make there, "
            "and a decision would have hidden the next one from them.")
    if uncounted_approaches:
        warnings.append(
            f"{uncounted_approaches} approaches have no counts in any interval; "
            "their routes are written at relative flow 0 and need a number before "
            "vehicles will use them.")
    return decisions, warnings


def summarise(decisions: list[RoutingDecision]) -> str:
    """Return a one-line summary of a decision set."""
    if not decisions:
        return "no routing decisions"
    approaches = {(d.junction_id, d.from_link_no) for d in decisions}
    intervals = {(d.interval_start, d.interval_end) for d in decisions}
    routes = sum(len(d.routes) for d in decisions)
    zero = sum(1 for d in decisions for v in d.routes.values() if v == 0)
    return (f"{len(approaches)} approaches, {len(intervals)} intervals, "
            f"{routes} route-intervals ({zero} at zero)")
