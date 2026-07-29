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
"""Push the scenario IR into Vissim over COM.

The only module besides :mod:`rt_vissim.com` that needs a Vissim licence.
Everything upstream produces the dataclasses in :mod:`rt_vissim.ir`; this turns
them into network objects.

Time handling follows the prior ORNL work in
``VISSIM_previous/SimulationGeneratorCodes/vissim_volume_population.ipynb``:
Vissim's vehicle-input volumes are indexed by time-interval number, so the
intervals are created first on the vehicle-input time-interval set, and each
input then carries one ``Volume(n)`` per interval.

Vissim counts simulation time from zero, while the IR carries seconds after
midnight, so every interval is shifted by the scenario's start time.
"""

from __future__ import annotations

from .ir import RoutingDecision, VehicleInput
from .routes import (FALLBACK_MAX_SPEED_KMH, ROUTE_END_OFFSET, decision_position,
                     minimum_gap)

#: Vissim's time-interval set for vehicle inputs.
VEHICLE_INPUT_TIS = 1

#: Vissim's time-interval set for static vehicle routes, which indexes
#: ``RelFlow(n)``.  Confirmed against Vissim 2026: adding an interval to set 2,
#: and only set 2, adds a RelFlow slot.
ROUTE_TIS = 2

#: ``VolType`` 1 is a stochastic (Poisson) arrival process, which is what a
#: turn-count derived volume implies.  Type 2 would be an exact vehicle count.
VOLUME_TYPE_STOCHASTIC = 1


def build_time_intervals(session, intervals: list[tuple[float, float]],
                         sim_start_time: float,
                         tis: int = VEHICLE_INPUT_TIS) -> dict[tuple[float, float], int]:
    """Create time intervals on one interval set and return their numbers.

    Interval 1 always exists and starts at 0, so it is reused rather than added.

    Args:
        session: A started :class:`~rt_vissim.com.VissimSession`.
        intervals: ``(start, end)`` pairs in seconds after midnight, sorted.
        sim_start_time: Scenario start in seconds after midnight, mapped to
            simulation time 0.
        tis: Which time-interval set to populate.

    Returns:
        ``{(start, end): interval number}``, numbered from 1.
    """
    time_ints = session.net.TimeIntervalSets.ItemByKey(tis).TimeInts
    numbering: dict[tuple[float, float], int] = {}

    for index, (start, end) in enumerate(intervals, start=1):
        if index > 1:
            time_ints.AddTimeInterval(index)
        interval = time_ints.ItemByKey(index)
        interval.SetAttValue("Start", max(0.0, start - sim_start_time))
        numbering[(start, end)] = index
    return numbering


def write_vehicle_inputs(session, inputs: list[VehicleInput],
                         sim_start_time: float) -> tuple[int, list[str]]:
    """Create one Vissim vehicle input per origin link, with per-interval volumes.

    An input is created once on its link and then given a volume for every
    interval.  Intervals the link has no demand for are written as 0 rather than
    left unset, so a gap in the counts does not silently inherit the previous
    interval's volume.

    Args:
        session: A started :class:`~rt_vissim.com.VissimSession`.
        inputs: The vehicle inputs to write.
        sim_start_time: Scenario start in seconds after midnight.

    Returns:
        ``(number of inputs created, warnings)``.
    """
    warnings: list[str] = []
    if not inputs:
        return 0, ["No vehicle inputs to write."]

    intervals = sorted({(vi.interval_start, vi.interval_end) for vi in inputs})
    numbering = build_time_intervals(session, intervals, sim_start_time)

    by_link: dict[int, dict[tuple[float, float], VehicleInput]] = {}
    for vi in inputs:
        by_link.setdefault(vi.link_no, {})[(vi.interval_start, vi.interval_end)] = vi

    created = 0
    for link_no, per_interval in sorted(by_link.items()):
        try:
            link = session.net.Links.ItemByKey(link_no)
        except Exception:  # noqa: BLE001 - COM raises a generic error for a missing key
            warnings.append(f"Link {link_no} not found; skipped its vehicle input.")
            continue

        veh_input = session.net.VehicleInputs.AddVehicleInput(0, link)
        name = next((vi.name for vi in per_interval.values() if vi.name), "")
        if name:
            veh_input.SetAttValue("Name", f"{name} (link {link_no})")

        for interval, number in numbering.items():
            vi = per_interval.get(interval)
            veh_input.SetAttValue(f"Volume({number})", float(vi.volume) if vi else 0.0)
            veh_input.SetAttValue(f"VolType({number})", VOLUME_TYPE_STOCHASTIC)
        created += 1

    return created, warnings


def network_max_speed(session) -> float:
    """Return the highest desired speed in the network, in km/h.

    Falls back to :data:`~rt_vissim.routes.FALLBACK_MAX_SPEED_KMH` when the
    distributions cannot be read; the value only sets the minimum decision gap.
    """
    try:
        speeds = []
        for dist in session.net.DesSpeedDistributions.GetAll():
            for point in dist.DesSpeedDistrDatPts.GetAll():
                speeds.append(float(point.AttValue("Fx")))
        if speeds:
            return max(speeds)
    except Exception:  # noqa: BLE001 - attribute layout varies by version
        pass
    return FALLBACK_MAX_SPEED_KMH


def write_routing_decisions(session, decisions: list[RoutingDecision],
                            sim_start_time: float, links: dict,
                            sim_resolution: int = 10) -> tuple[int, int, list[str]]:
    """Create one static routing decision per approach, with a route per exit.

    All of an approach's intervals share one decision object: Vissim indexes the
    relative flows by time interval, so the decision is created once and each
    route carries a ``RelFlow(n)`` per interval.

    Args:
        session: A started :class:`~rt_vissim.com.VissimSession`.
        decisions: Output of
            :func:`rt_vissim.routes.build_integrated_decisions`.
        sim_start_time: Scenario start in seconds after midnight.
        links: Output of :func:`rt_vissim.network.read_links_csv`, for lengths.
        sim_resolution: Simulation steps per second, for the minimum gap.

    Returns:
        ``(decisions created, routes created, warnings)``.
    """
    warnings: list[str] = []
    if not decisions:
        return 0, 0, ["No routing decisions to write."]

    gap = minimum_gap(network_max_speed(session), sim_resolution)
    warnings.append(f"Minimum decision gap {gap} m "
                    f"(fastest vehicle in one 1/{sim_resolution} s step).")

    intervals = sorted({(d.interval_start, d.interval_end) for d in decisions})
    numbering = build_time_intervals(session, intervals, sim_start_time, ROUTE_TIS)

    by_approach: dict[tuple, list[RoutingDecision]] = {}
    for decision in decisions:
        by_approach.setdefault((decision.junction_id, decision.from_link_no),
                               []).append(decision)

    made_decisions = made_routes = 0
    short = []
    for (junction_id, from_link), group in sorted(
            by_approach.items(), key=lambda kv: (str(kv[0][0]), kv[0][1])):
        try:
            link = session.net.Links.ItemByKey(from_link)
        except Exception:  # noqa: BLE001 - COM raises generically for a bad key
            warnings.append(f"Approach link {from_link} not found; skipped.")
            continue

        length = links[from_link].length if from_link in links else 0.0
        position, notes = decision_position(length, gap)
        if notes:
            short.append(f"link {from_link} ({junction_id})")

        vrd = session.net.VehicleRoutingDecisionsStatic.AddVehicleRoutingDecisionStatic(
            0, link, position)
        vrd.SetAttValue("Name", f"{group[0].name} (J{junction_id})".strip())
        made_decisions += 1

        exits = sorted({e for d in group for e in d.routes})
        for exit_link in exits:
            try:
                dest = session.net.Links.ItemByKey(exit_link)
            except Exception:  # noqa: BLE001
                warnings.append(f"Exit link {exit_link} not found; route skipped.")
                continue
            route = vrd.VehRoutSta.AddVehicleRouteStatic(0, dest, ROUTE_END_OFFSET)
            made_routes += 1
            for decision in group:
                number = numbering[(decision.interval_start, decision.interval_end)]
                route.SetAttValue(f"RelFlow({number})",
                                  float(decision.routes.get(exit_link, 0.0)))

    if short:
        warnings.append(f"{len(short)} approaches are too short to hold a route end "
                        f"and a decision gap; decision placed mid-link: "
                        f"{', '.join(short[:6])}"
                        + (" ..." if len(short) > 6 else ""))
    return made_decisions, made_routes, warnings
