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

from .ir import VehicleInput

#: Vissim's time-interval set for vehicle inputs.
VEHICLE_INPUT_TIS = 1

#: ``VolType`` 1 is a stochastic (Poisson) arrival process, which is what a
#: turn-count derived volume implies.  Type 2 would be an exact vehicle count.
VOLUME_TYPE_STOCHASTIC = 1


def build_time_intervals(session, intervals: list[tuple[float, float]],
                         sim_start_time: float) -> dict[tuple[float, float], int]:
    """Create the vehicle-input time intervals and return their numbers.

    Interval 1 always exists and starts at 0, so it is reused rather than added.

    Args:
        session: A started :class:`~rt_vissim.com.VissimSession`.
        intervals: ``(start, end)`` pairs in seconds after midnight, sorted.
        sim_start_time: Scenario start in seconds after midnight, mapped to
            simulation time 0.

    Returns:
        ``{(start, end): interval number}``, numbered from 1.
    """
    time_ints = session.net.TimeIntervalSets.ItemByKey(VEHICLE_INPUT_TIS).TimeInts
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
