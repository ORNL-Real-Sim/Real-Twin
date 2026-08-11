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
"""Simulator-agnostic intermediate representation (IR) of a RealTwin scenario.

The RealTwin SUMO pipeline goes straight from the MatchupTable to SUMO XML.  For
VISSIM there is no file format we can write by hand -- everything has to go in
through the COM API.  To keep the (testable, pure-python) data preparation apart
from the (Windows/VISSIM-only) COM calls, every ingestion module in this package
produces the dataclasses below, and :mod:`rt_vissim.com` is the only module that
knows how to push them into VISSIM.

Everything here is keyed on **VISSIM link numbers**, which is the common currency
of the MatchupTable (``FromLinkNo_Vissim`` / ``ToLinkNo_Vissim``).  OpenDRIVE road
IDs are deliberately *not* used: they do not survive regeneration of the ``.xodr``
(SUMO renumbers every road between netconvert versions) and VISSIM renumbers again
on import, so the link numbers read back out of the imported network in
:mod:`rt_vissim.network` are the only durable identifiers.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any
import json


@dataclass
class VehicleInput:
    """One VISSIM vehicle input (a demand injection on an origin link).

    Attributes:
        link_no: VISSIM link number of the origin link.
        interval_start: Start of the demand interval, in simulation seconds.
        interval_end: End of the demand interval, in simulation seconds.
        volume: Demand in vehicles per hour (VISSIM vehicle inputs are hourly).
        count: Raw vehicle count observed over the interval (kept for traceability).
        name: Human readable label, usually the intersection name.
    """

    link_no: int
    interval_start: float
    interval_end: float
    volume: float
    count: int = 0
    name: str = ""


@dataclass
class RoutingDecision:
    """One VISSIM static vehicle routing decision (all turns from one approach).

    A routing decision sits on the approach link and has one route per
    downstream link.  Relative flows are the turning ratios from the counts.

    Attributes:
        junction_id: Derived junction ID (matches ``JunctionID_OpenDrive``).
        from_link_no: VISSIM link number of the approach.
        interval_start: Start of the demand interval, in simulation seconds.
        interval_end: End of the demand interval, in simulation seconds.
        routes: ``{to_link_no: relative_flow}``.  Relative flows are the raw
            movement counts; VISSIM normalises them internally.
        name: Human readable label, usually ``"<intersection> <bound>"``.
    """

    junction_id: str | int
    from_link_no: int
    interval_start: float
    interval_end: float
    routes: dict[int, float] = field(default_factory=dict)
    name: str = ""


@dataclass
class SignalGroupTiming:
    """Timing of one signal group (NEMA phase) inside a fixed-time program.

    Green is expressed as an offset window inside the cycle so it maps directly
    onto VISSIM's fixed-time controller, which wants ``GreenEnd`` / ``RedEnd``
    rather than splits.

    Attributes:
        sg_no: VISSIM signal group number.  We reuse the Synchro phase number so
            the VISSIM model stays readable next to the Synchro model.
        phase: Synchro phase number (as a string, e.g. ``"2"``).
        green_start: Seconds into the cycle when green starts.
        green_end: Seconds into the cycle when green ends (start of yellow).
        yellow: Yellow (amber) duration in seconds.
        all_red: All-red duration in seconds.
        min_green: Minimum green in seconds (unused by fixed time, kept for RBC).
        max_green: Maximum green in seconds (the split used for fixed time).
        veh_ext: Vehicle extension in seconds (unused by fixed time, kept for RBC).
        ring: Synchro ring number.
        barrier: Synchro barrier number.
        position: Position within the ring, from Synchro's ``BRP`` code.  The
            barrier/ring/position triple is what orders the RBC sequence.
        split: Split in seconds -- ``max_green + yellow + all_red``.  Synchro
            does not store it; it is the sum, and the rings of a barrier must
            agree on it for the plan to be valid.
        coordinated: Whether this phase is a coordinated phase, i.e. named in
            Synchro's ``Reference Phase``.
        min_recall: Phase is served every cycle even with no call.
        max_recall: Phase is held to its maximum green every cycle.
        dual_entry: Phase may come up when only its ring partner is called.
        inhibit_max: Maximum green is inhibited while coordinated.
        start_up: Phase is green at controller start-up.
        protected_movements: Synchro movement codes served protected, e.g.
            ``["NBL", "SBL"]``.
        permitted_movements: Synchro movement codes served permitted.
        rtor_movements: Synchro movement codes allowed right-turn-on-red.
    """

    sg_no: int
    phase: str
    green_start: float = 0.0
    green_end: float = 0.0
    yellow: float = 3.0
    all_red: float = 0.0
    min_green: float = 0.0
    max_green: float = 0.0
    veh_ext: float = 2.0
    ring: int = 1
    barrier: int = 1
    position: int = 1
    split: float = 0.0
    coordinated: bool = False
    min_recall: bool = False
    max_recall: bool = False
    dual_entry: bool = False
    inhibit_max: bool = False
    start_up: bool = False
    protected_movements: list[str] = field(default_factory=list)
    permitted_movements: list[str] = field(default_factory=list)
    rtor_movements: list[str] = field(default_factory=list)


@dataclass
class SignalPlan:
    """A complete signal controller plan for one intersection.

    Attributes:
        sc_no: VISSIM signal controller number.
        junction_id: Derived junction ID this controller belongs to
            (matches ``JunctionID_OpenDrive``).
        synchro_intid: Synchro ``INTID`` this plan was built from.
        name: Human readable intersection name.
        cycle_time: Cycle length in seconds.
        offset: Offset in seconds.
        controller_type: ``"FIXEDTIME"`` or ``"RBC"``.  Only ``FIXEDTIME`` is
            written today; see the package README for the RBC roadmap.
        coordinated: Whether Synchro reported the intersection as coordinated.
        signal_groups: Timings, in phase order.
        movement_to_sg: ``{synchro_movement_code: sg_no}``, e.g. ``{"NBL": 5}``.
            Used to attach signal heads to the right lanes.
    """

    sc_no: int
    junction_id: str | int
    synchro_intid: str
    name: str = ""
    cycle_time: float = 90.0
    offset: float = 0.0
    controller_type: str = "FIXEDTIME"
    coordinated: bool = False
    signal_groups: list[SignalGroupTiming] = field(default_factory=list)
    movement_to_sg: dict[str, int] = field(default_factory=dict)


@dataclass
class SignalHead:
    """A signal head to place on one lane of one approach.

    Attributes:
        sc_no: Signal controller number.
        sg_no: Signal group number.
        junction_id: Derived junction ID (matches ``JunctionID_OpenDrive``).
        from_link_no: VISSIM link number of the approach the head sits on.
        to_link_no: VISSIM link number of the receiving link (identifies the turn).
        movement: Synchro movement code, e.g. ``"NBL"``.
        turn: RealTwin turn label: ``left`` / ``thru`` / ``right`` / ``Uturn``.
    """

    sc_no: int
    sg_no: int
    junction_id: str | int
    from_link_no: int
    to_link_no: int
    movement: str = ""
    turn: str = ""


@dataclass
class ScenarioIR:
    """Everything needed to build a VISSIM scenario, before any COM call.

    Attributes:
        network_name: Scenario / network name, e.g. ``"chatt"``.
        xodr_path: Path to the OpenDRIVE file to import.
        sim_start_time: Simulation start, in seconds after midnight.
        sim_end_time: Simulation end, in seconds after midnight.
        vehicle_inputs: Demand injections.
        routing_decisions: Static routing decisions.
        signal_plans: Signal controller plans.
        signal_heads: Signal heads to place.
        meta: Free-form provenance (input paths, counts, warnings).
    """

    network_name: str
    xodr_path: str = ""
    sim_start_time: float = 0.0
    sim_end_time: float = 3600.0
    vehicle_inputs: list[VehicleInput] = field(default_factory=list)
    routing_decisions: list[RoutingDecision] = field(default_factory=list)
    signal_plans: list[SignalPlan] = field(default_factory=list)
    signal_heads: list[SignalHead] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Return the IR as plain nested dicts/lists."""
        return asdict(self)

    def to_json(self, path: str, indent: int = 2) -> str:
        """Write the IR to ``path`` as JSON and return the path.

        Dumping the IR is the fastest way to review what the pipeline decided
        without a VISSIM licence in the loop.
        """
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=indent, default=str)
        return path

    def summary(self) -> str:
        """Return a one-paragraph human readable summary of the IR."""
        n_int = len({vi.name for vi in self.vehicle_inputs})
        return (
            f"Scenario '{self.network_name}': "
            f"{len(self.vehicle_inputs)} vehicle inputs ({n_int} origins/intersections), "
            f"{len(self.routing_decisions)} routing decisions, "
            f"{len(self.signal_plans)} signal controllers, "
            f"{len(self.signal_heads)} signal heads, "
            f"sim window {self.sim_start_time:.0f}-{self.sim_end_time:.0f}s."
        )
