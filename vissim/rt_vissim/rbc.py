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
"""Write Ring Barrier Controller (``.prbc``) files from the scenario IR.

Vissim's RBC is a real actuated controller: during a run Vissim hands it the
state of the detectors and signal heads and it returns the signal states for
the next step (manual, "Using RBC").  The timings live outside the ``.inpx`` in
a ``.prbc`` file, which is JSON.

Every duration in the file is in **tenths of a second** -- ``CycleLength: 1000``
is 100 s, ``Yellow: 42`` is 4.2 s.  The format was settled by reading the
controller files under ``VISSIM_previous/``, which also fix the field names and
nesting this module reproduces.

Keeping the plan actuated is the point.  RealTwin's SUMO path flattens Synchro
into a fixed-time ``tlLogic`` because SUMO has no equivalent; here the recalls,
extensions and minimum greens survive into the model.

.. warning::
   The manual is explicit that signal group numbers tie the two files together:
   "When you add, delete, or change signal group numbers, save your Vissim file.
   Otherwise your Vissim file may become incompatible with your controller
   files."  The signal groups created in Vissim must therefore carry the same
   numbers as :func:`build_controller` writes here, which is why both come from
   the Synchro phase number.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from .ir import SignalPlan
from .signal import sequence

#: Controller steps per second.  A tenth of a second is the finest Vissim
#: exchanges signal state at, and what the reference controllers use.
EXECUTION_FREQUENCY = 10

#: Where an offset is measured from.  Synchro's "Referenced To" is the leading
#: edge of green on the coordinated phase, which is this.
OFFSET_REFERENCE = "LeadingStartOfGreen"

#: How the pattern treats maximum green.  Every reference controller uses this
#: value and Vissim rejects the file outright for anything else -- "the value of
#: attribute MaxGreenMode is invalid" -- so it is fixed rather than derived.
#: The per-phase inhibit flag still comes from Synchro, on each signal group.
MAX_GREEN_MODE = "InhibitMaxGreen"

#: How permissive periods open and close for non-coordinated groups.  Note this
#: is coordination, not permissive left turns, despite the name.
PERMISSIVE_MODE = "SingleBand"

#: The ``.prbc`` schema version, as a ``[major, minor]`` pair.  Vissim refuses
#: the file outright if this is wrong -- "The supply data contains invalid
#: format version data" -- and the simulation then will not start at all, so it
#: is taken verbatim from the controllers under ``VISSIM_previous/``.
FORMAT_VERSION = [1, 1]


def tenths(seconds: float) -> int:
    """Convert seconds to the tenths of a second the format stores.

    Rounds halves upward rather than to even.  Synchro already stores its times
    in tenths, so an exact half never arises from real data, but ``round`` would
    send 0.05 s to 0 and 0.15 s to 2, which is a surprise not worth leaving in a
    unit conversion.
    """
    return int(math.floor(float(seconds) * 10 + 0.5))


def build_controller(plan: SignalPlan) -> dict:
    """Build the ``.prbc`` document for one intersection.

    Args:
        plan: A :class:`~rt_vissim.ir.SignalPlan` from
            :func:`rt_vissim.signal.build_signal_plan`.

    Returns:
        The complete document, ready for :func:`json.dump`.

    Raises:
        ValueError: If the plan has no signal groups.
    """
    if not plan.signal_groups:
        raise ValueError(f"Signal plan for INTID {plan.synchro_intid} has no "
                         "signal groups; nothing to write.")

    groups = sorted(plan.signal_groups, key=lambda g: g.sg_no)

    vehicle_groups = []
    for group in groups:
        split = tenths(group.split)
        red_clearance = tenths(group.all_red)
        # Synchro carries tenths on MaxGreen while the RBC editor writes whole
        # seconds, so the rounding remainder goes to yellow and the split stays
        # exact.  Reproducing this is what makes the output identical to a
        # hand-built controller rather than merely close.
        max_green = int(round(group.max_green)) * 10
        yellow = split - max_green - red_clearance
        if yellow <= 0:  # a plan too tight to round; keep Synchro's own yellow
            yellow = tenths(group.yellow)
            max_green = max(0, split - yellow - red_clearance)

        vehicle_groups.append({
            "DualEntry": bool(group.dual_entry),
            "ID": group.sg_no,
            "MaxGreen1": max_green,
            "MaxGreen2": 0,
            "MaxGreen3": 0,
            "MaxRecall": bool(group.max_recall),
            "MinGreen": tenths(group.min_green),
            "MinRecall": bool(group.min_recall),
            "Name": str(group.sg_no),
            "RedClearance": red_clearance,
            "StartUp": bool(group.start_up),
            "VehExtension": tenths(group.veh_ext),
            "Yellow": yellow,
        })

    in_pattern = [{
        "Coordinated": bool(group.coordinated),
        "ForceOff": 0,
        "InhibitMaxGreen": bool(group.inhibit_max),
        "Lead": False,
        "MaxRecall": bool(group.max_recall),
        "MinRecall": bool(group.min_recall),
        "OverridingMaxGreen": 0,
        "OverridingMinGreen": 0,
        "OverridingVehExtension": 0,
        "PermissivePeriodEnd": 0,
        "PermissivePeriodStart": 0,
        "Split": tenths(group.split),
        "UseMaxGreen2": False,
        "UseMaxGreen3": False,
        "VehicleSignalGroup": group.sg_no,
    } for group in groups]

    barriers = [{"RingGroups": [{"VehicleSignalGroups": ring} for ring in barrier]}
                for barrier in sequence(plan)]

    # One detector per signal group, calling and extending it.  That is what the
    # reference controllers do and it is the most that can be said before the
    # detectors themselves are placed on lanes: the port numbers written here
    # are what those detectors will have to carry.
    detectors = [{"CalledSGs": [group.sg_no],
                  "ExtendedSGs": [group.sg_no],
                  "ID": group.sg_no} for group in groups]

    return {
        "Controller": {
            "ExecutionFrequency": EXECUTION_FREQUENCY,
            "OffsetReference": OFFSET_REFERENCE,
            "PatternSchedule": {"PatternScheduleItems": [{"Pattern": 1,
                                                          "StartTime": 0}]},
            "Patterns": [{
                "CycleLength": tenths(plan.cycle_time),
                "ID": 1,
                "MaxGreenMode": MAX_GREEN_MODE,
                "Offset": tenths(plan.offset),
                "PedestrianSignalGroupsInPattern": [],
                "PermissiveMode": PERMISSIVE_MODE,
                "UseExplicitForceOffs": False,
                "UseExplicitPermissivePeriods": False,
                "VehicleSignalGroupsInPattern": in_pattern,
            }],
            "PedestrianDetectors": [],
            "PedestrianSignalGroups": [],
            "Sequence": {"BarrierGroups": barriers},
            "VehicleDetectors": detectors,
            "VehicleOverlapSignalGroups": [],
            "VehicleSignalGroups": vehicle_groups,
        },
        "FormatVersion": FORMAT_VERSION,
        # The filename carries the Synchro INTID, which does not match the
        # OpenDRIVE junction number, so say plainly which intersection this is.
        "GUI": {"Notes": (f"RealTwin: OpenDRIVE junction {plan.junction_id}, "
                          f"Synchro INTID {plan.synchro_intid}"
                          + (f" -- {plan.name}" if plan.name else ""))},
    }


def write_controller(plan: SignalPlan, path: str | Path) -> Path:
    """Write one intersection's ``.prbc`` file.

    Args:
        plan: The plan to write.
        path: Destination ``.prbc`` path.

    Returns:
        The written path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(build_controller(plan), handle, indent=2, sort_keys=True)
    return path


def write_controllers(plans: list[SignalPlan], outdir: str | Path, *,
                      stem: str = "rbc_timings") -> tuple[list[Path], list[str]]:
    """Write a ``.prbc`` file per plan, named after the Synchro ``INTID``.

    The naming follows the prior ORNL model (``rbc_timings_16.prbc``), so the
    generated controllers drop in beside the hand-built ones.

    Args:
        plans: Plans to write.
        outdir: Directory to write into.
        stem: Filename stem before the ``INTID``.

    Returns:
        ``(written paths, warnings)``.
    """
    outdir = Path(outdir)
    written: list[Path] = []
    warnings: list[str] = []
    for plan in plans:
        try:
            written.append(write_controller(
                plan, outdir / f"{stem}_{plan.synchro_intid}.prbc"))
        except ValueError as exc:
            warnings.append(str(exc))
    return written, warnings


def compare_controllers(generated: dict, reference: dict) -> list[str]:
    """Return the field-level differences between two ``.prbc`` documents.

    Written for checking generated controllers against the hand-built ones under
    ``VISSIM_previous/``.  Those files reflect an *older* Synchro timing plan for
    five of Chattanooga's six intersections, so a difference is not necessarily
    a defect -- it may simply mean the timings have been revised since.  The one
    intersection whose plan was unchanged (INTID 16) comes back clean.

    Args:
        generated: Output of :func:`build_controller`.
        reference: A ``.prbc`` document read from disk.

    Returns:
        One line per difference; empty when the two agree.
    """
    out: list[str] = []
    left, right = generated.get("Controller", {}), reference.get("Controller", {})

    for key in ("ExecutionFrequency", "OffsetReference"):
        if left.get(key) != right.get(key):
            out.append(f"{key}: {left.get(key)} vs {right.get(key)}")

    lp = (left.get("Patterns") or [{}])[0]
    rp = (right.get("Patterns") or [{}])[0]
    for key in ("CycleLength", "Offset", "MaxGreenMode", "PermissiveMode"):
        if lp.get(key) != rp.get(key):
            out.append(f"Pattern.{key}: {lp.get(key)} vs {rp.get(key)}")

    def by_id(items, key):
        return {item[key]: item for item in items or []}

    lg, rg = (by_id(left.get("VehicleSignalGroups"), "ID"),
              by_id(right.get("VehicleSignalGroups"), "ID"))
    if set(lg) != set(rg):
        out.append(f"signal groups: {sorted(lg)} vs {sorted(rg)}")
    for sg in sorted(set(lg) & set(rg)):
        for key in ("MinGreen", "MaxGreen1", "Yellow", "RedClearance",
                    "VehExtension", "MinRecall", "MaxRecall"):
            if lg[sg].get(key) != rg[sg].get(key):
                out.append(f"SG{sg}.{key}: {lg[sg].get(key)} vs {rg[sg].get(key)}")

    ls, rs = (by_id(lp.get("VehicleSignalGroupsInPattern"), "VehicleSignalGroup"),
              by_id(rp.get("VehicleSignalGroupsInPattern"), "VehicleSignalGroup"))
    for sg in sorted(set(ls) & set(rs)):
        for key in ("Split", "Coordinated"):
            if ls[sg].get(key) != rs[sg].get(key):
                out.append(f"SG{sg}.{key}: {ls[sg].get(key)} vs {rs[sg].get(key)}")

    def seq(doc):
        return [[ring["VehicleSignalGroups"] for ring in barrier["RingGroups"]]
                for barrier in doc.get("Sequence", {}).get("BarrierGroups", [])]

    if seq(left) != seq(right):
        out.append(f"sequence: {seq(left)} vs {seq(right)}")
    return out
