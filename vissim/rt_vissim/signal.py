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
"""Read Synchro UTDF signal timings into the scenario IR.

The Vissim counterpart of RealTwin's SUMO signal path, which turns the same
UTDF export into a NEMA ``tlLogic``.  Vissim keeps actuation instead: the
timings go to a Ring Barrier Controller, so nothing here collapses the plan to
fixed time.  :mod:`rt_vissim.rbc` writes the ``.prbc`` file from what this
produces.

Synchro's ``Phases`` section is a matrix -- one row per timing parameter, one
column ``D1``..``D8`` per phase -- and ``Timeplans`` is a flat
``RECORDNAME``/``DATA`` list per intersection.  The mapping onto RBC was derived
by reading a hand-built controller file against the UTDF it came from, and is
exact on the one intersection whose timings had not been revised since
(Chattanooga INTID 16, 30 of 30 fields).

Two parts of it are worth stating because they are not obvious:

**Split is not stored.**  Synchro records ``MaxGreen``, ``Yellow`` and
``AllRed``; the split is their sum.  The rings of a barrier must agree on that
sum, and the barriers must add to the cycle -- which is what
:func:`check_plan` tests, and the only check available on a corridor with no
reference controller to compare against.

**Yellow absorbs the rounding.**  ``MaxGreen`` carries tenths in Synchro
(6.3 s, 21.9 s) while the RBC editor writes whole seconds, so ``MaxGreen1`` is
rounded and the remainder goes to yellow, keeping the split exact.  Reproducing
that is what takes INTID 16 from approximately right to identical.
"""

from __future__ import annotations

import re
from pathlib import Path

from .ir import SignalGroupTiming, SignalPlan

#: Synchro ``Recall`` codes.
RECALL_MIN = 1
RECALL_MAX = 3

#: Phase columns Synchro writes, in order.
PHASE_COLUMNS = [f"D{i}" for i in range(1, 9)]


def _number(value) -> float | None:
    """Coerce a Synchro cell to a float, mapping blanks and text to ``None``."""
    if value is None:
        return None
    text = str(value).strip()
    if text in ("", "nan", "None"):
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_brp(value) -> tuple[int, int, int] | None:
    """Split a Synchro ``BRP`` code into barrier, ring and position.

    ``BRP`` is three digits: the barrier the phase sits in, the ring it runs in,
    and its position along that ring.  ``212`` is barrier 2, ring 1, position 2.

    Args:
        value: The raw cell, e.g. ``"212"``.

    Returns:
        ``(barrier, ring, position)``, or ``None`` when the cell is not a
        three-digit code.

    Examples:
        >>> parse_brp("111")
        (1, 1, 1)
        >>> parse_brp("212")
        (2, 1, 2)
    """
    number = _number(value)
    if number is None:
        return None
    text = str(int(number))
    if len(text) != 3:
        return None
    return int(text[0]), int(text[1]), int(text[2])


def coordinated_phases(reference_phase) -> set[int]:
    """Return the phases Synchro's ``Reference Phase`` names.

    The field concatenates the coordinated phase numbers, so ``206`` is phases 2
    and 6 -- the main-street through movements.  Zeros are separators, not
    phases.

    Args:
        reference_phase: The raw ``Reference Phase`` cell.

    Returns:
        The phase numbers, empty when the cell is blank.
    """
    if reference_phase is None:
        return set()
    digits = re.sub(r"\D", "", str(reference_phase).split(".")[0])
    return {int(d) for d in digits if d != "0"}


def read_synchro(path: str | Path) -> dict:
    """Parse a Synchro UTDF export.

    Delegates to RealTwin's own parser so both pipelines read the file the same
    way and a change there reaches both.

    Args:
        path: Path to the UTDF ``.csv``.

    Returns:
        ``{"Lanes": df, "Timeplans": df, "Phases": df}``.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ImportError: If the RealTwin parser cannot be imported.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Synchro UTDF file not found: {path}")

    from realtwin.func_lib._c_abstract_scenario.rt_demand_generation import (
        process_signal_from_utdf)
    return process_signal_from_utdf(str(path))


def build_signal_plan(synchro: dict, intid: str, *, junction_id: str | int = "",
                      sc_no: int = 0, name: str = "") -> tuple[SignalPlan | None,
                                                               list[str]]:
    """Build one intersection's plan from a parsed UTDF export.

    Args:
        synchro: Output of :func:`read_synchro`.
        intid: The Synchro ``INTID`` to build.
        junction_id: OpenDRIVE junction id this controller belongs to.
        sc_no: Vissim signal controller number.  Defaults to the ``INTID``,
            which keeps the Vissim model readable beside the Synchro one.
        name: Human readable intersection name.

    Returns:
        ``(plan, warnings)``; ``plan`` is ``None`` when the intersection has no
        usable phases.
    """
    warnings: list[str] = []
    phases = synchro.get("Phases")
    timeplans = synchro.get("Timeplans")
    if phases is None or timeplans is None:
        return None, [f"INTID {intid}: the UTDF export has no Phases/Timeplans."]

    rows = phases[phases["INTID"].astype(str) == str(intid)]
    if rows.empty:
        return None, [f"INTID {intid}: no phases in the UTDF export."]
    rows = rows.set_index("RECORDNAME")

    plan_rows = timeplans[timeplans["INTID"].astype(str) == str(intid)]
    settings = (plan_rows.set_index("RECORDNAME")["DATA"] if not plan_rows.empty
                else {})

    def setting(key):
        try:
            return settings.get(key)
        except AttributeError:  # pragma: no cover - empty timeplan block
            return None

    cycle = _number(setting("Cycle Length")) or 0.0
    offset = _number(setting("Offset")) or 0.0
    coordinated = coordinated_phases(setting("Reference Phase"))

    # Which phases serve a through movement, read from the Lanes block.
    #
    # Synchro carries a ``DualEntry`` row but leaves every cell blank on these
    # exports, so the value is not available and has to be chosen.  The manual
    # says dual entry is "often used for through-movement signal groups such
    # that if one signal group is called, the opposite through movement is
    # served as well", and the hand-built controllers set it on exactly the
    # phases that serve a through.  A split-phased side street runs its whole
    # approach, through included, on one phase, so that phase needs it too or
    # the approach waits for its own detector call every cycle.
    #
    # This is a modelling choice, not a conversion: nothing in the UTDF export
    # determines it.
    through_phases: set[int] = set()
    lanes = synchro.get("Lanes")
    if lanes is not None:
        block = lanes[lanes["INTID"].astype(str) == str(intid)]
        if not block.empty and "Phase1" in set(block["RECORDNAME"]):
            phase_row = block.set_index("RECORDNAME").loc["Phase1"]
            for column in phase_row.index:
                if len(str(column)) == 3 and str(column).endswith("T"):
                    number = _number(phase_row[column])
                    if number:
                        through_phases.add(int(number))

    def cell(record: str, column: str):
        if record not in rows.index or column not in rows.columns:
            return None
        return _number(rows.at[record, column])

    groups: list[SignalGroupTiming] = []
    for index, column in enumerate(PHASE_COLUMNS, start=1):
        if column not in rows.columns:
            continue
        max_green = cell("MaxGreen", column)
        yellow = cell("Yellow", column)
        all_red = cell("AllRed", column)
        # Synchro pads its matrix to eight columns; an unused phase is all zero.
        if not max_green:
            continue
        if yellow is None or all_red is None:
            warnings.append(f"INTID {intid} phase {index}: no yellow or all-red; "
                            "skipped.")
            continue

        brp = parse_brp(cell("BRP", column))
        if brp is None:
            warnings.append(f"INTID {intid} phase {index}: no BRP code, so its "
                            "place in the ring/barrier sequence is unknown; skipped.")
            continue
        barrier, ring, position = brp
        recall = cell("Recall", column) or 0

        groups.append(SignalGroupTiming(
            sg_no=index,
            phase=str(index),
            yellow=yellow,
            all_red=all_red,
            min_green=cell("MinGreen", column) or 0.0,
            max_green=max_green,
            veh_ext=cell("VehExt", column) or 0.0,
            ring=ring,
            barrier=barrier,
            position=position,
            split=max_green + yellow + all_red,
            coordinated=index in coordinated,
            min_recall=int(recall) == RECALL_MIN,
            max_recall=int(recall) == RECALL_MAX,
            dual_entry=index in through_phases,
            inhibit_max=bool(cell("InhibitMax", column)),
            start_up=index in coordinated,
        ))

    if not groups:
        return None, warnings + [f"INTID {intid}: no phases with a green time."]

    plan = SignalPlan(
        sc_no=sc_no or int(intid),
        junction_id=junction_id,
        synchro_intid=str(intid),
        name=name,
        cycle_time=cycle,
        offset=offset,
        controller_type="RBC",
        coordinated=bool(coordinated),
        signal_groups=groups,
    )
    return plan, warnings


def build_signal_plans(synchro: dict, junctions: dict[str, str], *,
                       names: dict[str, str] | None = None,
                       ) -> tuple[list[SignalPlan], list[str]]:
    """Build a plan for every signalised junction in the MatchupTable.

    Args:
        synchro: Output of :func:`read_synchro`.
        junctions: ``{junction id: Synchro INTID}``, from
            :meth:`rt_vissim.matchup.MatchupTable.signalized_junctions`.
        names: Optional ``{junction id: intersection name}`` for labelling.

    Returns:
        ``(plans, warnings)``.
    """
    names = names or {}
    plans: list[SignalPlan] = []
    warnings: list[str] = []
    for junction_id, intid in sorted(junctions.items(), key=lambda kv: str(kv[0])):
        plan, notes = build_signal_plan(synchro, intid, junction_id=junction_id,
                                        name=names.get(str(junction_id), ""))
        warnings.extend(notes)
        if plan is None:
            continue
        warnings.extend(f"junction {junction_id}: {w}" for w in check_plan(plan))
        plans.append(plan)
    return plans, warnings


def sequence(plan: SignalPlan) -> list[list[list[int]]]:
    """Return the barrier / ring / signal-group structure of a plan.

    Args:
        plan: A :class:`~rt_vissim.ir.SignalPlan`.

    Returns:
        ``[[[sg, ...], ...], ...]`` -- barriers outermost, then rings, then the
        signal groups in position order.
    """
    barriers: dict[int, dict[int, list[SignalGroupTiming]]] = {}
    for group in plan.signal_groups:
        barriers.setdefault(group.barrier, {}).setdefault(group.ring, []).append(group)
    return [[[g.sg_no for g in sorted(rings[r], key=lambda g: g.position)]
             for r in sorted(rings)]
            for _, rings in sorted(barriers.items())]


def check_plan(plan: SignalPlan) -> list[str]:
    """Check a plan against the arithmetic a ring-barrier controller requires.

    This is the check that works on a corridor with no reference controller to
    compare against.  Two things must hold, and both are the modeller's problem
    rather than ours when they do not:

    * within a barrier, every ring must serve the same total time, since the
      rings cross the barrier together;
    * the barriers must add up to the cycle length.

    Args:
        plan: A :class:`~rt_vissim.ir.SignalPlan`.

    Returns:
        Warnings; empty when the plan is arithmetically sound.
    """
    warnings: list[str] = []
    splits = {g.sg_no: g.split for g in plan.signal_groups}

    total = 0.0
    for index, barrier in enumerate(sequence(plan), start=1):
        sums = [round(sum(splits.get(sg, 0.0) for sg in ring), 1) for ring in barrier]
        if len(set(sums)) > 1:
            warnings.append(
                f"barrier {index} rings serve different totals {sums} s; the rings "
                "cross a barrier together, so they must agree.")
        total += max(sums) if sums else 0.0

    if plan.cycle_time and round(total, 1) != round(plan.cycle_time, 1):
        warnings.append(f"barriers total {total:.1f} s against a cycle length of "
                        f"{plan.cycle_time:.1f} s.")
    return warnings


def summarise(plans: list[SignalPlan]) -> str:
    """Return a one-line summary of a set of plans."""
    if not plans:
        return "no signal plans"
    groups = sum(len(p.signal_groups) for p in plans)
    cycles = sorted({p.cycle_time for p in plans})
    return (f"{len(plans)} controllers, {groups} signal groups, "
            f"cycle {'/'.join(f'{c:.0f}' for c in cycles)} s")
