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
"""Work out where the signal heads and detectors go.

:mod:`rt_vissim.signal` turns Synchro's timings into a controller; this decides
what the controller is wired to.  Both read the same UTDF export, but this one
also needs the network, because a head and a detector are physical objects on
links.

Signal heads go on the movement's **connector**.  A connector carries exactly
one movement, so the signal group is unambiguous, and nothing has to be assumed
about which end Vissim numbers lanes from.  The manual uses the same technique
to give a turn its own signal group, and connectors here are short -- a median
of 1.6 m on Chattanooga -- so the head still sits at the stop line.

Detectors cannot use that trick.  A call detector has to be **upstream** of the
stop line, on the lanes that serve the movement, because a connector is already
past the point where a call is any use.  Those lanes come from the connector's
``FromLanes``, which records which lanes of the approach it leaves from -- read
from the network rather than assumed.  On Chattanooga that returns lanes 2 and 3
for the two-lane left bay at link 33 and lane 1 for the shared through/right,
matching what Synchro says without anyone having to decide a convention.

Synchro's distances are in feet on these networks: 12 ft lanes, 40 mph speeds,
storage in feet.
"""

from __future__ import annotations

from .ir import Detector, SignalHead, SignalPlan
from .signal import _number

#: Synchro exports these networks in US customary units.
FEET_TO_METRES = 0.3048

#: Keep the detector this far from the start of the link.
DETECTOR_MARGIN = 1.0

#: Smallest detector worth creating; below this it barely spans a vehicle.
MIN_DETECTOR_LENGTH = 2.0


def _movement_row(lanes, intid: str, record: str) -> dict:
    """Return one ``Lanes`` record for an intersection as ``{movement: value}``.

    Args:
        lanes: The ``Lanes`` frame from :func:`rt_vissim.signal.read_synchro`.
        intid: Synchro ``INTID``.
        record: The ``RECORDNAME`` to pull, e.g. ``"Phase1"``.

    Returns:
        ``{movement code: raw value}``, empty when the record is absent.
    """
    rows = lanes[lanes["INTID"].astype(str) == str(intid)]
    if rows.empty or record not in set(rows["RECORDNAME"]):
        return {}
    row = rows.set_index("RECORDNAME").loc[record]
    return {c: row[c] for c in row.index
            if len(str(c)) == 3 and str(c)[-1] in "RTLU"}


def group_for_code(table: dict, code: str, groups: set) -> int | None:
    """Return the signal group a Synchro phase record names for one movement."""
    try:
        number = int(float(table.get(code)))
    except (TypeError, ValueError):
        return None
    return number if number in groups else None


def _shares_lane(shared: dict, bound: str, side: str) -> bool:
    """Whether the through lane of ``bound`` also serves the left or right turn.

    Synchro's ``Shared`` code on the through movement is 1 for a shared left,
    2 for a shared right and 3 for both.

    Args:
        shared: The ``Shared`` record, ``{movement: code}``.
        bound: Two-letter bound, e.g. ``"NB"``.
        side: ``"left"`` or ``"right"``.

    Returns:
        Whether that turn shares the through lane.
    """
    try:
        code = int(float(shared.get(bound + "T")))
    except (TypeError, ValueError):
        return False
    return code in ((1, 3) if side == "left" else (2, 3))


def _connector_index(links: dict) -> dict:
    """Return ``{(from link, to link): connector}`` for every connector."""
    return {(ln.from_link, ln.to_link): ln
            for ln in links.values() if ln.is_connector}


def _movements(matchup, junction_id) -> list:
    """Return the MatchupTable rows for one junction that carry a Synchro code."""
    frame = matchup.df
    rows = frame[frame["JunctionID_OpenDrive"].astype(str) == str(junction_id)]
    out = []
    for row in rows.itertuples(index=False):
        code = getattr(row, "Turn_Synchro", None)
        if code and str(code) not in ("None", "nan"):
            out.append((str(code), row))
    return out


def _first_connector(row, connectors: dict):
    """Return the connector a movement leaves its approach on."""
    from_link = int(float(row.FromLinkNo_Vissim))
    internal = str(getattr(row, "InternalLinks_Vissim", "") or "").split()
    first = int(internal[0]) if internal else int(float(row.ToLinkNo_Vissim))
    return from_link, connectors.get((from_link, first))


def build_signal_heads(matchup, links: dict, synchro: dict,
                       plans: list[SignalPlan],
                       ) -> tuple[list[SignalHead], list[str]]:
    """Build one signal head per signalised movement.

    Which group the head shows depends on how Synchro serves the turn:

    * ``Phase1`` only -- protected, and the head shows that group;
    * ``Phase1`` **and** ``PermPhase1`` -- protected-permissive, so the permitted
      phase becomes the head's *Or signal group* and the head is green whenever
      either group is;
    * ``PermPhase1`` only -- permitted throughout, so the head shows the opposing
      through group and the turn has to yield to it.

    The last two both need a conflict area before the turn actually yields;
    that is a separate stage, and the warnings say how many are waiting on it.

    Args:
        matchup: A :class:`rt_vissim.matchup.MatchupTable`.
        links: Output of :func:`rt_vissim.network.read_links_csv`.
        synchro: Output of :func:`rt_vissim.signal.read_synchro`.
        plans: Controllers from :func:`rt_vissim.signal.build_signal_plans`.

    Returns:
        ``(signal_heads, warnings)``.
    """
    lanes_table = synchro.get("Lanes")
    if lanes_table is None:
        return [], ["The UTDF export has no Lanes section; no signal heads built."]

    connectors = _connector_index(links)
    heads: list[SignalHead] = []
    warnings: list[str] = []
    inherited_notes: list[str] = []

    for plan in sorted(plans, key=lambda p: str(p.junction_id)):
        protected = _movement_row(lanes_table, plan.synchro_intid, "Phase1")
        permitted = _movement_row(lanes_table, plan.synchro_intid, "PermPhase1")
        shared = _movement_row(lanes_table, plan.synchro_intid, "Shared")
        groups = {g.sg_no for g in plan.signal_groups}

        for code, row in _movements(matchup, plan.junction_id):
            from_link, connector = _first_connector(row, connectors)
            if connector is None:
                warnings.append(f"junction {plan.junction_id} {code}: no connector "
                                f"leaves link {from_link}; no signal head placed.")
                continue

            def group_for(table) -> int | None:
                try:
                    number = int(float(table.get(code)))
                except (TypeError, ValueError):
                    return None
                return number if number in groups else None

            protected_sg = group_for(protected)
            permitted_sg = group_for(permitted)

            # Synchro names a phase per lane group, not per movement, so two
            # movements are left without one and have to inherit.
            inherited = ""
            if protected_sg is None and permitted_sg is None:
                bound = code[:2]
                if code.endswith("R") and _shares_lane(shared, bound, "right"):
                    # A right turn with no lane of its own runs in the through
                    # lane, so it is served by the through phase.
                    protected_sg = group_for_code(protected, bound + "T", groups)
                    permitted_sg = group_for_code(permitted, bound + "T", groups)
                    inherited = f"{bound}T (shared lane)"
                elif code.endswith("U"):
                    # A U-turn is made from the left-turn bay and Synchro has no
                    # column for it, so it takes the left turn's phase.
                    protected_sg = group_for_code(protected, bound + "L", groups)
                    permitted_sg = group_for_code(permitted, bound + "L", groups)
                    inherited = f"{bound}L (same bay)"

            if protected_sg is None and permitted_sg is None:
                warnings.append(f"junction {plan.junction_id} {code}: Synchro gives "
                                "it no phase and none can be inherited, so it is "
                                "left unsignalised.")
                continue
            if inherited:
                inherited_notes.append(f"{plan.junction_id} {code} <- {inherited}")

            secondary = None
            if (protected_sg is not None and permitted_sg is not None
                    and permitted_sg != protected_sg):
                secondary = permitted_sg

            heads.append(SignalHead(
                sc_no=plan.sc_no,
                sg_no=protected_sg if protected_sg is not None else permitted_sg,
                junction_id=plan.junction_id,
                from_link_no=from_link,
                to_link_no=int(float(row.ToLinkNo_Vissim)),
                connector_no=connector.no,
                pos=0.0,
                scnd_sg_no=secondary,
                movement=code,
                turn=str(getattr(row, "Turn", "")),
                permissive_only=protected_sg is None,
            ))

    if inherited_notes:
        warnings.append(f"{len(inherited_notes)} movements have no phase of their "
                        "own in Synchro and inherit one: "
                        f"{'; '.join(inherited_notes)}.")

    both = sum(1 for h in heads if h.scnd_sg_no is not None)
    only = sum(1 for h in heads if h.permissive_only)
    if both:
        warnings.append(f"{both} turns are protected-permissive and carry an Or "
                        "signal group; they yield only once a conflict area exists.")
    if only:
        warnings.append(f"{only} turns are permitted but never protected; they run "
                        "on the opposing through phase and need a conflict area.")
    return heads, warnings


def detector_placement(approach_length: float, setback: float, length: float,
                       ) -> tuple[float, float, bool]:
    """Fit a detector onto an approach, shrinking it when it will not fit.

    Synchro asks for a detector of ``length`` whose downstream edge sits
    ``setback`` back from the stop line, which is the downstream end of the
    approach.  Six of Chattanooga's twenty-two signalised approaches are shorter
    than the 30.5 m that Synchro's usual 50 ft + 50 ft needs, one of them only
    7.5 m.  Dropping the detector would leave its phase with no call at all, so
    it is shrunk instead and the caller says so.

    Args:
        approach_length: Length of the approach link, metres.
        setback: Distance from the stop line to the detector's downstream edge.
        length: Detector length Synchro asks for, metres.

    Returns:
        ``(pos, length, shortened)`` -- position measured from the start of the
        link, the length actually used, and whether it had to be reduced.
    """
    pos = approach_length - setback - length
    if pos >= DETECTOR_MARGIN:
        return round(pos, 2), round(length, 2), False

    # Take the space that is left between the margin and the setback.
    usable = approach_length - DETECTOR_MARGIN - setback
    if usable < MIN_DETECTOR_LENGTH:
        # Not even room for the setback: give up the setback before the detector.
        usable = approach_length - DETECTOR_MARGIN
    fitted = max(MIN_DETECTOR_LENGTH, min(length, usable))
    pos = max(DETECTOR_MARGIN, approach_length - setback - fitted)
    if pos + fitted > approach_length:
        fitted = max(MIN_DETECTOR_LENGTH, approach_length - pos)
    return round(pos, 2), round(fitted, 2), True


def build_detectors(matchup, links: dict, synchro: dict,
                    plans: list[SignalPlan],
                    ) -> tuple[list[Detector], list[str]]:
    """Build the vehicle detectors for every signalised junction.

    One detector per approach lane, on the lanes the movement's connector leaves
    from, calling the phase Synchro's ``DetectPhase1`` names.  Port numbers are
    handed out in order and are what the ``.prbc`` refers back to.

    Args:
        matchup: A :class:`rt_vissim.matchup.MatchupTable`.
        links: Output of :func:`rt_vissim.network.read_links_csv`.
        synchro: Output of :func:`rt_vissim.signal.read_synchro`.
        plans: Controllers from :func:`rt_vissim.signal.build_signal_plans`.

    Returns:
        ``(detectors, warnings)``.
    """
    lanes_table = synchro.get("Lanes")
    if lanes_table is None:
        return [], ["The UTDF export has no Lanes section; no detectors built."]

    connectors = _connector_index(links)
    detectors: list[Detector] = []
    warnings: list[str] = []
    shortened: set[str] = set()

    for plan in sorted(plans, key=lambda p: str(p.junction_id)):
        setbacks = _movement_row(lanes_table, plan.synchro_intid, "FirstDetect")
        sizes = _movement_row(lanes_table, plan.synchro_intid, "DetectSize1")
        phases = _movement_row(lanes_table, plan.synchro_intid, "DetectPhase1")
        groups = {g.sg_no for g in plan.signal_groups}
        taken: set[tuple[int, int]] = set()

        for code, row in _movements(matchup, plan.junction_id):
            try:
                sg_no = int(float(phases.get(code)))
            except (TypeError, ValueError):
                continue
            if sg_no not in groups:
                continue

            from_link, connector = _first_connector(row, connectors)
            approach = links.get(from_link)
            if connector is None or approach is None or approach.length <= 0:
                continue

            wanted_length = (_number(sizes.get(code)) or 0.0) * FEET_TO_METRES
            if wanted_length <= 0:
                continue
            setback = (_number(setbacks.get(code)) or 0.0) * FEET_TO_METRES

            pos, length, was_short = detector_placement(
                approach.length, setback, wanted_length)
            if was_short:
                shortened.add(f"link {from_link} ({approach.length:.1f} m)")

            for lane in (connector.from_lanes or [1]):
                if (from_link, lane) in taken:
                    continue  # a shared lane is detected once, by whichever
                              # movement claims it first
                taken.add((from_link, lane))
                # The port number is how the .prbc finds this detector, and its
                # VehicleDetectors are keyed on the signal group, so the port
                # must be the group rather than a running count.  Numbering them
                # sequentially left five of Chattanooga's six controllers with no
                # calls at all: their coordinated phases sat green forever while
                # every actuated phase stayed red.  Several lanes calling one
                # group share a port, which is ordinary multi-lane detection.
                detectors.append(Detector(
                    sc_no=plan.sc_no, sg_no=sg_no, junction_id=plan.junction_id,
                    link_no=from_link, lane=lane, pos=pos, length=length,
                    port_no=sg_no, movement=code, shortened=was_short))

    if shortened:
        warnings.append(
            f"{len(shortened)} approaches are too short for Synchro's detector "
            f"layout, so their detectors were shrunk to fit: "
            f"{', '.join(sorted(shortened))}. A shorter presence zone holds a "
            "call for less time.")
    return detectors, warnings


def summarise(heads: list[SignalHead], detectors: list[Detector]) -> str:
    """Return a one-line summary of what will be placed."""
    protected = sum(1 for h in heads if not h.permissive_only and h.scnd_sg_no is None)
    both = sum(1 for h in heads if h.scnd_sg_no is not None)
    permitted = sum(1 for h in heads if h.permissive_only)
    short = sum(1 for d in detectors if d.shortened)
    return (f"{len(heads)} signal heads ({protected} protected, {both} "
            f"protected-permissive, {permitted} permitted only), "
            f"{len(detectors)} detectors ({short} shortened)")


def rtor_allowed(matchup, synchro: dict, plans: list[SignalPlan]) -> dict:
    """Return which right turns Synchro permits on red.

    Synchro records this per movement in the ``Lanes`` section as ``Allow
    RTOR``, 1 for permitted.  At Chattanooga's INTID 16 every movement carries
    a 1, so every right turn there may go on red.

    Args:
        matchup: A :class:`rt_vissim.matchup.MatchupTable`.
        synchro: Output of :func:`rt_vissim.signal.read_synchro`.
        plans: Controllers from :func:`rt_vissim.signal.build_signal_plans`.

    Returns:
        ``{(junction id, movement code): allowed}``.
    """
    lanes_table = synchro.get("Lanes")
    if lanes_table is None:
        return {}

    out: dict = {}
    for plan in plans:
        allow = _movement_row(lanes_table, plan.synchro_intid, "Allow RTOR")
        for code, _row in _movements(matchup, plan.junction_id):
            value = _number(allow.get(code))
            out[(str(plan.junction_id), code)] = bool(value)
    return out
