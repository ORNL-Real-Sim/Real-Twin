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

Everything here is decided **per lane**, because a lane can show only one
signal.  Vissim's one per-head discriminator on a shared lane is the vehicle
class (manual p. 652); there is none by turning movement, so two turns out of
one lane cannot be given different indications -- nor can they be in reality.
Every yield is therefore a conflict area, never a head.

Signal heads go on the **link**, at the stop line, one per lane.  The manual
prefers it -- "Combined right turning and straight lanes ... it would be better
to place them on the link rather than on the connector" (p. 634) -- and it
avoids the doubled heads that appear when two connectors leave overlapping sets
of lanes, as Chattanooga's link 35 does with its triple right turn.  Heads go on
a connector only for the case the manual reserves it for: a turn needing its own
signal group, on "a connector not used by vehicles traveling straight ahead"
(p. 653).

Which lanes serve a movement comes from the connectors' ``FromLanes``, read from
the network rather than assumed, and from **every** connector the movement uses:
a turn made from two lanes has one connector per lane.  Taking only the first
left link 17 lane 2 crossing junction 9 with no signal at all.

Detectors are per lane too, **upstream** of the stop line, built from the same
lane table as the heads so neither can cover a lane the other misses.

Synchro's distances are in feet on these networks: 12 ft lanes, 40 mph speeds,
storage in feet.
"""

from __future__ import annotations

from .ir import Detector, LaneControl, SignalHead, SignalPlan
from .network import MAX_INTERNAL_HOPS
from .signal import _number

#: Which movement governs a lane when the movements on it disagree about the
#: signal group.  A turn cannot hold the through traffic beside it in the same
#: lane, so the through wins; the rest are settled by conflict areas.
TURN_PRECEDENCE = {"T": 3, "R": 2, "L": 1, "U": 0}

#: How far upstream of the stop line the head sits.
#:
#: The stop line is where the **connectors leave** the approach, which is not
#: the end of the link: on Chattanooga they leave about 0.2 m before it.  A
#: head past that point is never reached -- a turning vehicle is already on its
#: connector -- and every one of the 69 heads was bypassed that way, so traffic
#: drove through red everywhere.
#:
#: 1 m is Vissim's own figure for the same job.  Placing signal heads for a
#: node, it "inserts a signal head in the node for turns 1 m before the first
#: conflict area", and where there is less than 1 m of room it "inserts the
#: signal head further upstream" (manual p. 642).
HEAD_CLEARANCE = 1.0

#: Synchro exports these networks in US customary units.
FEET_TO_METRES = 0.3048

#: Keep the detector this far from the start of the link.
DETECTOR_MARGIN = 1.0

#: Smallest detector worth creating.  A presence detector has to hold a waiting
#: car, so the floor is a car length rather than a token couple of metres --
#: shorter than this and a stopped vehicle can sit past it and drop the call.
#: Only used where the approach has the room; a shorter approach gets whatever
#: fits.
MIN_DETECTOR_LENGTH = 6.0

#: Length to fall back on when Synchro says a movement has a detector but
#: leaves its size at zero and the intersection has no other size to copy.
#: 50 ft is Synchro's own default and what every other movement here uses.
DEFAULT_DETECTOR_FEET = 50.0

#: How far back from the stop line the detector's downstream edge sits.
#: Synchro's ``FirstDetect`` is 50 ft on these networks, which puts the detector
#: 15 m upstream -- far enough back that a queue can form between it and the
#: stop bar without the controller seeing it.  A stop-bar presence detector
#: belongs at the line, so the setback is set here rather than taken from
#: Synchro.  It also buys back most of the room the short approaches lacked.
#:
#: Not flush against the line, though: the signal heads sit at the stop bar, so
#: a detector ending there draws on top of them and neither can be picked out in
#: the network editor.  A couple of metres clears the heads while still
#: detecting a vehicle waiting at the line.
STOP_BAR_SETBACK = 2.0


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


def _outgoing(links: dict) -> dict[int, list]:
    """Return ``{link number: [connectors leaving it]}``."""
    out: dict[int, list] = {}
    for ln in links.values():
        if ln.is_connector and ln.from_link is not None:
            out.setdefault(int(ln.from_link), []).append(ln)
    return out


def _exits_of(connector, links: dict, outgoing: dict) -> set[int]:
    """Return every link outside the junction that this connector leads to.

    A movement is a path -- approach, connector, one or more internal links,
    connector, exit -- so the exit is found by following the connector forward
    until the walk leaves the junction interior.

    Args:
        connector: The connector to start from.
        links: Output of :func:`rt_vissim.network.read_links_csv`.
        outgoing: Output of :func:`_outgoing`.

    Returns:
        The link numbers reached, normally one.
    """
    exits: set[int] = set()
    seen: set[int] = set()
    frontier = [(connector, 0)]
    while frontier:
        conn, depth = frontier.pop()
        if conn.to_link is None:
            continue
        target = links.get(int(conn.to_link))
        if target is None:
            continue
        if not target.is_internal:
            exits.add(int(target.no))
            continue
        if target.no in seen or depth >= MAX_INTERNAL_HOPS:
            continue
        seen.add(target.no)
        frontier.extend((nxt, depth + 1) for nxt in outgoing.get(int(target.no), ()))
    return exits


def _exit_index(links: dict, outgoing: dict) -> dict[int, set[int]]:
    """Return ``{connector number: exits it reaches}``, computed once."""
    return {int(conn.no): _exits_of(conn, links, outgoing)
            for conns in outgoing.values() for conn in conns}


def _connectors_for(row, outgoing: dict, exits: dict) -> tuple[int, list]:
    """Return every connector a movement may leave its approach on.

    A turn served by more than one lane has one connector **per lane**, all
    reaching the same exit.  Vissim controls connectors, so each of them needs
    its own signal head; taking only the first leaves the other lanes crossing
    the junction unsignalised.  Chattanooga's link 17 is the case that matters:
    lanes 2 and 3 both run through and left, and only lane 3's pair was found.

    The MatchupTable cannot answer this, because it holds one row per movement
    and :func:`rt_vissim.network.trace_movements` deduplicates on the exit --
    correct for a movement table, lossy for lane-level control.  So the
    connectors are matched on ``(approach, exit)`` against the network instead.

    Args:
        row: A MatchupTable row for one movement.
        outgoing: Output of :func:`_outgoing`.
        exits: Output of :func:`_exit_index`.

    Returns:
        ``(approach link number, connectors ordered by lane)``.
    """
    from_link = int(float(row.FromLinkNo_Vissim))
    to_link = int(float(row.ToLinkNo_Vissim))
    found = [conn for conn in outgoing.get(from_link, ())
             if to_link in exits.get(int(conn.no), ())]
    found.sort(key=lambda conn: (min(conn.from_lanes) if conn.from_lanes else 0,
                                 int(conn.no)))
    return from_link, found


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


def build_lane_control(matchup, links: dict, synchro: dict,
                       plans: list[SignalPlan],
                       ) -> tuple[list[LaneControl], list[str]]:
    """Work out what the signal does to each lane of each signalised approach.

    Heads and detectors are both derived from this, so neither can cover a lane
    the other misses.  They used to resolve connectors independently, and
    Chattanooga's link 17 lane 2 lost its head *and* its detector that way.

    Which group a movement runs on depends on how Synchro serves the turn:

    * ``Phase1`` only -- protected, and the head shows that group;
    * ``Phase1`` **and** ``PermPhase1`` -- protected-permissive, so the permitted
      phase becomes the head's *Or signal group* and the head is green whenever
      either group is;
    * ``PermPhase1`` only -- permitted throughout, so the head shows the opposing
      through group and the turn has to yield to it.

    The last two both need a conflict area before the turn actually yields;
    that is a separate stage, and the warnings say how many are waiting on it.

    A lane carrying several movements gets one entry, because a lane can show
    only one indication.  Where those movements disagree on the group, the
    through governs and the rest are reported: no signal can hold the through
    and release the left in the same lane, so the conflict area has to.

    Args:
        matchup: A :class:`rt_vissim.matchup.MatchupTable`.
        links: Output of :func:`rt_vissim.network.read_links_csv`.
        synchro: Output of :func:`rt_vissim.signal.read_synchro`.
        plans: Controllers from :func:`rt_vissim.signal.build_signal_plans`.

    Returns:
        ``(lane_controls, warnings)``.
    """
    lanes_table = synchro.get("Lanes")
    if lanes_table is None:
        return [], ["The UTDF export has no Lanes section; no signal control built."]

    outgoing = _outgoing(links)
    exits = _exit_index(links, outgoing)
    warnings: list[str] = []
    inherited_notes: list[str] = []
    multi_lane: list[str] = []
    #: ``(approach link, lane) -> [one entry per movement served from it]``
    per_lane: dict[tuple[int, int], list[dict]] = {}

    for plan in sorted(plans, key=lambda p: str(p.junction_id)):
        protected = _movement_row(lanes_table, plan.synchro_intid, "Phase1")
        permitted = _movement_row(lanes_table, plan.synchro_intid, "PermPhase1")
        shared = _movement_row(lanes_table, plan.synchro_intid, "Shared")
        groups = {g.sg_no for g in plan.signal_groups}

        for code, row in _movements(matchup, plan.junction_id):
            from_link, movement_connectors = _connectors_for(row, outgoing, exits)
            if not movement_connectors:
                warnings.append(f"junction {plan.junction_id} {code}: no connector "
                                f"leaves link {from_link}; no signal head placed.")
                continue
            if len(movement_connectors) > 1:
                lanes = sorted({lane for conn in movement_connectors
                                for lane in (conn.from_lanes or [])})
                multi_lane.append(f"{plan.junction_id} {code} on lanes "
                                  f"{','.join(str(x) for x in lanes)}")

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

            # Record the movement against every lane it is made from, so a turn
            # served by two lanes reaches both.  Taking only the first left
            # Chattanooga's link 17 lane 2 crossing junction 9 unsignalised.
            for connector in movement_connectors:
                for lane in (connector.from_lanes or [1]):
                    per_lane.setdefault((from_link, int(lane)), []).append(dict(
                        code=code, sg=protected_sg if protected_sg is not None
                        else permitted_sg, scnd=secondary, perm_sg=permitted_sg,
                        plan=plan, connector=int(connector.no),
                        turn=str(getattr(row, "Turn", "")),
                        permissive_only=protected_sg is None))

    controls, warnings = _resolve_lanes(per_lane, warnings)

    if multi_lane:
        warnings.append(f"{len(multi_lane)} movements are served by more than one "
                        f"lane and get a head on each: {'; '.join(multi_lane)}.")
    if inherited_notes:
        warnings.append(f"{len(inherited_notes)} movements have no phase of their "
                        "own in Synchro and inherit one: "
                        f"{'; '.join(inherited_notes)}.")
    return controls, warnings


def _resolve_lanes(per_lane: dict, warnings: list[str],
                   ) -> tuple[list[LaneControl], list[str]]:
    """Reduce the movements on each lane to the one indication it can show.

    A lane shows one signal.  Vissim's only per-head discriminator on a shared
    lane is the vehicle class, so two turns out of one lane cannot be given
    different indications -- nor can they be in reality.

    Args:
        per_lane: ``{(link, lane): [movement entries]}``.
        warnings: Warnings collected so far, appended to.

    Returns:
        ``(lane_controls, warnings)``.
    """
    controls: list[LaneControl] = []
    unshowable: list[str] = []

    for (link_no, lane), entries in sorted(per_lane.items()):
        plan = entries[0]["plan"]
        variants = {(e["sg"], e["scnd"]) for e in entries}
        mixed = len(variants) > 1
        if not mixed:
            sg_no, scnd = variants.pop()
        else:
            # The through governs the lane; a turn cannot hold it back.
            lead = max(entries,
                       key=lambda e: TURN_PRECEDENCE.get(str(e["code"])[-1], -1))
            sg_no, scnd = lead["sg"], lead["scnd"]
            for entry in entries:
                if entry["sg"] == sg_no:
                    continue
                if entry["perm_sg"] == sg_no:
                    # An ordinary permitted turn from a shared lane: it runs on
                    # the governing phase and the conflict area holds it.
                    unshowable.append(
                        f"{plan.junction_id} {entry['code']} shares link {link_no} "
                        f"lane {lane}, so its phase {entry['sg']} cannot be shown; "
                        f"it runs permitted on {sg_no}")
                else:
                    warnings.append(
                        f"junction {plan.junction_id} {entry['code']} is protected "
                        f"on group {entry['sg']} but shares link {link_no} lane "
                        f"{lane} with {lead['code']} on group {sg_no}. No signal "
                        "can hold one and release the other from one lane, so it "
                        f"runs on {sg_no} and only the conflict area keeps it "
                        "safe. Check the Synchro Lanes record against the network.")

        controls.append(LaneControl(
            sc_no=plan.sc_no, sg_no=sg_no, junction_id=plan.junction_id,
            link_no=link_no, lane=lane, scnd_sg_no=scnd,
            movements=tuple(e["code"] for e in entries),
            connectors=tuple(e["connector"] for e in entries),
            turns=tuple(e["turn"] for e in entries),
            permissive_only=all(e["permissive_only"] for e in entries),
            mixed=mixed))

    if unshowable:
        warnings.append(f"{len(unshowable)} turns share a lane with a movement on "
                        f"another phase: {'; '.join(unshowable)}.")
    return controls, warnings


def stop_line(control: LaneControl, links: dict, length: float,
              guessed: list[str] | None = None) -> float:
    """Return where this lane's stop line is, measured along the approach.

    It is the point where the lane's connectors leave the link, **not** the end
    of the link: on Chattanooga they leave about 0.2 m before it.  Anything
    placed downstream of this is never reached, because a turning vehicle is on
    its connector by then.

    Args:
        control: The lane being signalised.
        links: Output of :func:`rt_vissim.network.read_links_csv`.
        length: Length of the approach link, metres, used as a fallback.
        guessed: Appended to when no connector position is recorded, so the
            caller can report how many were placed on the fallback.

    Returns:
        Distance in metres from the start of the approach link.
    """
    departures = [links[no].from_pos for no in control.connectors
                  if no in links and links[no].from_pos is not None]
    if departures:
        return min(departures)
    # No recorded departure: fall back to the end of the link and say so,
    # because this is the assumption that put every head past the diverge.
    if guessed is not None:
        guessed.append(f"link {control.link_no} lane {control.lane}")
    return length


def stop_line_position(control: LaneControl, links: dict, length: float,
                       guessed: list[str]) -> float:
    """Return where the head goes: ``HEAD_CLEARANCE`` upstream of the stop line."""
    return round(max(0.0, stop_line(control, links, length, guessed)
                     - HEAD_CLEARANCE), 2)


def build_signal_heads(matchup, links: dict, synchro: dict,
                       plans: list[SignalPlan],
                       ) -> tuple[list[SignalHead], list[str]]:
    """Build one signal head per lane of each signalised approach.

    The head goes on the **link**, at the stop line, which is what the manual
    prefers: "Combined right turning and straight lanes ... it would be better
    to place them on the link rather than on the connector" (p. 634).  It is
    also what a driver sees -- one lane, one indication -- and it avoids the
    doubled heads that appear when two connectors leave overlapping sets of
    lanes, as Chattanooga's link 35 does with its triple right turn.

    One record is one VISSIM signal head, so the length of the result is the
    number of heads that will appear in the network.

    Args:
        matchup: A :class:`rt_vissim.matchup.MatchupTable`.
        links: Output of :func:`rt_vissim.network.read_links_csv`.
        synchro: Output of :func:`rt_vissim.signal.read_synchro`.
        plans: Controllers from :func:`rt_vissim.signal.build_signal_plans`.

    Returns:
        ``(signal_heads, warnings)``.
    """
    controls, warnings = build_lane_control(matchup, links, synchro, plans)
    heads: list[SignalHead] = []
    guessed: list[str] = []

    for control in controls:
        approach = links.get(control.link_no)
        length = approach.length if approach is not None else 0.0
        heads.append(SignalHead(
            sc_no=control.sc_no,
            sg_no=control.sg_no,
            junction_id=control.junction_id,
            from_link_no=control.link_no,
            link_no=control.link_no,
            lane=control.lane,
            pos=stop_line_position(control, links, length, guessed),
            scnd_sg_no=control.scnd_sg_no,
            movement="/".join(dict.fromkeys(control.movements)),
            turn="/".join(dict.fromkeys(t for t in control.turns if t)),
            permissive_only=control.permissive_only,
        ))

    if guessed:
        warnings.append(
            f"{len(guessed)} heads were placed at the end of their approach "
            "because no connector departure position was recorded, so they may "
            "sit downstream of the stop line and be driven through: "
            f"{', '.join(guessed[:6])}"
            f"{' ...' if len(guessed) > 6 else ''}. Re-run stage 1 to record "
            "FromPos in the links CSV.")

    both = sum(1 for h in heads if h.scnd_sg_no is not None)
    only = sum(1 for h in heads if h.permissive_only)
    if both:
        warnings.append(f"{both} lanes are protected-permissive and carry an Or "
                        "signal group; they yield only once a conflict area exists.")
    if only:
        warnings.append(f"{only} lanes are permitted but never protected; they run "
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

    A detector shorter than a car is raised to ``MIN_DETECTOR_LENGTH`` wherever
    the approach has the room.  Synchro asks for 6 ft on two of Chattanooga's
    approaches -- an advance detector, meant to sit 224 ft upstream and count
    vehicles passing, not to hold one at the line.  Placed at the stop bar it
    would drop the call as soon as a car crept past it.

    Returns:
        ``(pos, length, shortened)`` -- position measured from the start of the
        link, the length actually used, and whether it had to be reduced.
    """
    length = max(length, MIN_DETECTOR_LENGTH)
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
    # The floor is what a car needs, not what the link has: on an approach too
    # short even for that, the detector ends at the stop line rather than
    # running off the end of it.
    fitted = min(fitted, approach_length - pos)
    return round(pos, 2), round(fitted, 2), True


def build_detectors(matchup, links: dict, synchro: dict,
                    plans: list[SignalPlan],
                    ) -> tuple[list[Detector], list[str]]:
    """Build the vehicle detectors for every signalised junction.

    One detector per lane, upstream of the stop line, calling the phase
    Synchro's ``DetectPhase1`` names.  Built from the same lane table as the
    heads, so a lane cannot be signalised without also being detected unless
    Synchro genuinely asks for no detector there.

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

    controls, _ = build_lane_control(matchup, links, synchro, plans)
    detectors: list[Detector] = []
    warnings: list[str] = []
    shortened: set[str] = set()

    #: Synchro's detector records, per controller.  ``numDetects`` is what says
    #: a detector exists; ``DetectSize1`` is only how long it is, and Synchro
    #: leaves it at zero for some movements that still have one.  Those get the
    #: intersection's usual detector length rather than being dropped -- on
    #: Chattanooga that is four right turns whose phases would otherwise go
    #: uncalled from their own lane.
    spec = {}
    for plan in plans:
        sizes = _movement_row(lanes_table, plan.synchro_intid, "DetectSize1")
        usual = [value for value in (_number(v) for v in sizes.values())
                 if value and value > 0]
        spec[plan.sc_no] = (
            sizes,
            _movement_row(lanes_table, plan.synchro_intid, "DetectPhase1"),
            _movement_row(lanes_table, plan.synchro_intid, "numDetects"),
            {g.sg_no for g in plan.signal_groups},
            max(set(usual), key=usual.count) if usual else DEFAULT_DETECTOR_FEET)

    for control in controls:
        sizes, phases, counts, groups, usual = spec.get(
            control.sc_no, ({}, {}, {}, set(), DEFAULT_DETECTOR_FEET))

        # A lane may serve several movements.  Prefer the one whose phase is the
        # lane's own, so the call reaches the group the head shows.
        candidates = []
        for code in dict.fromkeys(control.movements):
            try:
                sg_no = int(float(phases.get(code)))
            except (TypeError, ValueError):
                continue
            if sg_no not in groups:
                continue
            wanted = _number(counts.get(code))
            if wanted is not None and wanted <= 0:
                continue  # Synchro says this movement has no detector
            feet = _number(sizes.get(code)) or 0.0
            candidates.append((sg_no == control.sg_no, code, sg_no,
                               (feet if feet > 0 else usual) * FEET_TO_METRES))
        if not candidates:
            continue
        _, code, sg_no, wanted_length = max(candidates, key=lambda c: c[0])
        if wanted_length <= 0:
            continue

        approach = links.get(control.link_no)
        if approach is None or approach.length <= 0:
            continue
        # Measure from the head, not from the end of the link, so a detector
        # that has to be squeezed onto a short approach cannot end up drawn on
        # top of its own signal head -- or downstream of it.
        usable = stop_line(control, links, approach.length) - HEAD_CLEARANCE
        if usable <= DETECTOR_MARGIN:
            continue
        pos, length, was_short = detector_placement(
            usable, STOP_BAR_SETBACK, wanted_length)
        if was_short:
            shortened.add(f"link {control.link_no} ({approach.length:.1f} m)")

        # The port number is how the .prbc finds this detector, and its
        # VehicleDetectors are keyed on the signal group, so the port must be
        # the group rather than a running count.  Numbering them sequentially
        # left five of Chattanooga's six controllers with no calls at all:
        # their coordinated phases sat green forever while every actuated phase
        # stayed red.  Several lanes calling one group share a port, which is
        # ordinary multi-lane detection.
        detectors.append(Detector(
            sc_no=control.sc_no, sg_no=sg_no, junction_id=control.junction_id,
            link_no=control.link_no, lane=control.lane, pos=pos, length=length,
            port_no=sg_no, movement=code, shortened=was_short))

    if shortened:
        warnings.append(
            f"{len(shortened)} approaches are too short for Synchro's detector "
            f"layout, so their detectors were shrunk to fit: "
            f"{', '.join(sorted(shortened))}. A shorter presence zone holds a "
            "call for less time.")
    return detectors, warnings


def check_coverage(links: dict, heads: list[SignalHead],
                   detectors: list[Detector]) -> tuple[list[str], str]:
    """Check that every lane of a signalised approach is actually signalised.

    Counting what was created cannot show what was never created, so this
    counts the other way round: it enumerates the lanes the **network** has and
    asks which of them a head reached.  That denominator is what was missing
    when Chattanooga shipped with five unsignalised connectors -- the audit at
    the time asked only whether phases cycled and whether any two conflicting
    greens overlapped, and an unsignalised lane causes neither.

    The rule is stated in Vissim's own objects rather than in movements,
    because a check written in movements would share the assumption that
    produced the bug.

    Args:
        links: Output of :func:`rt_vissim.network.read_links_csv`.
        heads: Output of :func:`build_signal_heads`.
        detectors: Output of :func:`build_detectors`.

    Returns:
        ``(problems, summary)`` -- one line per gap, and the coverage counts.
    """
    outgoing = _outgoing(links)
    approaches = sorted({int(h.from_link_no) for h in heads})
    headed: dict[tuple[int, int], int] = {}
    for head in heads:
        if head.lane is None:
            continue
        key = (int(head.from_link_no), int(head.lane))
        headed[key] = headed.get(key, 0) + 1
    detected = {(int(d.link_no), int(d.lane)) for d in detectors}

    problems: list[str] = []
    total = covered = 0
    for approach in approaches:
        # The lanes that actually enter the junction, which is what needs a
        # signal -- a lane that feeds no connector goes nowhere.
        serving = sorted({int(lane) for conn in outgoing.get(approach, ())
                          for lane in (conn.from_lanes or [])})
        for lane in serving:
            total += 1
            count = headed.get((approach, lane), 0)
            if count == 1:
                covered += 1
            elif count == 0:
                problems.append(
                    f"link {approach} lane {lane}: no signal head, so it crosses "
                    "the junction unsignalised.")
            else:
                problems.append(
                    f"link {approach} lane {lane}: {count} signal heads, but a "
                    "lane can show only one indication.")

    # A head at or past the point where the connectors leave is never reached:
    # a turning vehicle is on its connector by then and drives through the red.
    # Every one of Chattanooga's 69 heads was placed that way once, so this is
    # checked against the geometry rather than trusted to the placement rule.
    placed = passed = 0
    for head in heads:
        if head.lane is None or head.connector_no:
            continue
        departures = [conn.from_pos
                      for conn in outgoing.get(int(head.from_link_no), ())
                      if conn.from_pos is not None
                      and int(head.lane) in (conn.from_lanes or [])]
        if not departures:
            continue
        placed += 1
        if head.pos > min(departures) - 0.01:
            passed += 1
            problems.append(
                f"link {head.from_link_no} lane {head.lane}: the head is at "
                f"{head.pos:.2f} m but the connectors leave at "
                f"{min(departures):.2f} m, so vehicles turn off before reaching "
                "it and drive through the red.")

    uncalled = sorted(key for key in headed if key not in detected)
    summary = (f"{covered}/{total} signalised lanes carry exactly one signal "
               f"head, {placed - passed}/{placed} of them upstream of the stop "
               f"line, {len(uncalled)} with no detector")
    return problems, summary


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
