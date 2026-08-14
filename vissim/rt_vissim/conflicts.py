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
"""Decide who yields where two movements cross.

Vissim finds the conflicts itself.  Importing Chattanooga leaves 606 conflict
areas already detected geometrically, and most already carry a decision -- but
166 come back ``UNDETERMINED``, which means neither stream yields and vehicles
drive through each other.  A permissive left is the case that matters: its
signal head is green at the same time as the opposing through, so nothing but a
conflict area stops it turning across moving traffic.

The manual prefers these to priority rules -- "conflict areas also take desired
acceleration, maximum acceleration as well as the vehicle length of the two
vehicles in both streams into account and reflect the driving behavior better
than priority rules" -- and they need no geometry from us, only a decision.

The decision generalises across every intersection, so nothing is per-junction:

* a through movement has priority over anything it crosses;
* a left yields to the opposing through;
* a right yields to whatever is already travelling where it is going;
* a U-turn yields to everything.

Right-of-way is therefore a ranking of the two turns, and only movements that
belong to the same junction are ranked -- two streams that merely pass near each
other in different intersections are left alone.
"""

from __future__ import annotations

#: Who gives way, by turn.  Higher wins.  A through movement is never the one
#: to yield; a U-turn always is.
TURN_PRIORITY = {"thru": 3, "left": 2, "right": 1, "Uturn": 0}

#: ``ConflictArea.Status`` values.
A_HAS_RIGHT_OF_WAY = "AHASRIGHTOFWAY"
B_HAS_RIGHT_OF_WAY = "BHASRIGHTOFWAY"
UNDETERMINED = "UNDETERMINED"
PASSIVE = "PASSIVE"


def link_turns(movements, links: dict | None = None) -> dict[int, tuple]:
    """Map every link inside a junction to the movement that uses it.

    A movement runs approach -> connector -> internal link(s) -> connector ->
    exit.  The internal links and the connectors between them belong to that one
    movement, so a conflict on any of them is a conflict for that turn.  The
    approach and exit links are deliberately left out: they carry every movement
    that uses them, so they say nothing about who should yield.

    Args:
        movements: The stage 1 movement table.
        links: Output of :func:`rt_vissim.network.read_links_csv`.  Supply it to
            attribute the connectors as well as the internal links.

    Returns:
        ``{link number: (junction id, turn)}``.
    """
    owner: dict[int, tuple] = {}
    connectors: dict[tuple, int] = {}
    if links:
        connectors = {(ln.from_link, ln.to_link): ln.no
                      for ln in links.values() if ln.is_connector}

    for row in movements.itertuples(index=False):
        junction = row.JunctionID_OpenDrive
        turn = str(row.Turn)
        internal = [int(x) for x in
                    str(getattr(row, "InternalLinks_Vissim", "") or "").split()]
        if not internal:
            continue

        for link_no in internal:
            owner[link_no] = (junction, turn)

        # The connectors stitching the path together carry the same movement.
        chain = [int(row.FromLinkNo_Vissim), *internal, int(row.ToLinkNo_Vissim)]
        for upstream, downstream in zip(chain, chain[1:]):
            connector = connectors.get((upstream, downstream))
            if connector is not None:
                owner[connector] = (junction, turn)
    return owner


def decide(turn_a: str, turn_b: str) -> str | None:
    """Return which side has right of way, or ``None`` when neither clearly does.

    Args:
        turn_a: Turn label of the movement on link A.
        turn_b: Turn label of the movement on link B.

    Returns:
        :data:`A_HAS_RIGHT_OF_WAY`, :data:`B_HAS_RIGHT_OF_WAY`, or ``None`` when
        the two rank equally and the rule cannot separate them.

    Examples:
        >>> decide("thru", "left")
        'AHASRIGHTOFWAY'
        >>> decide("right", "thru")
        'BHASRIGHTOFWAY'
        >>> decide("thru", "thru") is None
        True
    """
    rank_a = TURN_PRIORITY.get(turn_a)
    rank_b = TURN_PRIORITY.get(turn_b)
    if rank_a is None or rank_b is None or rank_a == rank_b:
        return None
    return A_HAS_RIGHT_OF_WAY if rank_a > rank_b else B_HAS_RIGHT_OF_WAY


def plan_conflicts(pairs, owners: dict[int, tuple]) -> tuple[dict, list[str]]:
    """Work out a status for each undetermined conflict area.

    Args:
        pairs: ``[(conflict id, link A, link B, current status), ...]``.
        owners: Output of :func:`link_turns`.

    Returns:
        ``({conflict id: new status}, warnings)``.  Conflicts already decided,
        or between links of different junctions, or that the rule cannot
        separate, are left out and counted in the warnings.
    """
    decisions: dict = {}
    already = other_junction = unranked = unknown = 0

    for conflict_id, link_a, link_b, status in pairs:
        if status not in (UNDETERMINED, None, ""):
            already += 1
            continue
        owner_a, owner_b = owners.get(link_a), owners.get(link_b)
        if owner_a is None or owner_b is None:
            unknown += 1
            continue
        if str(owner_a[0]) != str(owner_b[0]):
            # Two streams in different junctions that happen to cross; whoever
            # yields there is not a turn-priority question.
            other_junction += 1
            continue
        verdict = decide(owner_a[1], owner_b[1])
        if verdict is None:
            unranked += 1
            continue
        decisions[conflict_id] = verdict

    warnings: list[str] = []
    if already:
        warnings.append(f"{already} conflict areas already carry a decision and "
                        "were left alone.")
    if unknown:
        warnings.append(f"{unknown} conflict areas involve a link that belongs to "
                        "no single movement -- an approach or an exit carries them "
                        "all -- so no turn priority applies.")
    if other_junction:
        warnings.append(f"{other_junction} conflict areas cross two different "
                        "junctions and were left undetermined; they are not a "
                        "turn-priority question.")
    if unranked:
        warnings.append(f"{unranked} conflict areas are between movements of equal "
                        "rank, such as two through movements, which the signal "
                        "separates rather than a yield.")
    return decisions, warnings


def summarise(decisions: dict, pairs) -> str:
    """Return a one-line summary of what will change."""
    undetermined = sum(1 for _, _, _, status in pairs if status == UNDETERMINED)
    return (f"{len(decisions)} of {undetermined} undetermined conflict areas "
            f"resolved, out of {len(pairs)} in the network")
