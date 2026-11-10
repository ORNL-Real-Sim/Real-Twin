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
"""Decide who yields where two movements meet, by US right of way.

Vissim finds the conflicts itself: importing Phoenix leaves 4,805 conflict
areas, each with a status Vissim chose.  Its choices for diverges are right --
every branching conflict comes back ``UNDETERMINED``, which the attribute
reference defines as "Both red: for branching conflicts, so that vehicles can
'see' each other.  There is no right of way, as vehicles simply remain in their
original sequence."  Its choices for merges and crossings are not US rules:
Phoenix came back with 353 right turns ahead of the through traffic they merge
into, 117 lefts ahead of the cross street's through, and 39 U-turns ahead of
everything.  So every conflict between two movements of one intersection is
decided here, whatever Vissim gave it:

* **diverging** -- two movements leaving the same approach: both red
  (``UNDETERMINED``), Vissim's own status for branching;
* **merging or crossing** -- the movement that turns across or into traffic
  yields: through, then right, then left, then U-turn.  A left yields to the
  oncoming through *and* the oncoming right (Uniform Vehicle Code 11-401); a
  right, including a channelised right or one on red, yields to the through
  traffic it joins; a U-turn yields to everything;
* **unsignalised intersections** -- before the turns, a movement from the
  major road has priority over one from the minor road, as a stop or yield
  sign on the minor approach would give it (UVC 11-403).  The major road is the
  one with more lanes; where two roads tie, only the turns decide.  A U-turn
  still yields to everything, the minor road included;
* **ties at unsignalised intersections** -- two movements of equal rank, such
  as two throughs crossing where no road is major: the vehicle on the right
  has priority (UVC 11-401).  Which one is on the right is read off the two
  approaches' headings; two approaches meeting head-on have no right and keep
  Vissim's status.  Every tie in Phoenix and Chattanooga is perpendicular,
  and Vissim's import had already given each to the vehicle on the right; the
  rule makes that a decision rather than a coincidence;
* **a U-turn joining its exit** -- the U-turn yields to the traffic already on
  the exit link, as it yields to everything else.  Two of Phoenix's U-turns
  came back from the import ahead of it;
* **merges outside an intersection** -- a slip lane that left the approach
  early and rejoins the cross street, an on-ramp, a lane drop: the road with
  more lanes has priority and the narrower stream yields, as a yield sign at
  the end of the channel would make it.  Phoenix's 7th St southbound slip lane
  at Indian School Rd came back from the import ahead of the westbound through.

Left as Vissim set them: two movements of equal rank at a signal (two throughs
there are separated by the signal, not by a yield), conflict areas Vissim made
passive, a movement against its own approach (a diverge, which the import
already leaves both red or passive), other conflicts involving an approach or
exit link, and conflicts between two different intersections, which no turn
rule covers.

The manual prefers conflict areas to priority rules -- "conflict areas also
take desired acceleration, maximum acceleration as well as the vehicle length
of the two vehicles in both streams into account and reflect the driving
behavior better than priority rules" -- and they need no geometry from us,
only a decision.
"""

from __future__ import annotations
import itertools
from typing import NamedTuple

#: Who gives way, by turn.  Higher wins.  A through movement is never the one
#: to yield; a U-turn always is; a left yields to an oncoming right.
TURN_PRIORITY = {"thru": 3, "right": 2, "left": 1, "Uturn": 0}

#: Two approaches whose bearings differ by more than this are one road, arriving
#: from opposite sides.
OPPOSING_ANGLE = 135.0

#: One vehicle is on the other's right when their headings differ by a right
#: angle, give or take this much.  Phoenix's and Chattanooga's ties all lie
#: within 10 degrees of it.
CROSSING_TOLERANCE = 45.0

#: ``ConflictArea.Status`` values.
A_HAS_RIGHT_OF_WAY = "AHASRIGHTOFWAY"
B_HAS_RIGHT_OF_WAY = "BHASRIGHTOFWAY"
UNDETERMINED = "UNDETERMINED"
PASSIVE = "PASSIVE"


class Owner(NamedTuple):
    """The movement a junction-internal link belongs to."""

    junction: str
    turn: str
    approach: int
    exit: int
    #: Heading of travel on the approach, degrees clockwise from north.
    bearing: float | None = None


def _junction(value) -> str:
    """Junction ID as text, whether the table read it as ``2`` or ``2.0``."""
    return str(value).removesuffix(".0")


def _angle(a: float, b: float) -> float:
    d = abs(a - b) % 360
    return min(d, 360 - d)


def _bearing(value) -> float | None:
    """Bearing as a number, or ``None`` when the table has none."""
    try:
        bearing = float(value)
    except (TypeError, ValueError):
        return None
    return None if bearing != bearing else bearing      # NaN


#: An internal link shorter than this has no real extent.  SUMO models a
#: turnaround as a pivot on the spot and ``netconvert`` writes it out as a road
#: a tenth of a metre long, so Vissim ends up drawing a conflict area between
#: two links that occupy the same ground in opposite directions -- a polygon
#: that folds through itself into a bowtie.  Chattanooga has four; the shortest
#: internal link that is *not* one of them is 8.5 m, so there is no borderline
#: case.
DEGENERATE_LENGTH = 1.0


def degenerate_links(links: dict) -> set[int]:
    """Return junction-internal links with no real extent.

    Args:
        links: Output of :func:`rt_vissim.network.read_links_csv`.

    Returns:
        The link numbers, empty when the network has none.
    """
    return {no for no, link in links.items()
            if not link.is_connector and link.junction_key
            and 0 < link.length < DEGENERATE_LENGTH}


def degenerate_uturn_paths(links: dict, movements) -> list[set[int]]:
    """Return the links making up each U-turn that pivots on the spot.

    Such a U-turn retraces its own approach, so it conflicts with *itself*: its
    entry connector overlaps its exit road and its exit connector overlaps its
    approach.  Vissim duly draws a conflict area for each, and because the two
    sides occupy the same ground in opposite directions the polygon folds
    through itself into a bowtie.

    Returning the whole path -- approach, both connectors, the pivot link and
    the exit -- lets those self-conflicts be picked out exactly, without
    touching any conflict this U-turn has with genuinely other traffic.

    Args:
        links: Output of :func:`rt_vissim.network.read_links_csv`.
        movements: The stage 1 movement table.

    Returns:
        One set of link numbers per degenerate U-turn.
    """
    degenerate = degenerate_links(links)
    connectors = {(ln.from_link, ln.to_link): ln.no
                  for ln in links.values() if ln.is_connector}

    paths: list[set[int]] = []
    for row in movements.itertuples(index=False):
        if str(row.Turn) != "Uturn":
            continue
        internal = [int(x) for x in
                    str(getattr(row, "InternalLinks_Vissim", "") or "").split()]
        if len(internal) != 1 or internal[0] not in degenerate:
            continue
        approach = int(row.FromLinkNo_Vissim)
        exit_link = int(row.ToLinkNo_Vissim)
        path = {approach, internal[0], exit_link}
        for pair in ((approach, internal[0]), (internal[0], exit_link)):
            connector = connectors.get(pair)
            if connector is not None:
                path.add(connector)
        paths.append(path)
    return paths


def link_turns(movements, links: dict | None = None) -> dict[int, Owner]:
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
        ``{link number: Owner(junction, turn, approach link, exit link)}``.
    """
    owner: dict[int, Owner] = {}
    connectors: dict[tuple, int] = {}
    if links:
        connectors = {(ln.from_link, ln.to_link): ln.no
                      for ln in links.values() if ln.is_connector}

    for row in movements.itertuples(index=False):
        internal = [int(x) for x in
                    str(getattr(row, "InternalLinks_Vissim", "") or "").split()]
        if not internal:
            continue
        bearing = _bearing(getattr(row, "Bearing", None))
        who = Owner(_junction(row.JunctionID_OpenDrive), str(row.Turn),
                    int(row.FromLinkNo_Vissim), int(row.ToLinkNo_Vissim), bearing)

        for link_no in internal:
            owner[link_no] = who

        # The connectors stitching the path together carry the same movement.
        chain = [who.approach, *internal, who.exit]
        for upstream, downstream in itertools.pairwise(chain):
            connector = connectors.get((upstream, downstream))
            if connector is not None:
                owner[connector] = who
    return owner


def major_roads(movements, links: dict, signalised=()) -> dict[str, frozenset]:
    """Return the major road's approach links at each unsignalised junction.

    Approaches arriving from opposite sides (bearings more than
    :data:`OPPOSING_ANGLE` apart) are one road, and a road is as wide as its
    widest approach.  The major road is the strictly widest one: the arterial a
    side street or driveway meets.  Where two roads tie there is no major road,
    and the turns alone decide.

    Args:
        movements: The stage 1 movement table.
        links: Output of :func:`rt_vissim.network.read_links_csv`.
        signalised: Junction IDs with a signal controller; skipped, because the
            signal decides who goes there.

    Returns:
        ``{junction id: frozenset of the major road's approach links}``.
    """
    skip = {_junction(j) for j in signalised}
    legs: dict[str, dict[int, float]] = {}
    for row in movements.itertuples(index=False):
        junction = _junction(row.JunctionID_OpenDrive)
        if junction not in skip:
            legs.setdefault(junction, {})[int(row.FromLinkNo_Vissim)] = float(row.Bearing)

    out: dict[str, frozenset] = {}
    for junction, bearings in legs.items():
        roads, seen = [], set()
        for leg, bearing in bearings.items():
            if leg in seen:
                continue
            road = {leg} | {other for other, b in bearings.items()
                            if other not in seen and other != leg
                            and _angle(bearing, b) > OPPOSING_ANGLE}
            seen |= road
            width = max(links[x].num_lanes if x in links else 1 for x in road)
            roads.append((width, frozenset(road)))
        roads.sort(key=lambda r: -r[0])
        if len(roads) >= 2 and roads[0][0] > roads[1][0]:
            out[junction] = roads[0][1]
    return out


def merge_streams(links: dict, owned=()) -> dict[int, tuple]:
    """Map the links and connectors of a merge outside any intersection to their stream.

    A slip lane -- a channelised right that leaves the approach before the
    intersection -- is its own road and rejoins the cross street at a small
    OpenDRIVE junction of its own, outside the movement table.  So does any
    lane merge.  Each junction-internal link there is one stream; the
    connector leaving it belongs to the same stream.

    Args:
        links: Output of :func:`rt_vissim.network.read_links_csv`.
        owned: Link numbers that belong to a movement (keys of
            :func:`link_turns`); their junctions are intersections, not merges.

    Returns:
        ``{link or connector: (junction key, stream link, downstream link)}``.
    """
    owned = set(owned)
    out_of = {}                                   # internal link -> (connector, downstream link)
    for ln in links.values():
        if ln.is_connector and ln.from_link in links:
            up = links[ln.from_link]
            if up.junction_key and not up.is_connector and up.no not in owned:
                out_of[up.no] = (ln.no, ln.to_link)
    streams: dict[int, tuple] = {}
    for no, (connector, downstream) in out_of.items():
        key = (links[no].junction_key, no, downstream)
        streams[no] = key
        streams[connector] = key
    return streams


def decide(turn_a: str, turn_b: str, major_a: bool = False,
           major_b: bool = False) -> str | None:
    """Return which side has right of way, or ``None`` when neither clearly does.

    Args:
        turn_a: Turn label of the movement on link A.
        turn_b: Turn label of the movement on link B.
        major_a: Whether movement A comes from an unsignalised junction's major
            road.
        major_b: The same for movement B.

    Returns:
        :data:`A_HAS_RIGHT_OF_WAY`, :data:`B_HAS_RIGHT_OF_WAY`, or ``None`` when
        the two rank equally and the rule cannot separate them.

    Examples:
        >>> decide("thru", "left")
        'AHASRIGHTOFWAY'
        >>> decide("right", "thru")
        'BHASRIGHTOFWAY'
        >>> decide("left", "right")
        'BHASRIGHTOFWAY'
        >>> decide("thru", "thru") is None
        True
        >>> decide("right", "left", major_a=False, major_b=True)
        'BHASRIGHTOFWAY'
        >>> decide("Uturn", "left", major_a=True, major_b=False)
        'BHASRIGHTOFWAY'
    """
    # A U-turn may be made only where it does not interfere with other
    # traffic, so it yields even to the minor road.
    if (turn_a == "Uturn") != (turn_b == "Uturn"):
        return B_HAS_RIGHT_OF_WAY if turn_a == "Uturn" else A_HAS_RIGHT_OF_WAY
    if major_a != major_b:
        return A_HAS_RIGHT_OF_WAY if major_a else B_HAS_RIGHT_OF_WAY
    rank_a = TURN_PRIORITY.get(turn_a)
    rank_b = TURN_PRIORITY.get(turn_b)
    if rank_a is None or rank_b is None or rank_a == rank_b:
        return None
    return A_HAS_RIGHT_OF_WAY if rank_a > rank_b else B_HAS_RIGHT_OF_WAY


def from_the_right(bearing_a: float | None, bearing_b: float | None) -> str | None:
    """Return which side arrives from the other's right, or ``None``.

    A driver heading north has traffic from the right arriving from the east,
    heading west: a heading 90 degrees counter-clockwise of its own.

    Args:
        bearing_a: Heading of movement A's approach, degrees clockwise from north.
        bearing_b: The same for movement B.

    Returns:
        :data:`A_HAS_RIGHT_OF_WAY` or :data:`B_HAS_RIGHT_OF_WAY` for whichever
        is on the other's right, or ``None`` when a bearing is missing or the
        two do not cross at about a right angle.

    Examples:
        >>> from_the_right(0.0, 270.0)     # B heads west, so comes from A's right
        'BHASRIGHTOFWAY'
        >>> from_the_right(0.0, 90.0)
        'AHASRIGHTOFWAY'
        >>> from_the_right(0.0, 180.0) is None
        True
    """
    if bearing_a is None or bearing_b is None:
        return None
    turn = (bearing_b - bearing_a) % 360
    if abs(turn - 270) < CROSSING_TOLERANCE:
        return B_HAS_RIGHT_OF_WAY
    if abs(turn - 90) < CROSSING_TOLERANCE:
        return A_HAS_RIGHT_OF_WAY
    return None


def _joins_exit(link: int, exit_link: int, links: dict | None) -> bool:
    """Whether ``link`` is ``exit_link`` or a connector onto it."""
    if link == exit_link:
        return True
    ln = (links or {}).get(link)
    return bool(ln is not None and ln.is_connector and ln.to_link == exit_link)


def plan_conflicts(pairs, owners: dict[int, Owner],
                   self_conflicts: list[set[int]] | None = None,
                   major: dict[str, frozenset] | None = None,
                   links: dict | None = None,
                   signalised=None,
                   ) -> tuple[dict, list[str]]:
    """Work out the US right-of-way status of each conflict area.

    Args:
        pairs: ``[(conflict id, link A, link B, current status), ...]``.
        owners: Output of :func:`link_turns`.
        self_conflicts: Output of :func:`degenerate_uturn_paths`.
        major: Output of :func:`major_roads`.
        links: Output of :func:`rt_vissim.network.read_links_csv`.  Supply it
            to decide merges outside an intersection, such as the end of a
            slip lane, by lane count.
        signalised: Junction IDs with a signal controller.  Supply it to
            settle ties at the other junctions in favour of the vehicle on the
            right; without it ties keep Vissim's status everywhere.

    Returns:
        ``({conflict id: new status}, warnings)``, holding only the conflict
        areas whose status changes.  Those the rule leaves alone are counted
        in the warnings.
    """
    decisions: dict = {}
    self_conflicts = self_conflicts or []
    major = major or {}
    streams = merge_streams(links, owners) if links else {}
    controlled = None if signalised is None else {_junction(j) for j in signalised}
    tally = dict.fromkeys(("suppressed", "diverge", "ranked", "major", "right_hand",
                           "uturn_exit", "slip", "passive", "unknown",
                           "other_junction", "unranked"), 0)

    for conflict_id, link_a, link_b, status in pairs:
        # A U-turn that pivots on the spot retraces its own approach, so Vissim
        # finds it conflicting with itself.  That is not an interaction between
        # two streams, and the polygon drawn for it folds into a bowtie, so it
        # is switched off.  Only conflicts where *both* sides belong to the same
        # U-turn's own path qualify; anything that U-turn crosses is left alone.
        if any({link_a, link_b} <= path for path in self_conflicts):
            if status != PASSIVE:
                decisions[conflict_id] = PASSIVE
                tally["suppressed"] += 1
            continue
        owner_a, owner_b = owners.get(link_a), owners.get(link_b)
        if owner_a is None and owner_b is None and status != PASSIVE:
            # A merge outside any intersection -- a slip lane rejoining the
            # cross street, or two roads becoming one: the narrower stream
            # yields to the road it joins, as a yield sign there would make it.
            sa, sb = streams.get(link_a), streams.get(link_b)
            if (sa and sb and sa[0] == sb[0] and sa[2] == sb[2] and sa[1] != sb[1]):
                lanes_a, lanes_b = links[sa[1]].num_lanes, links[sb[1]].num_lanes
                if lanes_a != lanes_b:
                    wanted = A_HAS_RIGHT_OF_WAY if lanes_a > lanes_b else B_HAS_RIGHT_OF_WAY
                    tally["slip"] += 1
                    if wanted != status:
                        decisions[conflict_id] = wanted
                    continue
        if (owner_a is None) != (owner_b is None) and status != PASSIVE:
            # A U-turn joining its exit link yields to the traffic already on
            # it, as it yields to everything else.
            one, other = (owner_a, link_b) if owner_a else (owner_b, link_a)
            if one.turn == "Uturn" and _joins_exit(other, one.exit, links):
                wanted = B_HAS_RIGHT_OF_WAY if owner_a else A_HAS_RIGHT_OF_WAY
                tally["uturn_exit"] += 1
                if wanted != status:
                    decisions[conflict_id] = wanted
                continue
        if owner_a is None or owner_b is None or owner_a == owner_b:
            tally["unknown"] += 1
            continue
        if owner_a.junction != owner_b.junction:
            # Two streams in different junctions that happen to cross; whoever
            # yields there is not a turn-priority question.
            tally["other_junction"] += 1
            continue
        if owner_a.approach == owner_b.approach:
            # Leaving the same approach, the two streams start in one queue:
            # neither yields, and each sees the other.
            wanted = UNDETERMINED
            tally["diverge"] += 1
        elif status == PASSIVE:
            # Vissim judged this overlap not to be an interaction.
            tally["passive"] += 1
            continue
        else:
            road = major.get(owner_a.junction, frozenset())
            on_a, on_b = owner_a.approach in road, owner_b.approach in road
            wanted = decide(owner_a.turn, owner_b.turn, on_a, on_b)
            if wanted is None and controlled is not None and owner_a.junction not in controlled:
                # A tie with no sign or signal to settle it: the vehicle on
                # the right goes first (UVC 11-401).
                wanted = from_the_right(owner_a.bearing, owner_b.bearing)
                if wanted is not None:
                    tally["right_hand"] += 1
                    if wanted != status:
                        decisions[conflict_id] = wanted
                    continue
            if wanted is None:
                tally["unranked"] += 1
                continue
            uturn = (owner_a.turn == "Uturn") != (owner_b.turn == "Uturn")
            tally["major" if on_a != on_b and not uturn else "ranked"] += 1
        if wanted != status:
            decisions[conflict_id] = wanted

    notes = {
        "suppressed": "are a U-turn conflicting with its own approach, because "
                      "SUMO models a turnaround as a pivot on the spot and the "
                      "exit retraces the approach. They were set passive: the "
                      "bowtie polygon is not an interaction between two streams.",
        "diverge": "are two movements leaving the same approach: both red, so "
                   "each sees the other and neither yields.",
        "ranked": "are merges or crossings decided by the turns: through, then "
                  "right, then left, then U-turn.",
        "major": "are at an unsignalised junction, where the major road's "
                 "movement has priority over the minor road's.",
        "right_hand": "are a tie at an unsignalised junction -- two movements "
                      "of equal rank, neither on a major road -- where the "
                      "vehicle on the right has priority.",
        "uturn_exit": "are a U-turn joining its exit link, where the U-turn "
                      "yields to the traffic already on it.",
        "slip": "are a merge outside any intersection, such as a slip lane "
                "rejoining the cross street: the road with more lanes has "
                "priority and the narrower stream yields.",
        "passive": "were made passive by Vissim and stay passive.",
        "unknown": "involve a link that belongs to no single movement -- an "
                   "approach or an exit carries them all -- and keep Vissim's "
                   "status.",
        "other_junction": "cross two different junctions and keep Vissim's "
                          "status; they are not a turn-priority question.",
        "unranked": "are between movements of equal rank, such as two through "
                    "movements, which a signal separates rather than a yield, "
                    "or which meet head-on; they keep Vissim's status.",
    }
    warnings = [f"{tally[key]} conflict areas {text}" for key, text in notes.items()
                if tally[key]]
    return decisions, warnings


def summarise(decisions: dict, pairs) -> str:
    """Return a one-line summary of what will change."""
    return (f"{len(decisions)} of {len(pairs)} conflict areas set to the US "
            "right-of-way rule")
