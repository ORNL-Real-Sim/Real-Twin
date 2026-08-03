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
"""Derive junctions, approach bearings and turn movements from a Vissim network.

This is the Vissim counterpart of ``realtwin``'s
:func:`~realtwin.func_lib._c_abstract_scenario.rt_matchup_table_generation.format_junction_bearing`,
which reads a SUMO ``net.xml``.  That cannot be reused: importing OpenDRIVE into
Vissim renumbers everything, so neither the SUMO edge IDs nor the OpenDRIVE road
IDs survive as link numbers.  The *method* is the same, the source is different.

How Vissim represents an imported OpenDRIVE network
---------------------------------------------------
Every OpenDRIVE road becomes a Vissim **link**, including the connecting roads
*inside* junctions.  Vissim **connectors** are only the short (~1.5 m) stitches
between consecutive roads.  So one turn movement is not one connector -- it is a
path::

    approach link --conn--> internal link(s) --conn--> exit link

Vissim also names each link after the OpenDRIVE road it came from:

===============================  ==================================================
Link name                        Meaning
===============================  ==================================================
``390-0-Right``                  OpenDRIVE road 390, lane section 0, right side
``473: :12_0-0-Right``           road 473, OpenDRIVE road name ``:12_0``
===============================  ==================================================

The ``:12_0`` form is a SUMO *internal* edge, i.e. a path inside SUMO junction
12, which netconvert carries across when run with ``--output.original-names``.
That gives an exact, deterministic junction grouping straight out of the Vissim
model -- no geometric clustering needed.  Networks imported without original
names fall back to spatial clustering of the internal links.

The output is a dataframe with one row per turn movement, keyed on Vissim link
numbers, which :mod:`rt_vissim.matchup` turns into a MatchupTable workbook.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .com import com_objects_to_records

#: Attributes read for every link.  ``IsConn`` separates links from connectors.
LINK_ATTRIBUTES = ["No", "Name", "IsConn", "NumLanes", "Length2D"]

#: Vissim link name produced by the OpenDRIVE importer:
#: ``<roadId>[: <roadName>]-<laneSection>-<side>``.
LINK_NAME_RE = re.compile(r"^(?P<road>\d+)(?::\s*(?P<orig>.*?))?-(?P<sect>\d+)-(?P<side>\w+)$")

#: SUMO internal edge name: ``:<junctionId>_<index>``.
INTERNAL_EDGE_RE = re.compile(r"^:(?P<junction>[^_]+)_(?P<index>\d+)$")

#: Fallback single-linkage radius (metres) for grouping internal links into
#: junctions when link names carry no original SUMO names.
DEFAULT_JUNCTION_RADIUS = 30.0

#: Turn classification thresholds, in degrees of bearing change.
THRU_TOLERANCE = 20.0
UTURN_TOLERANCE = 20.0

#: Compass sector boundaries for the 4-way bound codes GridSmart and Synchro
#: use.  Quadrant edges at 45/135/225/315 reproduce the bound assignment in the
#: hand-filled Chattanooga MatchupTable (23.9 deg -> NB, 114.1 deg -> EB).
BOUND_SECTORS = (("NB", 315.0, 45.0), ("EB", 45.0, 135.0),
                 ("SB", 135.0, 225.0), ("WB", 225.0, 315.0))

#: Turn label -> movement-code suffix, so a northbound left becomes ``NBL``.
TURN_SUFFIX = {"right": "R", "thru": "T", "left": "L", "Uturn": "U"}

#: Sort order within an approach, matching RealTwin's SUMO matchup table.
TURN_ORDER = {"right": 1, "thru": 2, "left": 3, "Uturn": 4}

#: Guard against pathological graphs when walking through a junction.
MAX_INTERNAL_HOPS = 6


@dataclass
class VissimLink:
    """One Vissim link or connector, with geometry and parsed name.

    Attributes:
        no: Vissim link number.
        name: Raw Vissim link name.
        is_connector: Whether this is a connector rather than a link.
        num_lanes: Lane count.
        length: 2D length in metres.
        points: Polyline as ``[(x, y), ...]`` in the network coordinate system.
        from_link: For connectors, the upstream link number.
        to_link: For connectors, the downstream link number.
        road_id: OpenDRIVE road ID parsed from the name, if present.
        orig_name: OpenDRIVE road name parsed from the name, e.g. ``":12_0"``.
        junction_key: Junction label parsed from ``orig_name``; ``None`` for
            links that are not inside a junction.
    """

    no: int
    name: str = ""
    is_connector: bool = False
    num_lanes: int = 1
    length: float = 0.0
    points: list[tuple[float, float]] = field(default_factory=list)
    from_link: int | None = None
    to_link: int | None = None
    road_id: str | None = None
    orig_name: str = ""
    junction_key: str | None = None

    @property
    def is_internal(self) -> bool:
        """Whether this link is a path inside a junction."""
        return self.junction_key is not None

    @property
    def midpoint(self) -> tuple[float, float] | None:
        """Centroid of the polyline, or ``None`` when geometry is unavailable."""
        if not self.points:
            return None
        return (sum(p[0] for p in self.points) / len(self.points),
                sum(p[1] for p in self.points) / len(self.points))

    def outbound_bearing(self) -> float | None:
        """Bearing (deg) of the first segment -- the direction the link leaves in."""
        return _bearing_between(self.points[0], self.points[1]) if len(self.points) >= 2 else None

    def inbound_bearing(self) -> float | None:
        """Bearing (deg) of the last segment -- the direction the link arrives on.

        Mirrors SUMO's approach bearing, taken from the last two shape points of
        the incoming lane.
        """
        return _bearing_between(self.points[-2], self.points[-1]) if len(self.points) >= 2 else None


# ---------------------------------------------------------------------- #
# Name parsing
# ---------------------------------------------------------------------- #
def parse_link_name(name: str) -> tuple[str | None, str]:
    """Split a Vissim OpenDRIVE-import link name.

    Args:
        name: Raw link name, e.g. ``"473: :12_0-0-Right"`` or ``"390-0-Right"``.

    Returns:
        ``(road_id, original_name)``.  Either element may be empty/``None`` when
        the name does not follow the importer's convention -- for hand-drawn
        links, for instance.

    Examples:
        >>> parse_link_name("390-0-Right")
        ('390', '')
        >>> parse_link_name("473: :12_0-0-Right")
        ('473', ':12_0')
    """
    match = LINK_NAME_RE.match((name or "").strip())
    if not match:
        return None, ""
    return match.group("road"), (match.group("orig") or "").strip()


def junction_key_from_name(orig_name: str) -> str | None:
    """Return the junction label encoded in a SUMO internal edge name.

    Args:
        orig_name: OpenDRIVE road name, e.g. ``":12_0"``.

    Returns:
        The junction label (``"12"``), or ``None`` when the name is not an
        internal edge.
    """
    match = INTERNAL_EDGE_RE.match((orig_name or "").strip())
    return match.group("junction") if match else None


# ---------------------------------------------------------------------- #
# Reading the network over COM
# ---------------------------------------------------------------------- #
def read_links(session) -> dict[int, VissimLink]:
    """Read every link and connector from a live Vissim session.

    Args:
        session: A started :class:`~rt_vissim.com.VissimSession`.

    Returns:
        ``{link number: VissimLink}``, geometry and parsed names included.
    """
    links_col = session.net.Links
    records = com_objects_to_records(links_col, LINK_ATTRIBUTES)

    links: dict[int, VissimLink] = {}
    for rec in records:
        no = int(rec["No"])
        name = str(rec.get("Name") or "")
        road_id, orig_name = parse_link_name(name)
        links[no] = VissimLink(
            no=no,
            name=name,
            is_connector=bool(rec.get("IsConn")),
            num_lanes=int(rec.get("NumLanes") or 1),
            length=float(rec.get("Length2D") or 0.0),
            road_id=road_id,
            orig_name=orig_name,
            junction_key=junction_key_from_name(orig_name),
        )

    # Geometry and connector endpoints need per-object reads: connectors carry
    # FromLink/ToLink, plain links do not, so a single GetMultipleAttributes
    # over the whole collection cannot cover both.
    for obj in links_col:
        no = int(obj.AttValue("No"))
        link = links.get(no)
        if link is None:
            continue
        link.points = read_link_points(obj)
        if link.is_connector:
            link.from_link = _opt_int(_safe_attr(obj, "FromLink\\No"))
            link.to_link = _opt_int(_safe_attr(obj, "ToLink\\No"))

    return links


def read_links_csv(path: str | Path) -> dict[int, VissimLink]:
    """Rebuild the link table from the CSV stage 1 wrote.

    The connectivity the demand stage needs -- which connector joins which pair
    of links -- is all in that file, so vehicle inputs and routing decisions can
    be built and checked without a Vissim licence in the loop.  Geometry is not
    restored: ``points`` stays empty, so this is not enough for
    :func:`extract_network`, which needs bearings.

    Args:
        path: Path to ``<name>_links.csv``.

    Returns:
        ``{link number: VissimLink}``.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Links CSV not found: {path}")

    df = pd.read_csv(path)
    links: dict[int, VissimLink] = {}
    for row in df.itertuples(index=False):
        name = "" if pd.isna(row.Name) else str(row.Name)
        road_id, orig_name = parse_link_name(name)
        links[int(row.LinkNo)] = VissimLink(
            no=int(row.LinkNo),
            name=name,
            is_connector=bool(row.IsConnector),
            num_lanes=_opt_int(row.NumLanes) or 0,
            length=0.0 if pd.isna(row.Length2D) else float(row.Length2D),
            from_link=_opt_int(row.FromLink),
            to_link=_opt_int(row.ToLink),
            road_id=road_id,
            orig_name=orig_name,
            junction_key=junction_key_from_name(orig_name),
        )
    return links


def read_link_points(link_obj) -> list[tuple[float, float]]:
    """Read a link's polyline from its COM object.

    Vissim exposes link geometry as a ``LinkPolyPts`` collection; older builds
    name it ``LinkPolyPoints``.  Both are tried.

    Args:
        link_obj: A Vissim ``ILink`` COM object.

    Returns:
        ``[(x, y), ...]``; empty when the geometry could not be read.
    """
    for attr in ("LinkPolyPts", "LinkPolyPoints"):
        collection = getattr(link_obj, attr, None)
        if collection is None:
            continue
        try:
            rows = collection.GetMultipleAttributes(["X", "Y"])
            return [(float(x), float(y)) for x, y in rows]
        except Exception:  # noqa: BLE001 - fall back to per-point reads
            try:
                return [(float(p.AttValue("X")), float(p.AttValue("Y"))) for p in collection]
            except Exception:  # noqa: BLE001 - geometry unavailable
                continue
    return []


# ---------------------------------------------------------------------- #
# Topology
# ---------------------------------------------------------------------- #
def build_graph(links: dict[int, VissimLink]) -> tuple[dict[int, list[tuple[int, int]]],
                                                       dict[int, list[tuple[int, int]]]]:
    """Build the link-to-link graph implied by the connectors.

    Args:
        links: Output of :func:`read_links`.

    Returns:
        ``(successors, predecessors)``, each ``{link no: [(other link no,
        connector no), ...]}``.
    """
    successors: dict[int, list[tuple[int, int]]] = defaultdict(list)
    predecessors: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for link in links.values():
        if not link.is_connector:
            continue
        if link.from_link is None or link.to_link is None:
            continue
        successors[link.from_link].append((link.to_link, link.no))
        predecessors[link.to_link].append((link.from_link, link.no))
    return dict(successors), dict(predecessors)


def derive_junctions(links: dict[int, VissimLink],
                     radius: float = DEFAULT_JUNCTION_RADIUS) -> dict[str, list[int]]:
    """Group the junction-internal links into junctions.

    Uses the SUMO internal edge names Vissim preserved from OpenDRIVE when they
    are available, since that grouping is exact.  Falls back to single-linkage
    spatial clustering of internal links otherwise.

    Args:
        links: Output of :func:`read_links`.
        radius: Clustering radius in metres, used only by the fallback.

    Returns:
        ``{junction id: [internal link numbers]}``.
    """
    named: dict[str, list[int]] = defaultdict(list)
    for link in links.values():
        if link.is_connector:
            continue
        if link.junction_key is not None:
            named[link.junction_key].append(link.no)
    if named:
        return {jid: sorted(nos) for jid, nos in sorted(named.items())}

    return _cluster_internal_links(links, radius=radius)


def _cluster_internal_links(links: dict[int, VissimLink],
                            radius: float) -> dict[str, list[int]]:
    """Fallback junction grouping by single-linkage clustering.

    Used when the network was imported from an OpenDRIVE file without original
    names, so the internal links cannot be identified by name.  A link counts as
    junction-internal when it is short and sits between two connectors.
    """
    successors, predecessors = build_graph(links)
    candidates = [
        ln for ln in links.values()
        if not ln.is_connector and ln.midpoint
        and ln.no in successors and ln.no in predecessors
        and ln.length <= radius * 2
    ]
    if not candidates:
        return {}

    parent = {c.no: c.no for c in candidates}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    radius_sq = radius * radius
    for i, ci in enumerate(candidates):
        xi, yi = ci.midpoint
        for cj in candidates[i + 1:]:
            xj, yj = cj.midpoint
            if (xi - xj) ** 2 + (yi - yj) ** 2 <= radius_sq:
                ra, rb = find(ci.no), find(cj.no)
                if ra != rb:
                    parent[rb] = ra

    groups: dict[int, list[int]] = defaultdict(list)
    for c in candidates:
        groups[find(c.no)].append(c.no)

    def centre(members: list[int]) -> tuple[float, float]:
        pts = [links[m].midpoint for m in members]
        return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))

    ordered = sorted(groups.values(), key=lambda m: (round(centre(m)[0], 1), round(centre(m)[1], 1)))
    return {str(i): sorted(members) for i, members in enumerate(ordered, start=1)}


def trace_movements(links: dict[int, VissimLink], internal: set[int],
                    successors: dict[int, list[tuple[int, int]]],
                    approach: int) -> list[tuple[int, list[int]]]:
    """Walk from an approach link through a junction to every reachable exit.

    Args:
        links: Output of :func:`read_links`.
        internal: Internal link numbers belonging to one junction.
        successors: Successor map from :func:`build_graph`.
        approach: Link number of the approach.

    Returns:
        ``[(exit link number, [internal links traversed]), ...]``, deduplicated
        on the exit link so one movement is reported per approach/exit pair.
    """
    results: list[tuple[int, list[int]]] = []
    seen_exits: set[int] = set()
    # (current link, path of internal links so far)
    stack: list[tuple[int, list[int]]] = [(approach, [])]

    while stack:
        current, path = stack.pop()
        if len(path) > MAX_INTERNAL_HOPS:
            continue
        for nxt, _conn_no in successors.get(current, []):
            if nxt in internal:
                if nxt in path:  # cycle guard
                    continue
                stack.append((nxt, path + [nxt]))
            elif path and nxt not in seen_exits:
                # Left the junction: `path` is non-empty so we really did pass
                # through it rather than following a plain mid-block join.
                seen_exits.add(nxt)
                results.append((nxt, path))
    return results


def build_movement_table(links: dict[int, VissimLink],
                         junctions: dict[str, list[int]],
                         min_legs: int = 3) -> pd.DataFrame:
    """Build the per-movement table that seeds the MatchupTable.

    Args:
        links: Output of :func:`read_links`.
        junctions: Output of :func:`derive_junctions`.
        min_legs: Minimum distinct approaches for a group to count as a
            junction.  Two-leg groups are mid-block link joins, not
            intersections; SUMO applies the same filter via an exit count.

    Returns:
        One row per movement, sorted the way RealTwin sorts its SUMO matchup
        table: by junction, then approach bearing measured from 337.5 deg, then
        turn.  Columns: ``JunctionID_Vissim``, ``Bearing``, ``Numbering``,
        ``Bound``, ``FromLinkNo_Vissim``, ``ToLinkNo_Vissim``,
        ``InternalLinks_Vissim``, ``Turn``, ``Movement``.
    """
    successors, predecessors = build_graph(links)
    rows = []

    for junction_id, internal_nos in junctions.items():
        internal = set(internal_nos)

        # Approaches feed an internal link from outside the junction.
        approaches: set[int] = set()
        for internal_no in internal_nos:
            for pred, _conn in predecessors.get(internal_no, []):
                if pred not in internal:
                    approaches.add(pred)
        if len(approaches) < min_legs:
            continue

        for approach in sorted(approaches):
            from_link = links.get(approach)
            if from_link is None:
                continue
            bearing_in = from_link.inbound_bearing()
            if bearing_in is None:
                continue
            bound = bearing_to_bound(bearing_in)

            for exit_no, path in trace_movements(links, internal, successors, approach):
                to_link = links.get(exit_no)
                if to_link is None:
                    continue
                # The exit link's own heading describes the turn better than the
                # internal path, whose curvature is an artefact of the geometry.
                bearing_out = to_link.outbound_bearing()
                if bearing_out is None:
                    continue

                turn = classify_turn(bearing_in, bearing_out)
                # Signed change of heading: positive clockwise (right), negative
                # anticlockwise (left).  Ordering an approach's movements by this
                # puts them in true R, T, L, U order, which the MatchupTable
                # stage relies on.  Ordering by the turn *label* instead leaves
                # movements sharing a label in arbitrary order.
                delta = (bearing_out - bearing_in) % 360
                signed = delta if delta <= 180 else delta - 360
                rows.append({
                    "_delta": round(signed, 2),
                    "JunctionID_Vissim": junction_id,
                    "Bearing": round(bearing_in, 2),
                    "Numbering": int(round(bearing_in / 10.0)),
                    "Bound": bound,
                    "FromLinkNo_Vissim": approach,
                    "ToLinkNo_Vissim": exit_no,
                    "InternalLinks_Vissim": " ".join(str(p) for p in path),
                    "Turn": turn,
                    "Movement": f"{bound}{TURN_SUFFIX.get(turn, '')}",
                })

    columns = ["JunctionID_Vissim", "Bearing", "Numbering", "Bound",
               "FromLinkNo_Vissim", "ToLinkNo_Vissim", "InternalLinks_Vissim",
               "Turn", "Movement", "_delta"]
    df = pd.DataFrame(rows, columns=columns)
    if df.empty:
        return df

    # Same ordering as RealTwin's SUMO table: approaches start at 337.5 deg so a
    # northbound approach is listed first.
    df["_shifted"] = (df["Bearing"] - 337.5) % 360
    df["_turn_order"] = df["Turn"].map(TURN_ORDER)
    # Turn label first, then the signed change of heading, largest first.  The
    # tie-break matters: two movements sharing a label would otherwise sit in
    # arbitrary order, and the MatchupTable stage -- which resolves a duplicated
    # label by relabelling the second of the pair -- would correct whichever
    # happened to come first.  Ordering by angle makes that the milder turn,
    # which is the one that is really the through movement.
    df = df.sort_values(by=["JunctionID_Vissim", "_shifted", "_turn_order", "_delta"],
                        ascending=[True, True, True, False],
                        na_position="last").drop(
                            columns=["_shifted", "_turn_order", "_delta"])
    return df.reset_index(drop=True)


def extract_network(session, *, radius: float = DEFAULT_JUNCTION_RADIUS,
                    min_legs: int = 3) -> tuple[pd.DataFrame, dict[int, VissimLink], dict[str, list[int]]]:
    """Read a Vissim session and derive its movement table in one call.

    Args:
        session: A started :class:`~rt_vissim.com.VissimSession`.
        radius: Clustering radius in metres, used only by the fallback grouping.
        min_legs: Minimum approaches for a group to count as a junction.

    Returns:
        ``(movement_table, links, junctions)``.
    """
    links = read_links(session)
    junctions = derive_junctions(links, radius=radius)
    return build_movement_table(links, junctions, min_legs=min_legs), links, junctions


# ---------------------------------------------------------------------- #
# Geometry helpers
# ---------------------------------------------------------------------- #
def _bearing_between(p1: tuple[float, float], p2: tuple[float, float]) -> float | None:
    """Return the compass bearing (deg clockwise from north) from ``p1`` to ``p2``.

    Computed in the projected plane.  Vissim networks imported from a
    SUMO-generated OpenDRIVE are in UTM, where the deviation from true north is
    meridian convergence -- well under a degree across a network this size, which
    is immaterial for classifying approaches and turns.

    Returns:
        Bearing in ``[0, 360)``, or ``None`` when the points coincide.
    """
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    if dx == 0 and dy == 0:
        return None
    return math.degrees(math.atan2(dx, dy)) % 360


def classify_turn(bearing_in: float, bearing_out: float) -> str:
    """Classify a movement from its bearing change.

    Args:
        bearing_in: Bearing the vehicle arrives on, in degrees.
        bearing_out: Bearing the vehicle leaves on, in degrees.

    Returns:
        ``"thru"``, ``"right"``, ``"left"`` or ``"Uturn"`` -- the labels
        RealTwin's SUMO matchup table uses.
    """
    delta = (bearing_out - bearing_in) % 360
    if delta <= THRU_TOLERANCE or delta >= 360 - THRU_TOLERANCE:
        return "thru"
    if abs(delta - 180) <= UTURN_TOLERANCE:
        return "Uturn"
    return "right" if delta < 180 else "left"


def bearing_to_bound(bearing: float) -> str:
    """Map an approach bearing to a 4-way bound code (``NB``/``EB``/``SB``/``WB``)."""
    b = bearing % 360
    for name, lo, hi in BOUND_SECTORS:
        if lo > hi:  # the wrap-around sector
            if b >= lo or b < hi:
                return name
        elif lo <= b < hi:
            return name
    return "NB"


def _safe_attr(obj, name: str):
    """Read a COM attribute, returning ``None`` when it is absent."""
    try:
        return obj.AttValue(name)
    except Exception:  # noqa: BLE001 - attribute not present on this object
        return None


def _opt_int(value) -> int | None:
    """Coerce to ``int``, mapping unparseable values to ``None``."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
