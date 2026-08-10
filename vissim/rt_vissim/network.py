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
import xml.etree.ElementTree as ET
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

#: Turn classification thresholds, ported from SUMO's ``NBNode::getDirection``
#: so that both pipelines label a movement the same way.  A movement inside
#: :data:`STRAIGHT_MAX` is through *unless* another movement off the same
#: approach is straighter, in which case it is a part-turn -- which RealTwin
#: maps to plain left/right.  Below :data:`PART_MIN` that demotion never
#: applies.  U-turns are structural in SUMO (the reverse edge), which is a
#: near-180 degree movement here.
STRAIGHT_MAX = 44.0
PART_MIN = 6.0
UTURN_MIN = 170.0

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


def read_opendrive_junctions(path: str | Path) -> dict[str, str]:
    """Return ``{road id: junction id}`` for the roads inside a junction.

    Every ``<road>`` in an OpenDRIVE file carries a ``junction`` attribute: the
    id of the junction it belongs to, or ``-1`` when it is an ordinary road.
    Both the attribute and the junction's ``id`` are mandatory in the standard,
    so this works for any conforming file whatever produced it -- unlike reading
    the junction out of a link's name, which only works for a network SUMO
    exported with original names preserved.

    The junction's ``name`` is deliberately ignored.  It is optional in the
    standard and holds whatever the producer chose to put there.

    Args:
        path: Path to the ``.xodr`` file.

    Returns:
        ``{road id: junction id}``, containing only roads inside a junction.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"OpenDRIVE file not found: {path}")

    root = ET.parse(path).getroot()
    mapping: dict[str, str] = {}
    for road in root.findall("road"):
        junction = (road.get("junction") or "-1").strip()
        road_id = (road.get("id") or "").strip()
        if road_id and junction and junction != "-1":
            mapping[road_id] = junction
    return mapping


def read_opendrive_junction_names(path: str | Path) -> dict[str, str]:
    """Return ``{junction id: junction name}`` from an OpenDRIVE file.

    The ``name`` attribute is optional and holds whatever the producer chose, so
    the pipeline never groups or labels by it.  It is worth recording, though:
    ``netconvert`` writes the SUMO junction id there, which is the only way to
    line a Vissim MatchupTable up against a SUMO one once junctions are numbered
    by OpenDRIVE id.

    Args:
        path: Path to the ``.xodr`` file.

    Returns:
        ``{junction id: name}``, skipping junctions with no name.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"OpenDRIVE file not found: {path}")

    root = ET.parse(path).getroot()
    names: dict[str, str] = {}
    for junction in root.findall("junction"):
        jid = (junction.get("id") or "").strip()
        name = (junction.get("name") or "").strip()
        if jid and name:
            names[jid] = name
    return names


def _junction_key(road_id: str | None, orig_name: str,
                  road_junction: dict[str, str] | None) -> str | None:
    """Return the junction a link belongs to, preferring the OpenDRIVE file.

    Args:
        road_id: OpenDRIVE road id parsed from the Vissim link name.
        orig_name: The road's original name, if the importer kept one.
        road_junction: Output of :func:`read_opendrive_junctions`, or ``None``.

    Returns:
        The junction id, or ``None`` when the link is not inside a junction.
    """
    if road_junction is not None:
        return road_junction.get(str(road_id)) if road_id else None
    return junction_key_from_name(orig_name)


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
# Junctions from OpenDRIVE geometry
# ---------------------------------------------------------------------- #
#: Points sampled along each ``<geometry>`` record.  Enough to pin a short
#: junction road's curvature without making the cost matrix expensive.
GEOMETRY_SAMPLES = 12

#: Points each road and link is resampled to before they are compared.
SHAPE_POINTS = 6

#: Rows of the cost matrix built at once, to bound peak memory on large networks.
MATCH_BLOCK = 256


def sample_geometry(geo) -> list[tuple[float, float]]:
    """Return points along one OpenDRIVE ``<geometry>`` record.

    Every primitive is evaluated in closed form.  Approximating them as straight
    chords is not good enough here: ``netconvert`` writes almost every junction
    road as ``paramPoly3`` (1278 of 1407 records on Chattanooga, with no arcs at
    all), so a chord misplaces precisely the short curved roads being matched.

    Args:
        geo: A ``<geometry>`` element.

    Returns:
        ``[(x, y), ...]`` in the file's coordinate system.
    """
    x0, y0 = float(geo.get("x")), float(geo.get("y"))
    hdg, length = float(geo.get("hdg")), float(geo.get("length"))
    cos_h, sin_h = math.cos(hdg), math.sin(hdg)

    def to_global(u: float, v: float) -> tuple[float, float]:
        return (x0 + u * cos_h - v * sin_h, y0 + u * sin_h + v * cos_h)

    arc, poly, cubic = geo.find("arc"), geo.find("paramPoly3"), geo.find("poly3")
    spiral = geo.find("spiral")

    points = []
    for step in range(GEOMETRY_SAMPLES + 1):
        t = step / GEOMETRY_SAMPLES
        s = length * t

        if poly is not None:
            def coeff(name: str) -> float:
                return float(poly.get(name, 0.0))
            p = t if poly.get("pRange", "normalized") == "normalized" else s
            u = coeff("aU") + coeff("bU") * p + coeff("cU") * p**2 + coeff("dU") * p**3
            v = coeff("aV") + coeff("bV") * p + coeff("cV") * p**2 + coeff("dV") * p**3
            points.append(to_global(u, v))
        elif arc is not None and abs(float(arc.get("curvature", 0.0))) > 1e-12:
            k = float(arc.get("curvature"))
            points.append((x0 + (math.sin(hdg + k * s) - math.sin(hdg)) / k,
                           y0 - (math.cos(hdg + k * s) - math.cos(hdg)) / k))
        elif cubic is not None:
            def coeff3(name: str) -> float:
                return float(cubic.get(name, 0.0))
            v = coeff3("a") + coeff3("b") * s + coeff3("c") * s**2 + coeff3("d") * s**3
            points.append(to_global(s, v))
        elif spiral is not None:
            # A Fresnel integral would be exact; the mean-curvature arc is within
            # a few centimetres over the short records netconvert emits.
            start = float(spiral.get("curvStart", 0.0))
            end = float(spiral.get("curvEnd", 0.0))
            k = start + (end - start) * t / 2.0
            if abs(k) > 1e-12:
                points.append((x0 + (math.sin(hdg + k * s) - math.sin(hdg)) / k,
                               y0 - (math.cos(hdg + k * s) - math.cos(hdg)) / k))
            else:
                points.append(to_global(s, 0.0))
        else:
            points.append(to_global(s, 0.0))
    return points


def read_opendrive_roads(path: str | Path) -> dict[str, dict]:
    """Return the drivable roads of an OpenDRIVE file, with shape and lanes.

    Roads with no driving lane are left out because Vissim does not import them:
    measured on an OSM extract, 516 of 1032 roads carried a driving lane and
    Vissim created exactly 516 links, the other 516 being sidewalk and
    restricted ways.

    Args:
        path: Path to the ``.xodr`` file.

    Returns:
        ``{road id: {"points", "length", "lanes"}}``.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"OpenDRIVE file not found: {path}")

    root = ET.parse(path).getroot()
    roads: dict[str, dict] = {}
    for road in root.findall("road"):
        points: list[tuple[float, float]] = []
        for geo in road.findall("./planView/geometry"):
            try:
                points.extend(sample_geometry(geo))
            except (TypeError, ValueError):
                continue
        if not points:
            continue
        lanes = len([lane for lane in road.findall(".//lane")
                     if lane.get("type") == "driving"])
        if lanes == 0:
            continue
        roads[road.get("id")] = {
            "points": points,
            "length": float(road.get("length") or 0.0),
            "lanes": lanes,
        }
    return roads


def resample(points: list[tuple[float, float]], count: int) -> list[tuple[float, float]]:
    """Resample a polyline to ``count`` points evenly spaced along its length."""
    if not points:
        return []
    if len(points) == 1:
        return [points[0]] * count

    cumulative = [0.0]
    for a, b in zip(points, points[1:]):
        cumulative.append(cumulative[-1] + math.dist(a, b))
    total = cumulative[-1]
    if total <= 0:
        return [points[0]] * count

    out, segment = [], 0
    for i in range(count):
        target = total * i / (count - 1)
        while segment < len(cumulative) - 2 and cumulative[segment + 1] < target:
            segment += 1
        span = cumulative[segment + 1] - cumulative[segment]
        f = 0.0 if span <= 0 else (target - cumulative[segment]) / span
        a, b = points[segment], points[segment + 1]
        out.append((a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1])))
    return out


def assign_junctions_by_geometry(links: dict[int, VissimLink], xodr_path: str | Path,
                                 ) -> tuple[dict[int, str | None], list[str]]:
    """Map each Vissim link to an OpenDRIVE junction by where it physically is.

    The alternative to reading the OpenDRIVE road ID out of the Vissim link
    name.  The name carries it today, but no standard requires that, whereas a
    link's geometry *is* the road's geometry and cannot drift.

    Vissim imports one link per drivable road, so the two sets are in bijection
    and this is an assignment problem, not a nearest-neighbour search -- solving
    it as the latter lets several links claim one road and collapses.  The cost
    combines shape, length and lane count, which no two distinct roads share.

    Vissim also shifts the network on import, so the translation is recovered
    first by aligning centroids.  No link name is read anywhere in here.

    Args:
        links: Output of :func:`read_links`.
        xodr_path: The OpenDRIVE file the network was imported from.

    Returns:
        ``({link number: junction id or None}, warnings)``.

    Raises:
        ImportError: If numpy or scipy is unavailable.
    """
    try:
        import numpy as np
        from scipy.optimize import linear_sum_assignment
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "Geometric junction identification needs numpy and scipy. "
            "Install them, or pass junction_source='name'.") from exc

    warnings: list[str] = []
    road_junction = read_opendrive_junctions(xodr_path)
    roads = read_opendrive_roads(xodr_path)

    candidates = [ln for ln in links.values()
                  if not ln.is_connector and len(ln.points) >= 2]
    if not candidates or not roads:
        return {}, ["No geometry to match; junctions could not be derived "
                    "from the OpenDRIVE file."]

    road_ids = list(roads)
    road_shape = np.array([resample(roads[r]["points"], SHAPE_POINTS) for r in road_ids])
    road_length = np.array([roads[r]["length"] for r in road_ids])
    road_lanes = np.array([roads[r]["lanes"] for r in road_ids], dtype=float)

    link_shape = np.array([resample(ln.points, SHAPE_POINTS) for ln in candidates])
    link_length = np.array([ln.length for ln in candidates])
    link_lanes = np.array([ln.num_lanes for ln in candidates], dtype=float)

    if len(candidates) != len(road_ids):
        warnings.append(
            f"{len(candidates)} Vissim links against {len(road_ids)} drivable "
            "OpenDRIVE roads: not a one-to-one import, so some links will take "
            "no junction.")

    shift = (road_shape.reshape(-1, 2).mean(axis=0)
             - link_shape.reshape(-1, 2).mean(axis=0))
    moved = link_shape + shift

    # Blocked so the n x m x SHAPE_POINTS array never materialises whole.
    cost = np.empty((len(candidates), len(road_ids)))
    for start in range(0, len(candidates), MATCH_BLOCK):
        stop = min(start + MATCH_BLOCK, len(candidates))
        diff = moved[start:stop, None, :, :] - road_shape[None, :, :, :]
        cost[start:stop] = (np.sqrt((diff ** 2).sum(axis=3)).mean(axis=2)
                            + 2.0 * np.abs(link_length[start:stop, None] - road_length[None, :])
                            + 5.0 * np.abs(link_lanes[start:stop, None] - road_lanes[None, :]))

    rows, cols = linear_sum_assignment(cost)
    assigned: dict[int, str | None] = {ln.no: None for ln in candidates}
    residuals = []
    for row, col in zip(rows, cols):
        assigned[candidates[row].no] = road_junction.get(road_ids[col])
        residuals.append(cost[row, col])

    if residuals:
        worst = max(residuals)
        median = float(np.median(residuals))
        # The residual is dominated by a constant lateral offset -- Vissim link
        # geometry is the lane centreline, the OpenDRIVE planView is the road
        # reference line -- so it is roughly half a carriageway even when every
        # match is right.  Only a residual far above that median is suspicious.
        if worst > max(20.0, 5 * median):
            warnings.append(
                f"Worst geometry match residual {worst:.1f} m against a median of "
                f"{median:.1f} m; check the junctions derived near it.")
    return assigned, warnings


# ---------------------------------------------------------------------- #
# Reading the network over COM
# ---------------------------------------------------------------------- #
def read_links(session, road_junction: dict[str, str] | None = None,
               ) -> dict[int, VissimLink]:
    """Read every link and connector from a live Vissim session.

    Args:
        session: A started :class:`~rt_vissim.com.VissimSession`.
        road_junction: Output of :func:`read_opendrive_junctions`.  Supply it
            whenever the ``.xodr`` is available: junction membership then comes
            from the OpenDRIVE file rather than from link names, which only
            carry it for a network SUMO exported with original names.

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
            junction_key=_junction_key(road_id, orig_name, road_junction),
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


def read_links_csv(path: str | Path,
                   road_junction: dict[str, str] | None = None,
                   ) -> dict[int, VissimLink]:
    """Rebuild the link table from the CSV stage 1 wrote.

    The connectivity the demand stage needs -- which connector joins which pair
    of links -- is all in that file, so vehicle inputs and routing decisions can
    be built and checked without a Vissim licence in the loop.  Geometry is not
    restored: ``points`` stays empty, so this is not enough for
    :func:`extract_network`, which needs bearings.

    Args:
        path: Path to ``<name>_links.csv``.
        road_junction: Output of :func:`read_opendrive_junctions`.

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
            junction_key=_junction_key(road_id, orig_name, road_junction),
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


def check_nodes_against_junctions(session, junctions: dict[str, list[int]],
                                  ) -> list[str]:
    """Compare the junctions derived here with the nodes Vissim built on import.

    The Vissim manual states that on OpenDRIVE import it "generates segment
    nodes from the OpenDRIVE junctions in the *.xodr file, if this file contains
    junctions with the corresponding connecting roads", and that it "adopts the
    name and ID" of each.  So a node should exist for every junction this module
    finds -- the two readings share the same precondition -- and the ids are a
    free check that they agree.

    The nodes cannot replace the grouping.  A node holds link *segments*, and an
    approach contributes the stretch of itself that lies inside the node polygon,
    so its members are the junction's internal links plus every approach: 25
    against 17 at Chattanooga's largest junction.

    The manual also warns that merging overlapping segment nodes "disrupts the
    direct mapping between the junctions from OpenDRIVE and the generated segment
    nodes", so a disagreement here may mean the network was edited rather than
    that the grouping is wrong.

    Args:
        session: A started :class:`~rt_vissim.com.VissimSession`.
        junctions: Output of :func:`derive_junctions`.

    Returns:
        Warnings; empty when the ids agree or the network has no nodes.
    """
    try:
        node_ids = set()
        for node in session.net.Nodes.GetAll():
            label = str(node.AttValue("Name") or "").strip()
            if label:
                node_ids.add(label.split(":")[0].strip())
    except Exception:  # noqa: BLE001 - older builds may not expose Nodes
        return []

    if not node_ids:
        return ["Vissim created no nodes for this network, so the junction "
                "grouping could not be cross-checked against the importer."]

    derived = set(junctions)
    warnings = []
    missing = sorted(node_ids - derived)
    extra = sorted(derived - node_ids)
    if missing:
        warnings.append(f"Vissim has nodes for junctions {missing} that the "
                        "OpenDRIVE grouping did not produce.")
    if extra:
        warnings.append(f"Junctions {extra} were derived but Vissim built no "
                        "node for them.")
    return warnings


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
        turn.  Columns: ``JunctionID_OpenDrive``, ``Bearing``, ``Numbering``,
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

            # Gather the approach's movements before labelling any of them:
            # SUMO's rule asks whether another movement off this same approach
            # is straighter, so no movement can be classified in isolation.
            # Signed change of heading is positive clockwise (right).  Ordering
            # an approach's movements by it puts them in true R, T, L, U order,
            # which the MatchupTable stage relies on; ordering by the turn
            # *label* leaves movements sharing a label in arbitrary order.
            found = []
            for exit_no, path in trace_movements(links, internal, successors, approach):
                to_link = links.get(exit_no)
                if to_link is None:
                    continue
                # The exit link's own heading describes the turn better than the
                # internal path, whose curvature is an artefact of the geometry.
                bearing_out = to_link.outbound_bearing()
                if bearing_out is None:
                    continue
                found.append((exit_no, path, signed_delta(bearing_in, bearing_out)))

            for (exit_no, path, signed), turn in zip(
                    found, classify_turns([f[2] for f in found])):
                rows.append({
                    "_delta": round(signed, 2),
                    "JunctionID_OpenDrive": junction_id,
                    "Bearing": round(bearing_in, 2),
                    "Numbering": int(round(bearing_in / 10.0)),
                    "Bound": bound,
                    "FromLinkNo_Vissim": approach,
                    "ToLinkNo_Vissim": exit_no,
                    "InternalLinks_Vissim": " ".join(str(p) for p in path),
                    "Turn": turn,
                    "Movement": f"{bound}{TURN_SUFFIX.get(turn, '')}",
                })

    columns = ["JunctionID_OpenDrive", "Bearing", "Numbering", "Bound",
               "FromLinkNo_Vissim", "ToLinkNo_Vissim", "InternalLinks_Vissim",
               "Turn", "Movement", "_delta"]
    df = pd.DataFrame(rows, columns=columns)
    if df.empty:
        return df

    # Same ordering as RealTwin's SUMO table: approaches start at 337.5 deg so a
    # northbound approach is listed first.
    # Junction ids are strings, but sorting them as strings puts 10 before 2.
    # RealTwin's SUMO table is in numeric order, and the two are meant to be
    # readable side by side, so sort numerically where the id is a number.
    df["_junction"] = pd.to_numeric(df["JunctionID_OpenDrive"], errors="coerce")
    df["_junction_text"] = df["JunctionID_OpenDrive"].astype(str)
    df["_shifted"] = (df["Bearing"] - 337.5) % 360
    df["_turn_order"] = df["Turn"].map(TURN_ORDER)
    # Turn label first, then the signed change of heading, largest first.  The
    # tie-break matters: two movements sharing a label would otherwise sit in
    # arbitrary order, and the MatchupTable stage -- which resolves a duplicated
    # label by relabelling the second of the pair -- would correct whichever
    # happened to come first.  Ordering by angle makes that the milder turn,
    # which is the one that is really the through movement.
    df = df.sort_values(
        by=["_junction", "_junction_text", "_shifted", "_turn_order", "_delta"],
        ascending=[True, True, True, True, False],
        na_position="last").drop(
            columns=["_junction", "_junction_text", "_shifted", "_turn_order", "_delta"])
    return df.reset_index(drop=True)


#: How a link is tied back to an OpenDRIVE junction.  ``geometry`` matches the
#: link's shape against the file's roads and reads the junction from there;
#: ``name`` parses the OpenDRIVE road ID out of the Vissim link name.
JUNCTION_SOURCES = ("geometry", "name")


def extract_network(session, *, radius: float = DEFAULT_JUNCTION_RADIUS,
                    min_legs: int = 3, xodr_path: str | Path | None = None,
                    junction_source: str = "geometry",
                    ) -> tuple[pd.DataFrame, dict[int, VissimLink], dict[str, list[int]]]:
    """Read a Vissim session and derive its movement table in one call.

    Args:
        session: A started :class:`~rt_vissim.com.VissimSession`.
        radius: Clustering radius in metres, used only by the fallback grouping.
        min_legs: Minimum approaches for a group to count as a junction.
        xodr_path: The OpenDRIVE file the network was imported from.  Supply it
            whenever possible: junction membership then comes from the file
            rather than from a naming convention.
        junction_source: ``"geometry"`` (default) matches each link's shape to
            an OpenDRIVE road and takes that road's junction; ``"name"`` parses
            the road ID out of the Vissim link name.  Geometry is the default
            because the naming convention is PTV's, not the standard's, so
            nothing guarantees it survives; the two were measured to agree on
            every link of three networks.  Falls back to ``"name"`` when no
            OpenDRIVE file is given.

    Returns:
        ``(movement_table, links, junctions)``.

    Raises:
        ValueError: If ``junction_source`` is not one of :data:`JUNCTION_SOURCES`.
    """
    if junction_source not in JUNCTION_SOURCES:
        raise ValueError(f"junction_source must be one of {JUNCTION_SOURCES}, "
                         f"got {junction_source!r}")

    road_junction = None
    if xodr_path is not None:
        road_junction = read_opendrive_junctions(xodr_path)
        print(f"  :OpenDRIVE: {len(set(road_junction.values()))} junctions, "
              f"{len(road_junction)} roads inside one")
    elif junction_source == "geometry":
        junction_source = "name"
        print("  :No OpenDRIVE file given; falling back to name-based junctions.")

    links = read_links(session, road_junction)

    if junction_source == "geometry":
        assigned, warnings = assign_junctions_by_geometry(links, xodr_path)
        for warning in warnings:
            print(f"  :WARNING: {warning}")
        if assigned:
            agreed = sum(1 for no, key in assigned.items()
                         if links[no].junction_key == key)
            print(f"  :Junctions from geometry; the link names would have given "
                  f"the same answer for {agreed} of {len(assigned)} links.")
            for no, key in assigned.items():
                links[no].junction_key = key

    junctions = derive_junctions(links, radius=radius)
    for warning in check_nodes_against_junctions(session, junctions):
        print(f"  :WARNING: {warning}")
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


def signed_delta(bearing_in: float, bearing_out: float) -> float:
    """Return the change of heading, positive clockwise (right).

    Args:
        bearing_in: Bearing the vehicle arrives on, in degrees.
        bearing_out: Bearing the vehicle leaves on, in degrees.

    Returns:
        Degrees in ``[-180, 180)``.  An exact reversal comes back as ``-180``,
        which classifies as a U-turn either way.
    """
    return ((bearing_out - bearing_in + 180) % 360) - 180


def classify_turns(deltas: list[float]) -> list[str]:
    """Classify every movement off one approach, the way SUMO does.

    A single movement cannot be classified on its own.  SUMO's
    ``NBNode::getDirection`` calls a movement straight when it is inside
    :data:`STRAIGHT_MAX` *and* no other movement off that approach is
    straighter; where one is, this becomes a part-turn, which RealTwin's
    ``direction_mapping`` folds into plain ``left`` / ``right``.

    A fixed threshold cannot express that.  Chattanooga has two skewed
    approaches whose through movement sits at -21.3 and -28.4 degrees: both are
    well inside SUMO's 44 degree band with nothing straighter beside them, so
    SUMO calls them through, while a symmetric 20 degree cut called them left
    and put two lefts on one approach.

    Args:
        deltas: Signed heading changes for the approach's movements, in the
            order the labels are wanted back.

    Returns:
        One of ``"thru"``, ``"right"``, ``"left"``, ``"Uturn"`` per movement.
    """
    turning = [abs(d) for d in deltas if abs(d) <= UTURN_MIN]
    straightest = min(turning) if turning else None

    labels: list[str] = []
    for delta in deltas:
        magnitude = abs(delta)
        if magnitude > UTURN_MIN:
            labels.append("Uturn")
        elif magnitude < STRAIGHT_MAX:
            # Demoted to a part-turn only when something else is straighter.
            if (magnitude > PART_MIN and straightest is not None
                    and magnitude > straightest):
                labels.append("right" if delta > 0 else "left")
            else:
                labels.append("thru")
        else:
            labels.append("right" if delta > 0 else "left")
    return labels


def classify_turn(bearing_in: float, bearing_out: float) -> str:
    """Classify one movement with no knowledge of its neighbours.

    Kept for callers that have a single movement in hand.  Prefer
    :func:`classify_turns`: without the approach's other movements this cannot
    apply SUMO's "is anything straighter?" test, so it will call a skewed
    part-turn a through movement.

    Args:
        bearing_in: Bearing the vehicle arrives on, in degrees.
        bearing_out: Bearing the vehicle leaves on, in degrees.

    Returns:
        ``"thru"``, ``"right"``, ``"left"`` or ``"Uturn"``.
    """
    return classify_turns([signed_delta(bearing_in, bearing_out)])[0]


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
