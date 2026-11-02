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

"""Pure-Python OpenDRIVE reader modeled after the C++ libOpenDRIVE API.

The package provides an OpenDRIVE XML parser, road and lane geometry queries,
road mark and signal/object metadata access, mesh helpers, routing graph
helpers, and optional SUMO conversion helpers.

Typical usage example:
    ```python
    import pyopendrive as odr

    odr_map = odr.load("network.xodr")
    first_road = odr_map.get_roads()[0]
    position = first_road.ref_line.get_xyz(0.0)
    ```
"""

# Copyright 2026 Xiangyong Luo
#
# This file is part of pyopendrive.
#
# This file is adapted from libopendrive, originally licensed under the
# Apache License, Version 2.0.
#
# Modifications:
# - Converted from C++ to Python.
# - Refactored APIs for Python package usage.
# - Integrated with the pyopendrive package and viewer workflow.
#
# Licensed under the Apache License, Version 2.0.
# You may obtain a copy of the License at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Sequence
import heapq
import math
from pathlib import Path
from typing import Optional
import xml.etree.ElementTree as ET

import sumolib

Dict = dict
List = list
Tuple = tuple

Vec2D = Tuple[float, float]
Vec3D = Tuple[float, float, float]
Line3D = List[Vec3D]


@dataclass
class Mesh3D:
    """Triangle mesh container matching libOpenDRIVE's mesh data shape."""

    vertices: List[Vec3D] = field(default_factory=list)
    indices: List[int] = field(default_factory=list)
    normals: List[Vec3D] = field(default_factory=list)
    st_coordinates: List[Vec2D] = field(default_factory=list)

    def add_mesh(self, other: "Mesh3D") -> None:
        offset = len(self.vertices)
        self.vertices.extend(other.vertices)
        self.indices.extend(i + offset for i in other.indices)
        self.normals.extend(other.normals)
        self.st_coordinates.extend(other.st_coordinates)

    def get_obj(self) -> str:
        lines = []
        lines.extend(f"v {x} {y} {z}" for x, y, z in self.vertices)
        lines.extend(f"vn {nx} {ny} {nz}" for nx, ny, nz in self.normals)
        for i in range(0, len(self.indices), 3):
            tri = self.indices[i: i + 3]
            if len(tri) == 3:
                a, b, c = (idx + 1 for idx in tri)
                lines.append(f"f {a} {b} {c}")
        return "\n".join(lines)


@dataclass
class RoadsMesh(Mesh3D):
    """Mesh with vertex-start indices keyed by OpenDRIVE road id."""

    road_start_indices: Dict[int, str] = field(default_factory=dict)


@dataclass
class LanesMesh(RoadsMesh):
    """Road mesh bucket that also tracks lane section and lane starts."""

    lanesec_start_indices: Dict[int, float] = field(default_factory=dict)
    lane_start_indices: Dict[int, int] = field(default_factory=dict)


@dataclass
class RoadmarksMesh(LanesMesh):
    """Lane mesh bucket with additional road-mark type indices."""

    roadmark_type_start_indices: Dict[int, str] = field(default_factory=dict)


@dataclass
class RoadObjectsMesh(RoadsMesh):
    """Road mesh bucket with road-object start indices."""

    road_object_start_indices: Dict[int, str] = field(default_factory=dict)


@dataclass
class RoadSignalsMesh(RoadsMesh):
    """Road mesh bucket with road-signal start indices."""

    road_signal_start_indices: Dict[int, str] = field(default_factory=dict)


@dataclass
class RoadNetworkMesh:
    """Combined network mesh split into lane, mark, object, and signal layers."""

    lanes_mesh: LanesMesh = field(default_factory=LanesMesh)
    roadmarks_mesh: RoadmarksMesh = field(default_factory=RoadmarksMesh)
    road_objects_mesh: RoadObjectsMesh = field(default_factory=RoadObjectsMesh)
    road_signals_mesh: RoadSignalsMesh = field(default_factory=RoadSignalsMesh)

    def get_mesh(self) -> Mesh3D:
        mesh = Mesh3D()
        mesh.add_mesh(self.lanes_mesh)
        mesh.add_mesh(self.roadmarks_mesh)
        mesh.add_mesh(self.road_objects_mesh)
        mesh.add_mesh(self.road_signals_mesh)
        return mesh


def _float(node: ET.Element, name: str, default: float = 0.0) -> float:
    value = node.get(name)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _int(node: ET.Element, name: str, default: int = 0) -> int:
    value = node.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _offset_xy(offset: object) -> Tuple[float, float]:
    if isinstance(offset, dict):
        try:
            return (float(offset.get("x", 0.0)), float(offset.get("y", 0.0)))
        except (TypeError, ValueError):
            return (0.0, 0.0)
    return (0.0, 0.0)


def _projection_for_sumo(proj4: str) -> str:
    vertical_grid_prefixes = (
        "+geoidgrids=",
        "+geoid_crs=",
        "+vunits=",
        "+vto_meter=",
    )
    tokens = [
        token
        for token in proj4.split()
        if not token.startswith(vertical_grid_prefixes)
    ]
    if tokens and not any(
        token.startswith(("+proj=", "+init=", "+epsg=", "+crs="))
        for token in tokens
    ):
        tokens.insert(0, "+proj=tmerc")
        tokens.extend(["+k=1", "+x_0=0", "+y_0=0", "+datum=WGS84", "+units=m", "+no_defs"])
    return " ".join(tokens)


def _bool(node: ET.Element, name: str, default: bool = False) -> bool:
    value = node.get(name)
    return default if value is None else value.lower() in {"1", "true", "yes"}


def _children(node: Optional[ET.Element], name: str) -> List[ET.Element]:
    return [] if node is None else list(node.findall(name))


def _normalize(v: Vec3D) -> Vec3D:
    n = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    return (0.0, 0.0, 0.0) if n == 0 else (v[0] / n, v[1] / n, v[2] / n)


def _cross(a: Vec3D, b: Vec3D) -> Vec3D:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _next_towards_zero(i: int) -> int:
    if i > 0:
        return i - 1
    return i + 1 if i < 0 else 0


def _upper_key(keys: Sequence[float], s: float) -> int:
    lo, hi = 0, len(keys)
    while lo < hi:
        mid = (lo + hi) // 2
        if keys[mid] <= s:
            lo = mid + 1
        else:
            hi = mid
    return lo


@dataclass
class Poly3:
    """Cubic polynomial stored in absolute-s coordinates."""

    a: float = math.nan
    b: float = math.nan
    c: float = math.nan
    d: float = math.nan

    @classmethod
    def from_odr(cls, s0: float, a: float, b: float, c: float, d: float) -> "Poly3":
        return cls(
            a - b * s0 + c * s0 * s0 - d * s0 * s0 * s0,
            b - 2 * c * s0 + 3 * d * s0 * s0,
            c - 3 * d * s0,
            d,
        )

    def get(self, s: float) -> float:
        return self.a + self.b * s + self.c * s * s + self.d * s * s * s

    def get_grad(self, s: float) -> float:
        return self.b + 2 * self.c * s + 3 * self.d * s * s

    def is_zero(self) -> bool:
        return self.a == 0 and self.b == 0 and self.c == 0 and self.d == 0

    def isnan(self) -> bool:
        return (
            math.isnan(self.a)
            or math.isnan(self.b)
            or math.isnan(self.c)
            or math.isnan(self.d)
        )

    def negate(self) -> "Poly3":
        return Poly3(-self.a, -self.b, -self.c, -self.d)

    def approximate_linear(
        self, eps: float, s_start: float, s_end: float
    ) -> List[float]:
        if s_start == s_end:
            return []
        if self.c == 0 and self.d == 0:
            return [s_start, s_end]
        span = abs(s_end - s_start)
        step = max(eps, min(5.0, math.sqrt(max(eps, 1e-6)) * 10.0))
        n = max(1, int(math.ceil(span / step)))
        return [s_start + (s_end - s_start) * i / n for i in range(n + 1)]


@dataclass
class CubicSpline:
    """Piecewise cubic spline indexed by segment start position."""

    s0_to_poly: Dict[float, Poly3] = field(default_factory=dict)

    def empty(self) -> bool:
        return not self.s0_to_poly

    def size(self) -> int:
        return len(self.s0_to_poly)

    def get_poly(self, s: float, extend_start: bool = True) -> Poly3:
        if not self.s0_to_poly:
            return Poly3()
        keys = sorted(self.s0_to_poly)
        if not extend_start and s < keys[0]:
            return Poly3()
        idx = _upper_key(keys, s)
        if idx > 0:
            idx -= 1
        return self.s0_to_poly[keys[idx]]

    def get(
        self, s: float, default_val: float = 0.0, extend_start: bool = True
    ) -> float:
        poly = self.get_poly(s, extend_start)
        return default_val if poly.isnan() else poly.get(s)

    def get_grad(
        self, s: float, default_val: float = 0.0, extend_start: bool = True
    ) -> float:
        poly = self.get_poly(s, extend_start)
        return default_val if poly.isnan() else poly.get_grad(s)

    def negate(self) -> "CubicSpline":
        return CubicSpline({s0: poly.negate() for s0, poly in self.s0_to_poly.items()})

    def add(self, other: "CubicSpline") -> "CubicSpline":
        if not other.s0_to_poly:
            return CubicSpline(dict(self.s0_to_poly))
        if not self.s0_to_poly:
            return CubicSpline(dict(other.s0_to_poly))
        out: Dict[float, Poly3] = {}
        for s0 in sorted(set(self.s0_to_poly) | set(other.s0_to_poly)):
            a = self.get_poly(s0, False)
            b = other.get_poly(s0, False)
            if a.isnan() or b.isnan():
                out[s0] = b if a.isnan() else a
            else:
                out[s0] = Poly3(a.a + b.a, a.b + b.b, a.c + b.c, a.d + b.d)
        return CubicSpline(out)

    def approximate_linear(
        self, eps: float, s_start: float, s_end: float
    ) -> List[float]:
        if s_start == s_end or not self.s0_to_poly:
            return []
        keys = sorted(self.s0_to_poly)
        vals = {s_start, s_end}
        for i, s0 in enumerate(keys):
            next_s0 = keys[i + 1] if i + 1 < len(keys) else s_end
            a = max(s_start, s0)
            b = min(s_end, next_s0)
            if a < b:
                vals.update(self.s0_to_poly[s0].approximate_linear(eps, a, b))
        return sorted(vals)


class RoadGeometry:
    """Base class for OpenDRIVE plan-view geometry segments."""

    type = "unknown"

    def __init__(self, s0: float, x0: float, y0: float, hdg0: float, length: float):
        self.s0 = s0
        self.x0 = x0
        self.y0 = y0
        self.hdg0 = hdg0
        self.length = length
        self.xml_node: Optional[ET.Element] = None

    def get_xy(self, s: float) -> Vec2D:
        raise NotImplementedError

    def get_grad(self, s: float) -> Vec2D:
        raise NotImplementedError

    def approximate_linear(self, eps: float) -> List[float]:
        return [self.s0, self.s0 + self.length]


class Line(RoadGeometry):
    """Straight OpenDRIVE reference-line segment."""

    type = "line"

    def get_xy(self, s: float) -> Vec2D:
        ds = s - self.s0
        return (math.cos(self.hdg0) * ds + self.x0, math.sin(self.hdg0) * ds + self.y0)

    def get_grad(self, s: float) -> Vec2D:
        return (math.cos(self.hdg0), math.sin(self.hdg0))


class Arc(RoadGeometry):
    """Circular OpenDRIVE reference-line segment."""

    type = "arc"

    def __init__(
        self,
        s0: float,
        x0: float,
        y0: float,
        hdg0: float,
        length: float,
        curvature: float,
    ):
        super().__init__(s0, x0, y0, hdg0, length)
        self.curvature = curvature

    def get_xy(self, s: float) -> Vec2D:
        if abs(self.curvature) < 1e-12:
            return Line(self.s0, self.x0, self.y0, self.hdg0, self.length).get_xy(s)
        angle = (s - self.s0) * self.curvature - math.pi / 2
        r = 1.0 / self.curvature
        return (
            r * (math.cos(self.hdg0 + angle) - math.sin(self.hdg0)) + self.x0,
            r * (math.sin(self.hdg0 + angle) + math.cos(self.hdg0)) + self.y0,
        )

    def get_grad(self, s: float) -> Vec2D:
        return (
            math.sin((math.pi / 2) - self.curvature *
                     (s - self.s0) - self.hdg0),
            math.cos((math.pi / 2) - self.curvature *
                     (s - self.s0) - self.hdg0),
        )

    def approximate_linear(self, eps: float) -> List[float]:
        if abs(self.curvature) < 1e-12:
            return super().approximate_linear(eps)
        step = max(eps, min(self.length, 0.01 / abs(self.curvature)))
        n = max(1, int(math.ceil(self.length / step)))
        return [self.s0 + self.length * i / n for i in range(n + 1)]


class Spiral(RoadGeometry):
    """Clothoid-like segment approximated by numerical integration."""

    type = "spiral"

    def __init__(
        self,
        s0: float,
        x0: float,
        y0: float,
        hdg0: float,
        length: float,
        curv_start: float,
        curv_end: float,
    ):
        super().__init__(s0, x0, y0, hdg0, length)
        self.curv_start = curv_start
        self.curv_end = curv_end

    def _theta(self, u: float) -> float:
        if self.length == 0:
            return self.hdg0
        return (
            self.hdg0
            + self.curv_start * u
            + 0.5 * (self.curv_end - self.curv_start) * u * u / self.length
        )

    def get_xy(self, s: float) -> Vec2D:
        ds = max(0.0, min(self.length, s - self.s0))
        if ds == 0:
            return (self.x0, self.y0)
        n = max(8, min(400, int(math.ceil(ds / 1.0))))
        h = ds / n
        x = y = 0.0
        for i in range(n + 1):
            w = 4 if i % 2 else 2
            if i in [0, n]:
                w = 1
            th = self._theta(i * h)
            x += w * math.cos(th)
            y += w * math.sin(th)
        return (self.x0 + x * h / 3.0, self.y0 + y * h / 3.0)

    def get_grad(self, s: float) -> Vec2D:
        th = self._theta(max(0.0, min(self.length, s - self.s0)))
        return (math.cos(th), math.sin(th))

    def approximate_linear(self, eps: float) -> List[float]:
        n = max(1, int(math.ceil(self.length / max(eps, 1.0))))
        return [self.s0 + self.length * i / n for i in range(n + 1)]


class ParamPoly3(RoadGeometry):
    """Parametric cubic OpenDRIVE reference-line segment."""

    type = "paramPoly3"

    def __init__(
        self,
        s0: float,
        x0: float,
        y0: float,
        hdg0: float,
        length: float,
        aU: float,
        bU: float,
        cU: float,
        dU: float,
        aV: float,
        bV: float,
        cV: float,
        dV: float,
        pRange_normalized: bool = True,
    ):
        super().__init__(s0, x0, y0, hdg0, length)
        self.aU, self.bU, self.cU, self.dU = aU, bU, cU, dU
        self.aV, self.bV, self.cV, self.dV = aV, bV, cV, dV
        self.pRange_normalized = pRange_normalized

    def _p(self, s: float) -> float:
        ds = s - self.s0
        return ds / self.length if self.pRange_normalized and self.length else ds

    def get_xy(self, s: float) -> Vec2D:
        p = self._p(s)
        u = self.aU + self.bU * p + self.cU * p * p + self.dU * p * p * p
        v = self.aV + self.bV * p + self.cV * p * p + self.dV * p * p * p
        return (
            math.cos(self.hdg0) * u - math.sin(self.hdg0) * v + self.x0,
            math.sin(self.hdg0) * u + math.cos(self.hdg0) * v + self.y0,
        )

    def get_grad(self, s: float) -> Vec2D:
        p = self._p(s)
        du = self.bU + 2 * self.cU * p + 3 * self.dU * p * p
        dv = self.bV + 2 * self.cV * p + 3 * self.dV * p * p
        return (
            math.cos(self.hdg0) * du - math.sin(self.hdg0) * dv,
            math.sin(self.hdg0) * du + math.cos(self.hdg0) * dv,
        )

    def approximate_linear(self, eps: float) -> List[float]:
        n = max(1, int(math.ceil(self.length / max(eps, 2.0))))
        return [self.s0 + self.length * i / n for i in range(n + 1)]


@dataclass
class RefLine:
    """Road reference line with plan-view geometry and elevation profile."""

    road_id: str
    length: float
    elevation_profile: CubicSpline = field(default_factory=CubicSpline)
    s0_to_geometry: Dict[float, RoadGeometry] = field(default_factory=dict)

    def get_geometries(self) -> List[RoadGeometry]:
        return [self.s0_to_geometry[k] for k in sorted(self.s0_to_geometry)]

    def get_geometry_s0(self, s: float) -> float:
        if not self.s0_to_geometry:
            return math.nan
        keys = sorted(self.s0_to_geometry)
        idx = _upper_key(keys, s)
        return keys[max(0, idx - 1)]

    def get_geometry(self, s: float) -> Optional[RoadGeometry]:
        s0 = self.get_geometry_s0(s)
        return None if math.isnan(s0) else self.s0_to_geometry[s0]

    def get_xyz(self, s: float) -> Vec3D:
        geom = self.get_geometry(s)
        x, y = geom.get_xy(s) if geom else (0.0, 0.0)
        return (x, y, self.elevation_profile.get(s))

    def get_grad(self, s: float) -> Vec3D:
        geom = self.get_geometry(s)
        x, y = geom.get_grad(s) if geom else (0.0, 0.0)
        return (x, y, self.elevation_profile.get_grad(s))

    def approximate_linear(
        self, eps: float, s_start: float, s_end: float
    ) -> List[float]:
        vals = {s_start, s_end}
        for geom in self.get_geometries():
            g0, g1 = geom.s0, geom.s0 + geom.length
            if g1 < s_start or g0 > s_end:
                continue
            vals.update(
                s for s in geom.approximate_linear(eps) if s_start <= s <= s_end
            )
        vals.update(
            s
            for s in self.elevation_profile.approximate_linear(eps, s_start, s_end)
            if s_start <= s <= s_end
        )
        return sorted(vals)

    def get_line(self, s_start: float, s_end: float, eps: float) -> Line3D:
        return [self.get_xyz(s) for s in self.approximate_linear(eps, s_start, s_end)]

    def match(self, x: float, y: float) -> float:
        # Ternary/golden style search on the reference-line distance.
        lo, hi = 0.0, self.length
        for _ in range(80):
            m1 = lo + (hi - lo) / 3.0
            m2 = hi - (hi - lo) / 3.0
            p1, p2 = self.get_xyz(m1), self.get_xyz(m2)
            d1 = (p1[0] - x) ** 2 + (p1[1] - y) ** 2
            d2 = (p2[0] - x) ** 2 + (p2[1] - y) ** 2
            if d1 < d2:
                hi = m2
            else:
                lo = m1
        return (lo + hi) * 0.5


@dataclass(frozen=True, order=True)
class LaneKey:
    """Stable identity for a lane within a road and lane section."""

    road_id: str
    lanesection_s0: float
    lane_id: int

    def to_string(self) -> str:
        return f"{self.road_id}/{self.lanesection_s0:.6f}/{self.lane_id}"


@dataclass
class HeightOffset:
    inner: float = 0.0
    outer: float = 0.0


@dataclass(order=True)
class RoadMarksLine:
    road_id: str
    lanesection_s0: float
    lane_id: int
    group_s0: float
    width: float = -1.0
    length: float = 0.0
    space: float = 0.0
    t_offset: float = 0.0
    s_offset: float = 0.0
    name: str = ""
    rule: str = ""


@dataclass(order=True)
class RoadMarkGroup:
    road_id: str
    lanesection_s0: float
    lane_id: int
    width: float = -1.0
    height: float = 0.0
    s_offset: float = 0.0
    type: str = ""
    weight: str = ""
    color: str = ""
    material: str = ""
    lane_change: str = ""
    roadmark_lines: List[RoadMarksLine] = field(
        default_factory=list, compare=False)


@dataclass
class RoadMark:
    road_id: str
    lanesection_s0: float
    lane_id: int
    group_s0: float
    s_start: float
    s_end: float
    t_offset: float
    width: float
    type: str


@dataclass
class Lane:
    """OpenDRIVE lane with width, border, link, and road-mark data."""

    road_id: str
    lanesection_s0: float
    id: int
    level: bool = False
    type: str = ""
    predecessor: int = 0
    successor: int = 0
    lane_width: CubicSpline = field(default_factory=CubicSpline)
    outer_border: CubicSpline = field(default_factory=CubicSpline)
    s_to_height_offset: Dict[float, HeightOffset] = field(default_factory=dict)
    roadmark_groups: List[RoadMarkGroup] = field(default_factory=list)
    xml_node: Optional[ET.Element] = None

    @property
    def key(self) -> LaneKey:
        return LaneKey(self.road_id, self.lanesection_s0, self.id)

    def get_roadmarks(self, s_start: float, s_end: float) -> List[RoadMark]:
        if s_start == s_end or not self.roadmark_groups:
            return []
        groups = sorted(
            self.roadmark_groups,
            key=lambda g: (g.lanesection_s0 + g.s_offset, g.type, g.width),
        )
        roadmarks: List[RoadMark] = []
        for i, group in enumerate(groups):
            g0 = group.lanesection_s0 + group.s_offset
            g1 = (
                groups[i + 1].lanesection_s0 + groups[i + 1].s_offset
                if i + 1 < len(groups)
                else s_end
            )
            a, b = max(g0, s_start), min(g1, s_end)
            if b <= a:
                continue
            width = 0.25 if group.weight == "bold" else 0.12
            if group.width > 0:
                width = group.width
            if not group.roadmark_lines:
                roadmarks.append(
                    RoadMark(
                        self.road_id,
                        self.lanesection_s0,
                        self.id,
                        g0,
                        a,
                        b,
                        0.0,
                        width,
                        group.type,
                    )
                )
                continue
            for line in sorted(group.roadmark_lines):
                line_width = line.width if line.width > 0 else width
                period = line.length + line.space
                if period == 0:
                    continue
                s = line.group_s0 + line.s_offset
                while s < b:
                    roadmarks.append(
                        RoadMark(
                            self.road_id,
                            self.lanesection_s0,
                            self.id,
                            line.group_s0,
                            max(a, s),
                            min(b, s + line.length),
                            line.t_offset,
                            line_width,
                            group.type + line.name,
                        )
                    )
                    s += period
        return roadmarks


@dataclass
class LaneSection:
    """Contiguous section of a road containing lane definitions."""

    road_id: str
    s0: float
    id_to_lane: Dict[int, Lane] = field(default_factory=dict)
    xml_node: Optional[ET.Element] = None

    def get_lanes(self) -> List[Lane]:
        return [self.id_to_lane[k] for k in sorted(self.id_to_lane)]

    def get_lane_id(self, s: float, t: float) -> int:
        if self.id_to_lane[0].outer_border.get(s) == t:
            return 0
        borders = sorted(
            (lane.outer_border.get(s), lane_id)
            for lane_id, lane in self.id_to_lane.items()
        )
        idx = 0
        while idx < len(borders) and borders[idx][0] < t:
            idx += 1
        if idx == len(borders):
            idx -= 1
        if borders[idx][1] <= 0 and idx != 0 and t != borders[idx][0]:
            idx -= 1
        return borders[idx][1]

    def get_lane(self, id_or_s: float, t: Optional[float] = None) -> Lane:
        lane_id = int(id_or_s) if t is None else self.get_lane_id(
            float(id_or_s), t)
        return self.id_to_lane[lane_id]


@dataclass
class RoadLink:
    id: str = ""
    type: str = "none"
    contact_point: str = "none"


@dataclass
class RoadNeighbor:
    id: str
    side: str
    direction: str


@dataclass
class SpeedRecord:
    max: str = ""
    unit: str = ""


@dataclass
class Crossfall(CubicSpline):
    sides: Dict[float, str] = field(default_factory=dict)

    def get_crossfall(self, s: float, on_left_side: bool) -> float:
        if not self.s0_to_poly:
            return 0.0
        keys = sorted(self.s0_to_poly)
        idx = max(0, _upper_key(keys, s) - 1)
        side = self.sides.get(keys[idx], "both")
        if on_left_side and side == "right":
            return 0.0
        if not on_left_side and side == "left":
            return 0.0
        return self.s0_to_poly[keys[idx]].get(s)


@dataclass
class LaneValidityRecord:
    from_lane: int
    to_lane: int


@dataclass
class RoadObjectRepeat:
    s0: float = math.nan
    length: float = 0.0
    distance: float = 0.0
    t_start: float = math.nan
    t_end: float = math.nan
    width_start: float = math.nan
    width_end: float = math.nan
    height_start: float = math.nan
    height_end: float = math.nan
    z_offset_start: float = math.nan
    z_offset_end: float = math.nan


@dataclass
class RoadObjectCorner:
    id: int
    pt: Vec3D
    height: float
    type: str = "road"


@dataclass
class RoadObjectOutline:
    id: int
    fill_type: str = ""
    lane_type: str = ""
    outer: bool = True
    closed: bool = True
    outline: List[RoadObjectCorner] = field(default_factory=list)


@dataclass
class RoadObject:
    road_id: str
    id: str
    s0: float
    t0: float
    z0: float
    length: float
    valid_length: float
    width: float
    radius: float
    height: float
    hdg: float
    pitch: float
    roll: float
    type: str = ""
    name: str = ""
    orientation: str = ""
    subtype: str = ""
    is_dynamic: bool = False
    repeats: List[RoadObjectRepeat] = field(default_factory=list)
    outlines: List[RoadObjectOutline] = field(default_factory=list)
    lane_validities: List[LaneValidityRecord] = field(default_factory=list)


@dataclass
class RoadSignal:
    road_id: str
    id: str
    name: str
    s0: float
    t0: float
    is_dynamic: bool
    zOffset: float
    value: float
    height: float
    width: float
    hOffset: float
    pitch: float
    roll: float
    orientation: str
    country: str
    type: str
    subtype: str
    unit: str
    text: str
    lane_validities: List[LaneValidityRecord] = field(default_factory=list)


@dataclass
class Road:
    """OpenDRIVE road with geometry, lanes, links, objects, and signals."""

    id: str
    length: float
    junction: str
    name: str
    left_hand_traffic: bool = False
    predecessor: RoadLink = field(default_factory=RoadLink)
    successor: RoadLink = field(default_factory=RoadLink)
    neighbors: List[RoadNeighbor] = field(default_factory=list)
    lane_offset: CubicSpline = field(default_factory=CubicSpline)
    superelevation: CubicSpline = field(default_factory=CubicSpline)
    crossfall: Crossfall = field(default_factory=Crossfall)
    ref_line: RefLine = field(init=False)
    s_to_lanesection: Dict[float, LaneSection] = field(default_factory=dict)
    s_to_type: Dict[float, str] = field(default_factory=dict)
    s_to_speed: Dict[float, SpeedRecord] = field(default_factory=dict)
    id_to_object: Dict[str, RoadObject] = field(default_factory=dict)
    id_to_signal: Dict[str, RoadSignal] = field(default_factory=dict)
    xml_node: Optional[ET.Element] = None

    def __post_init__(self) -> None:
        self.ref_line = RefLine(self.id, self.length)

    def get_lanesections(self) -> List[LaneSection]:
        return [self.s_to_lanesection[k] for k in sorted(self.s_to_lanesection)]

    def get_road_objects(self) -> List[RoadObject]:
        return list(self.id_to_object.values())

    def get_road_signals(self) -> List[RoadSignal]:
        return list(self.id_to_signal.values())

    def get_lanesection_s0(self, s: float) -> float:
        if not self.s_to_lanesection:
            return math.nan
        keys = sorted(self.s_to_lanesection)
        idx = max(0, _upper_key(keys, s) - 1)
        lane_sec = self.s_to_lanesection[keys[idx]]
        if s < lane_sec.s0 or s > self.get_lanesection_end(lane_sec):
            return math.nan
        return lane_sec.s0

    def get_lanesection(self, s: float) -> LaneSection:
        s0 = self.get_lanesection_s0(s)
        if math.isnan(s0):
            raise RuntimeError("no valid lanesection")
        return self.s_to_lanesection[s0]

    def get_lanesection_end(self, lanesection_or_s0) -> float:
        s0 = (
            lanesection_or_s0.s0
            if isinstance(lanesection_or_s0, LaneSection)
            else float(lanesection_or_s0)
        )
        keys = sorted(self.s_to_lanesection)
        if s0 not in self.s_to_lanesection:
            return math.nan
        idx = keys.index(s0)
        return (
            self.length
            if idx + 1 == len(keys)
            else math.nextafter(keys[idx + 1], -math.inf)
        )

    def get_lanesection_length(self, lanesection_or_s0) -> float:
        s0 = (
            lanesection_or_s0.s0
            if isinstance(lanesection_or_s0, LaneSection)
            else float(lanesection_or_s0)
        )
        return self.get_lanesection_end(s0) - s0

    def get_xyz(self, s: float, t: float, h: float) -> Vec3D:
        s_vec = self.ref_line.get_grad(s)
        theta = self.superelevation.get(s)
        e_s = _normalize(s_vec)
        e_t = _normalize(
            (
                math.cos(theta) * -e_s[1] + math.sin(theta) * -e_s[2] * e_s[0],
                math.cos(theta) * e_s[0] + math.sin(theta) * -e_s[2] * e_s[1],
                math.sin(theta) * (e_s[0] * e_s[0] + e_s[1] * e_s[1]),
            )
        )
        e_h = _normalize(_cross(s_vec, e_t))
        p0 = self.ref_line.get_xyz(s)
        return (
            p0[0] + e_t[0] * t + e_h[0] * h,
            p0[1] + e_t[1] * t + e_h[1] * h,
            p0[2] + e_t[2] * t + e_h[2] * h,
        )

    def get_surface_pt(self, s: float, t: float) -> Vec3D:
        s = min(max(s, 0.0), self.length)
        ls = self.s_to_lanesection[self.get_lanesection_s0(s)]
        lane = ls.get_lane(s, t)
        inner = ls.get_lane(_next_towards_zero(lane.id))
        t_inner = inner.outer_border.get(s)
        if lane.level:
            h_t = -math.tan(self.crossfall.get_crossfall(s,
                            lane.id > 0)) * abs(t_inner)
            h_t += math.tan(self.superelevation.get(s)) * (t - t_inner)
        else:
            h_t = -math.tan(self.crossfall.get_crossfall(s,
                            lane.id > 0)) * abs(t)
        if lane.s_to_height_offset:
            keys = sorted(lane.s_to_height_offset)
            idx = max(0, _upper_key(keys, s) - 1)
            ho = lane.s_to_height_offset[keys[idx]]
            t_outer = lane.outer_border.get(s)
            p = (t - t_inner) / (t_outer - t_inner) if t_outer != t_inner else 0.0
            h_t += p * (ho.outer - ho.inner) + ho.inner
        return self.get_xyz(s, t, h_t)

    def get_lane_border_line(
        self, lane: Lane, eps: float, outer: bool = True
    ) -> Line3D:
        s0 = lane.lanesection_s0
        s1 = self.get_lanesection_end(s0)
        border = lane.outer_border
        if not outer:
            border = (
                self.s_to_lanesection[s0]
                .get_lane(_next_towards_zero(lane.id))
                .outer_border
            )
        vals = set(self.ref_line.approximate_linear(eps, s0, s1))
        vals.update(border.approximate_linear(eps, s0, s1))
        return [self.get_surface_pt(s, border.get(s)) for s in sorted(vals)]

    def get_lane_mesh(self, lane: Lane, eps: float) -> Mesh3D:
        s0 = lane.lanesection_s0
        s1 = self.get_lanesection_end(s0)
        ls = self.s_to_lanesection[s0]
        inner = ls.get_lane(_next_towards_zero(lane.id))
        vals = set(self.ref_line.approximate_linear(eps, s0, s1))
        vals.update(lane.outer_border.approximate_linear(eps, s0, s1))
        vals.update(inner.outer_border.approximate_linear(eps, s0, s1))
        mesh = Mesh3D()
        for s in sorted(vals):
            t_outer = lane.outer_border.get(s)
            t_inner = math.nextafter(inner.outer_border.get(s), t_outer)
            mesh.vertices.append(self.get_surface_pt(s, t_outer))
            mesh.vertices.append(self.get_surface_pt(s, t_inner))
            mesh.st_coordinates.append((s, t_outer))
            mesh.st_coordinates.append((s, t_inner))
        ccw = lane.id < 0
        for idx in range(3, len(mesh.vertices), 2):
            if ccw:
                mesh.indices.extend(
                    [idx - 3, idx - 1, idx, idx - 3, idx, idx - 2])
            else:
                mesh.indices.extend(
                    [idx - 3, idx, idx - 1, idx - 3, idx - 2, idx])
        return mesh

    def get_roadmark_mesh(self, lane: Lane, roadmark: RoadMark, eps: float) -> Mesh3D:
        vals = set(
            self.ref_line.approximate_linear(
                eps, roadmark.s_start, roadmark.s_end)
        )
        vals.update(
            lane.outer_border.approximate_linear(
                eps, roadmark.s_start, roadmark.s_end)
        )
        mesh = Mesh3D()
        for s in sorted(vals):
            edge_a = lane.outer_border.get(
                s) + roadmark.width * 0.5 + roadmark.t_offset
            edge_b = edge_a - roadmark.width
            mesh.vertices.append(self.get_surface_pt(s, edge_a))
            mesh.vertices.append(self.get_surface_pt(s, edge_b))
            mesh.st_coordinates.append((s, edge_a))
            mesh.st_coordinates.append((s, edge_b))
        for idx in range(3, len(mesh.vertices), 2):
            mesh.indices.extend([idx - 3, idx, idx - 1, idx - 3, idx - 2, idx])
        return mesh


@dataclass(frozen=True)
class JunctionLaneLink:
    from_lane: int
    to_lane: int


@dataclass
class JunctionConnection:
    id: str
    incoming_road: str
    connecting_road: str
    contact_point: str
    lane_links: List[JunctionLaneLink] = field(default_factory=list)


@dataclass(frozen=True)
class JunctionPriority:
    high: str
    low: str


@dataclass
class JunctionController:
    id: str
    type: str
    sequence: int


@dataclass
class Junction:
    name: str
    id: str
    id_to_connection: Dict[str, JunctionConnection] = field(
        default_factory=dict)
    id_to_controller: Dict[str, JunctionController] = field(
        default_factory=dict)
    priorities: List[JunctionPriority] = field(default_factory=list)


@dataclass(frozen=True)
class RoutingGraphEdge:
    from_lane: LaneKey
    to_lane: LaneKey
    weight: float = 0.0


class RoutingGraph:
    """Directed weighted lane graph for successor/predecessor traversal."""

    def __init__(self) -> None:
        self.edges: List[RoutingGraphEdge] = []
        self.lane_key_to_successors: Dict[LaneKey,
                                          List[Tuple[LaneKey, float]]] = {}
        self.lane_key_to_predecessors: Dict[LaneKey,
                                            List[Tuple[LaneKey, float]]] = {}

    def add_edge(self, edge: RoutingGraphEdge) -> None:
        if edge.from_lane == edge.to_lane:
            return
        self.edges.append(edge)
        self.lane_key_to_successors.setdefault(edge.from_lane, []).append(
            (edge.to_lane, edge.weight)
        )
        self.lane_key_to_predecessors.setdefault(edge.to_lane, []).append(
            (edge.from_lane, edge.weight)
        )

    def get_lane_successors(self, lane_key: LaneKey) -> List[LaneKey]:
        return [k for k, _ in self.lane_key_to_successors.get(lane_key, [])]

    def get_lane_predecessors(self, lane_key: LaneKey) -> List[LaneKey]:
        return [k for k, _ in self.lane_key_to_predecessors.get(lane_key, [])]

    def shortest_path(self, from_lane: LaneKey, to_lane: LaneKey) -> List[LaneKey]:
        pq = [(0.0, from_lane)]
        dist = {from_lane: 0.0}
        prev: Dict[LaneKey, LaneKey] = {}
        while pq:
            d, lane = heapq.heappop(pq)
            if lane == to_lane:
                break
            if d != dist[lane]:
                continue
            for nxt, weight in self.lane_key_to_successors.get(lane, []):
                nd = d + weight
                if nd < dist.get(nxt, math.inf):
                    dist[nxt] = nd
                    prev[nxt] = lane
                    heapq.heappush(pq, (nd, nxt))
        if to_lane not in dist:
            return []
        path = [to_lane]
        while path[-1] != from_lane:
            path.append(prev[path[-1]])
        return list(reversed(path))


class OpenDriveMap:
    """Mutable OpenDRIVE road-network object.

    The map can be created empty with ``OpenDriveMap()`` and populated with
    methods such as :meth:`addRoad` and :meth:`addJunction`. Use
    :func:`readXodr` to load an existing ``.xodr`` file into an
    :class:`OpenDriveMap`.
    """

    def __init__(
        self,
        xodr_file: str | Path | None = None,
        center_map: bool = False,
        with_road_objects: bool = True,
        with_lateral_profile: bool = True,
        with_lane_height: bool = True,
        abs_z_for_for_local_road_obj_outline: bool = False,
        fix_spiral_edge_cases: bool = True,
        with_road_signals: bool = True,
    ):
        self.xodr_file = "" if xodr_file is None else str(xodr_file)
        self.proj4 = ""
        self.offset = ""
        self.x_offs = 0.0
        self.y_offs = 0.0
        self.id_to_road: Dict[str, Road] = {}
        self.id_to_junction: Dict[str, Junction] = {}
        self.root = ET.Element("OpenDRIVE")
        self.tree = ET.ElementTree(self.root)
        self.__sumoNet = sumolib.net.Net()
        self.xml_doc = self.tree
        self.xml_parse_result = True
        if xodr_file is not None:
            self.loadXodr(
                xodr_file,
                center_map=center_map,
                with_road_objects=with_road_objects,
                with_lateral_profile=with_lateral_profile,
                with_lane_height=with_lane_height,
                abs_z_for_for_local_road_obj_outline=(
                    abs_z_for_for_local_road_obj_outline
                ),
                fix_spiral_edge_cases=fix_spiral_edge_cases,
                with_road_signals=with_road_signals,
            )

    def loadXodr(
        self,
        xodr_file: str | Path,
        *,
        center_map: bool = False,
        with_road_objects: bool = True,
        with_lateral_profile: bool = True,
        with_lane_height: bool = True,
        abs_z_for_for_local_road_obj_outline: bool = False,
        fix_spiral_edge_cases: bool = True,
        with_road_signals: bool = True,
        clear: bool = True,
    ) -> "OpenDriveMap":
        """Load an OpenDRIVE file into this map and return ``self``.

        Args:
            xodr_file: Source OpenDRIVE ``.xodr`` path.
            center_map: Whether to subtract the mean plan-view geometry origin.
            with_road_objects: Whether to parse road objects.
            with_lateral_profile: Whether to parse superelevation/crossfall.
            with_lane_height: Whether to parse lane height records.
            abs_z_for_for_local_road_obj_outline: Whether local object outlines
                should be marked as absolute-z outlines.
            fix_spiral_edge_cases: Whether degenerate spirals become line/arc
                geometries.
            with_road_signals: Whether to parse road signals.
            clear: Whether to clear existing roads and junctions first.

        Returns:
            This :class:`OpenDriveMap` instance.
        """

        self.xodr_file = str(xodr_file)
        self.proj4 = ""
        self.offset = ""
        self.x_offs = 0.0
        self.y_offs = 0.0
        if clear:
            self.id_to_road.clear()
            self.id_to_junction.clear()
        self.tree = ET.parse(xodr_file)
        self.root = self.tree.getroot()
        self.xml_doc = self.tree
        self.xml_parse_result = True
        self._parse(
            center_map,
            with_road_objects,
            with_lateral_profile,
            with_lane_height,
            abs_z_for_for_local_road_obj_outline,
            fix_spiral_edge_cases,
            with_road_signals,
        )
        return self

    def addRoad(self, road: Road, *, replace: bool = False) -> Road:
        """Add a :class:`Road` to the map and return the stored road.

        If ``replace`` is false and the id already exists, ``"_dup"`` suffixes
        are appended, matching the reader's duplicate-id behavior.
        """

        road_id = road.id
        if not replace:
            while road_id in self.id_to_road:
                road_id += "_dup"
            if road_id != road.id:
                road.id = road_id
                road.ref_line.road_id = road_id
        self.id_to_road[road_id] = road
        if road.xml_node is None:
            road.xml_node = self._road_to_xml(road)
        if road.xml_node is not None and road.xml_node not in list(self.root):
            self.root.append(road.xml_node)

        return road

    def addJunction(self, junction: Junction, *, replace: bool = False) -> Junction:
        """Add a :class:`Junction` to the map and return the stored junction."""

        junction_id = junction.id
        if not replace:
            while junction_id in self.id_to_junction:
                junction_id += "_dup"
            if junction_id != junction.id:
                junction.id = junction_id
        self.id_to_junction[junction_id] = junction

        existing_junction_xml = next(
            (
                node
                for node in self.root.findall("junction")
                if node.get("id") == junction.id
            ),
            None,
        )
        if existing_junction_xml is None:
            self.root.append(self._junction_to_xml(junction))

        return junction

    def saveXodr(
        self,
        xodr_file: str | Path,
        *,
        encoding: str = "utf-8",
        xml_declaration: bool = True,
    ) -> Path:
        """Save the map's OpenDRIVE XML data to an ``.xodr`` file.

        This writes the XML tree currently attached to the map. Parsed Python
        objects are not fully re-serialized, so unchanged source metadata and
        unsupported OpenDRIVE elements are preserved when the map was loaded
        from a file.

        Args:
            xodr_file: Destination OpenDRIVE ``.xodr`` path.
            encoding: Output text encoding.
            xml_declaration: Whether to include the XML declaration.

        Returns:
            The destination path as a :class:`pathlib.Path`.
        """

        output_path = Path(xodr_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.tree.write(
            output_path,
            encoding=encoding,
            xml_declaration=xml_declaration,
        )
        return output_path

    def getRoad(self, id: str) -> Road:
        return self.id_to_road[id]

    def getRoads(self) -> List[Road]:
        return list(self.id_to_road.values())

    def getJunction(self, id: str) -> Junction:
        return self.id_to_junction[id]

    def getJunctions(self) -> List[Junction]:
        return list(self.id_to_junction.values())

    def getGeoProj(self) -> str:
        """Return the OpenDRIVE header ``geoReference`` projection string."""

        return self.proj4

    def getOffset(self) -> str:
        """Return the OpenDRIVE header ``offset`` string."""

        if self.offset:
            # convert string like dict to dict
            try:
                offset_dict = eval(self.offset)
                if isinstance(offset_dict, dict):
                    return offset_dict
            except Exception:
                pass
        return self.offset

    def getBoundary(self) -> Tuple[float, float, float, float]:
        """Return map bounds as ``(west, south, east, north)``.

        Header bounds are used when available. For manually-created maps with
        no header bounds, the bounds are estimated from reference-line samples.
        """

        header = self.root.find("header")
        if header is not None:
            values = [
                _float(header, "west", math.nan),
                _float(header, "south", math.nan),
                _float(header, "east", math.nan),
                _float(header, "north", math.nan),
            ]
            if all(math.isfinite(value) for value in values):
                return (values[0], values[1], values[2], values[3])

        xs: List[float] = []
        ys: List[float] = []
        for road in self.getRoads():
            if not road.ref_line.s0_to_geometry:
                continue
            samples = road.ref_line.approximate_linear(5.0, 0.0, road.length)
            for s in samples:
                x, y, _ = road.ref_line.get_xyz(s)
                xs.append(x)
                ys.append(y)
        if not xs or not ys:
            return (0.0, 0.0, 0.0, 0.0)
        return (min(xs), min(ys), max(xs), max(ys))

    def getSignal(self, id: str, road_id: Optional[str] = None) -> RoadSignal:
        """Return a signal by id, optionally scoped to one road."""

        roads = [self.getRoad(road_id)] if road_id is not None else self.getRoads()
        for road in roads:
            signal = road.id_to_signal.get(id)
            if signal is not None:
                return signal
        raise KeyError(id)

    def getSignals(self, road_id: Optional[str] = None) -> List[RoadSignal]:
        """Return all signals, optionally scoped to one road."""

        roads = [self.getRoad(road_id)] if road_id is not None else self.getRoads()
        return [
            signal
            for road in roads
            for signal in road.get_road_signals()
        ]

    def getLane(self, lane_id: Optional[int] = None) -> Lane:
        """Return a lane by road id and lane id, or by :class:`LaneKey`."""

        if lane_id is None:
            raise TypeError("lane_id must be provided")

        lanes = self.getLanes()
        for lane in lanes:
            if lane.id == lane_id:
                return lane
        raise KeyError(f"Lane with id {lane_id} not found")

    def getLanes(self, road_id: Optional[str] = None) -> List[Lane]:
        """Return lanes for the map, a road, or a road section at ``s``."""

        roads = [self.getRoad(road_id)] if road_id is not None else self.getRoads()
        lanes: List[Lane] = []
        for road in roads:
            sections = road.get_lanesections()
            for section in sections:
                lanes.extend(section.get_lanes())
        return lanes

    def convertXY2LonLat(self, x: float, y: float) -> Tuple[float, float]:
        """Convert from map-local XY coordinates to longitude and latitude."""

        # USE sumolib's implementation for better accuracy, especially for larger maps.
        return self.__sumoNet.convertXY2LonLat(x=x, y=y)

    def convertLonLat2XY(self, lon: float, lat: float) -> Tuple[float, float]:
        """Convert from longitude and latitude to map-local XY coordinates."""

        # USE sumolib's implementation for better accuracy, especially for larger maps.
        return self.__sumoNet.convertLonLat2XY(lon=lon, lat=lat)

    def getRoadNetworkMesh(self, eps: float) -> RoadNetworkMesh:
        out = RoadNetworkMesh()
        for road in self.id_to_road.values():
            out.lanes_mesh.road_start_indices[len(
                out.lanes_mesh.vertices)] = road.id
            out.roadmarks_mesh.road_start_indices[len(out.roadmarks_mesh.vertices)] = (
                road.id
            )
            for ls in road.get_lanesections():
                out.lanes_mesh.lanesec_start_indices[len(out.lanes_mesh.vertices)] = (
                    ls.s0
                )
                out.roadmarks_mesh.lanesec_start_indices[
                    len(out.roadmarks_mesh.vertices)
                ] = ls.s0
                for lane in ls.get_lanes():
                    out.lanes_mesh.lane_start_indices[len(out.lanes_mesh.vertices)] = (
                        lane.id
                    )
                    out.lanes_mesh.add_mesh(road.get_lane_mesh(lane, eps))
                    out.roadmarks_mesh.lane_start_indices[
                        len(out.roadmarks_mesh.vertices)
                    ] = lane.id
                    for roadmark in lane.get_roadmarks(
                        ls.s0, road.get_lanesection_end(ls)
                    ):
                        out.roadmarks_mesh.roadmark_type_start_indices[
                            len(out.roadmarks_mesh.vertices)
                        ] = roadmark.type
                        out.roadmarks_mesh.add_mesh(
                            road.get_roadmark_mesh(lane, roadmark, eps)
                        )
            out.road_objects_mesh.road_start_indices[
                len(out.road_objects_mesh.vertices)
            ] = road.id
            for obj in road.id_to_object.values():
                out.road_objects_mesh.road_object_start_indices[
                    len(out.road_objects_mesh.vertices)
                ] = obj.id
            out.road_signals_mesh.road_start_indices[
                len(out.road_signals_mesh.vertices)
            ] = road.id
            for sig in road.id_to_signal.values():
                out.road_signals_mesh.road_signal_start_indices[
                    len(out.road_signals_mesh.vertices)
                ] = sig.id
        return out

    def getRoutingGraph(self) -> RoutingGraph:
        graph = RoutingGraph()
        for road in self.id_to_road.values():
            sections = road.get_lanesections()
            for idx, ls in enumerate(sections):
                prev_ls = (
                    sections[idx - 1]
                    if idx > 0
                    else self._linked_lanesection(road.predecessor)
                )
                next_ls = (
                    sections[idx + 1]
                    if idx + 1 < len(sections)
                    else self._linked_lanesection(road.successor)
                )
                for lane in ls.get_lanes():
                    if lane.id == 0:
                        continue
                    follows_road = lane.id < 0
                    pred_ls = prev_ls if follows_road else next_ls
                    succ_ls = next_ls if follows_road else prev_ls
                    pred_id = lane.predecessor if follows_road else lane.successor
                    succ_id = lane.successor if follows_road else lane.predecessor
                    if pred_ls and pred_id in pred_ls.id_to_lane:
                        graph.add_edge(
                            RoutingGraphEdge(
                                pred_ls.id_to_lane[pred_id].key,
                                lane.key,
                                road.get_lanesection_length(ls),
                            )
                        )
                    if succ_ls and succ_id in succ_ls.id_to_lane:
                        graph.add_edge(
                            RoutingGraphEdge(
                                lane.key,
                                succ_ls.id_to_lane[succ_id].key,
                                road.get_lanesection_length(ls),
                            )
                        )
        for junction in self.id_to_junction.values():
            for conn in junction.id_to_connection.values():
                incoming = self.id_to_road.get(conn.incoming_road)
                connecting = self.id_to_road.get(conn.connecting_road)
                if not incoming or not connecting:
                    continue
                incoming_ls = (
                    incoming.get_lanesections()[-1]
                    if incoming.successor.type == "junction"
                    and incoming.successor.id == junction.id
                    else incoming.get_lanesections()[0]
                )
                connecting_ls = (
                    connecting.get_lanesections()[0]
                    if conn.contact_point == "start"
                    else connecting.get_lanesections()[-1]
                )
                for link in conn.lane_links:
                    if (
                        link.from_lane in incoming_ls.id_to_lane
                        and link.to_lane in connecting_ls.id_to_lane
                    ):
                        graph.add_edge(
                            RoutingGraphEdge(
                                incoming_ls.id_to_lane[link.from_lane].key,
                                connecting_ls.id_to_lane[link.to_lane].key,
                                incoming.get_lanesection_length(incoming_ls),
                            )
                        )
        return graph

    def _parse(
        self,
        center_map: bool,
        with_road_objects: bool,
        with_lateral_profile: bool,
        with_lane_height: bool,
        abs_z_outline: bool,
        fix_spiral_edge_cases: bool,
        with_road_signals: bool,
    ) -> None:
        header = self.root.find("header")
        if header is not None and header.find("geoReference") is not None:
            self.proj4 = (header.findtext("geoReference") or "").strip()
        if header is not None and header.find("offset") is not None:
            self.offset = (header.find("offset").attrib or "")

        # Set location for __sumoNet so lon/lat conversion has projection and
        # netOffset metadata. OpenDRIVE offset is optional; SUMO still expects a
        # netOffset key, so default to 0,0 when the header omits <offset>.
        if self.proj4:
            x_off, y_off = _offset_xy(self.offset)
            self.__sumoNet.setLocation(
                netOffset=f"{x_off},{y_off}",
                convBoundary=None,
                origBoundary=None,
                projParameter=_projection_for_sumo(self.proj4),
            )

        if center_map:
            xs, ys = [], []
            for geom in self.root.findall("./road/planView/geometry"):
                xs.append(_float(geom, "x"))
                ys.append(_float(geom, "y"))
            self.x_offs = sum(xs) / len(xs) if xs else 0.0
            self.y_offs = sum(ys) / len(ys) if ys else 0.0
        self._parse_junctions()
        for road_node in self.root.findall("road"):
            road_id = road_node.get("id", "")
            road = Road(
                road_id,
                max(0.0, _float(road_node, "length")),
                road_node.get("junction", ""),
                road_node.get("name", ""),
                road_node.get("rule", "RHT").lower() == "lht",
            )
            road.xml_node = road_node
            self.addRoad(road)
            self._parse_road_links(road, road_node)
            self._parse_types(road, road_node)
            self._parse_ref_line(road, road_node, fix_spiral_edge_cases)
            self._parse_splines(road, road_node, with_lateral_profile)
            self._parse_lanes(road, road_node, with_lane_height)
            if with_road_objects:
                self._parse_objects(road, road_node, abs_z_outline)
            if with_road_signals:
                self._parse_signals(road, road_node)

    def _parse_junctions(self) -> None:
        for jn in self.root.findall("junction"):
            jid = jn.get("id", "")
            junction = Junction(jn.get("name", ""), jid)
            for cn in jn.findall("connection"):
                cp = cn.get("contactPoint", "start")
                conn = JunctionConnection(
                    cn.get("id", ""),
                    cn.get("incomingRoad", ""),
                    cn.get("connectingRoad", ""),
                    cp if cp in {"start", "end"} else "start",
                )
                for ln in cn.findall("laneLink"):
                    conn.lane_links.append(
                        JunctionLaneLink(_int(ln, "from"), _int(ln, "to"))
                    )
                junction.id_to_connection[conn.id] = conn
            for pn in jn.findall("priority"):
                junction.priorities.append(
                    JunctionPriority(pn.get("high", ""), pn.get("low", ""))
                )
            for ctrl in jn.findall("controller"):
                controller = JunctionController(
                    ctrl.get("id", ""), ctrl.get(
                        "type", ""), _int(ctrl, "sequence")
                )
                junction.id_to_controller[controller.id] = controller
            self.addJunction(junction)

    def _parse_road_links(self, road: Road, road_node: ET.Element) -> None:
        link = road_node.find("link")
        if link is None:
            return
        for name in ("predecessor", "successor"):
            node = link.find(name)
            if node is None:
                continue
            typ = node.get("elementType", "road")
            cp = node.get("contactPoint", "none")
            setattr(
                road,
                name,
                RoadLink(
                    node.get("elementId", ""),
                    typ if typ in {"road", "junction"} else "road",
                    cp if cp in {"start", "end"} else "none",
                ),
            )
        for nb in link.findall("neighbor"):
            road.neighbors.append(
                RoadNeighbor(
                    nb.get("elementId", ""), nb.get(
                        "side", ""), nb.get("direction", "")
                )
            )

    def _parse_types(self, road: Road, road_node: ET.Element) -> None:
        for type_node in road_node.findall("type"):
            s = max(0.0, _float(type_node, "s"))
            road.s_to_type[s] = type_node.get("type", "")
            speed = type_node.find("speed")
            if speed is not None:
                road.s_to_speed[s] = SpeedRecord(
                    speed.get("max", ""), speed.get("unit", "")
                )

    def _parse_ref_line(
        self, road: Road, road_node: ET.Element, fix_spiral_edge_cases: bool
    ) -> None:
        for hdr in road_node.findall("./planView/geometry"):
            s0 = max(0.0, _float(hdr, "s"))
            x0 = _float(hdr, "x") - self.x_offs
            y0 = _float(hdr, "y") - self.y_offs
            hdg0 = _float(hdr, "hdg")
            length = max(0.0, _float(hdr, "length"))
            child = next(iter(list(hdr)), None)
            if child is None:
                continue
            if child.tag == "line":
                geom: RoadGeometry = Line(s0, x0, y0, hdg0, length)
            elif child.tag == "arc":
                geom = Arc(s0, x0, y0, hdg0, length,
                           _float(child, "curvature"))
            elif child.tag == "spiral":
                c0, c1 = _float(child, "curvStart"), _float(child, "curvEnd")
                if fix_spiral_edge_cases and abs(c0) < 1e-6 and abs(c1) < 1e-6:
                    geom = Line(s0, x0, y0, hdg0, length)
                elif fix_spiral_edge_cases and abs(c1 - c0) < 1e-6:
                    geom = Arc(s0, x0, y0, hdg0, length, c0)
                else:
                    geom = Spiral(s0, x0, y0, hdg0, length, c0, c1)
            elif child.tag == "paramPoly3":
                prange = (
                    child.get("pRange") or hdr.get("pRange") or "normalized"
                ).lower()
                geom = ParamPoly3(
                    s0,
                    x0,
                    y0,
                    hdg0,
                    length,
                    _float(child, "aU"),
                    _float(child, "bU"),
                    _float(child, "cU"),
                    _float(child, "dU"),
                    _float(child, "aV"),
                    _float(child, "bV"),
                    _float(child, "cV"),
                    _float(child, "dV"),
                    prange != "arclength",
                )
            else:
                continue
            geom.xml_node = child
            road.ref_line.s0_to_geometry[s0] = geom

    def _parse_splines(
        self, road: Road, road_node: ET.Element, with_lateral_profile: bool
    ) -> None:
        targets = [
            (road.ref_line.elevation_profile, "./elevationProfile/elevation"),
            (road.lane_offset, "./lanes/laneOffset"),
        ]
        if with_lateral_profile:
            targets.append(
                (road.superelevation, "./lateralProfile/superelevation"))
        for spline, path in targets:
            spline.s0_to_poly[0.0] = Poly3.from_odr(0.0, 0.0, 0.0, 0.0, 0.0)
            for node in road_node.findall(path):
                s0 = max(0.0, _float(node, "s"))
                spline.s0_to_poly[s0] = Poly3.from_odr(
                    s0,
                    _float(node, "a"),
                    _float(node, "b"),
                    _float(node, "c"),
                    _float(node, "d"),
                )
        if with_lateral_profile:
            for node in road_node.findall("./lateralProfile/crossfall"):
                s0 = max(0.0, _float(node, "s"))
                road.crossfall.s0_to_poly[s0] = Poly3.from_odr(
                    s0,
                    _float(node, "a"),
                    _float(node, "b"),
                    _float(node, "c"),
                    _float(node, "d"),
                )
                road.crossfall.sides[s0] = node.get("side", "both").lower()

    def _parse_lanes(
        self, road: Road, road_node: ET.Element, with_lane_height: bool
    ) -> None:
        for ls_node in road_node.findall("./lanes/laneSection"):
            s0 = _float(ls_node, "s")
            ls = LaneSection(road.id, s0)
            ls.xml_node = ls_node
            road.s_to_lanesection[s0] = ls
            for lane_node in ls_node.findall(".//lane"):
                lane_id = _int(lane_node, "id")
                lane = Lane(
                    road.id,
                    s0,
                    lane_id,
                    _bool(lane_node, "level"),
                    lane_node.get("type", ""),
                )
                lane.xml_node = lane_node
                pred = lane_node.find("./link/predecessor")
                succ = lane_node.find("./link/successor")
                if pred is not None:
                    lane.predecessor = _int(pred, "id")
                if succ is not None:
                    lane.successor = _int(succ, "id")
                for width in lane_node.findall("width"):
                    s_offset = max(0.0, _float(width, "sOffset"))
                    poly = Poly3.from_odr(
                        s0 + s_offset,
                        _float(width, "a"),
                        _float(width, "b"),
                        _float(width, "c"),
                        _float(width, "d"),
                    )
                    if lane_id == 0 and not poly.is_zero():
                        poly = Poly3.from_odr(
                            s0 + s_offset, 0.0, 0.0, 0.0, 0.0)
                    lane.lane_width.s0_to_poly[s0 + s_offset] = poly
                if lane_id == 0 and not lane.lane_width.s0_to_poly:
                    lane.lane_width.s0_to_poly[s0] = Poly3.from_odr(
                        s0, 0.0, 0.0, 0.0, 0.0
                    )
                if with_lane_height:
                    for height in lane_node.findall("height"):
                        lane.s_to_height_offset[
                            s0 + max(0.0, _float(height, "sOffset"))
                        ] = HeightOffset(
                            _float(height, "inner"), _float(height, "outer")
                        )
                self._parse_roadmarks(lane, lane_node, s0)
                ls.id_to_lane[lane_id] = lane
            self._derive_lane_borders(road, ls)

    def _parse_roadmarks(
        self, lane: Lane, lane_node: ET.Element, lanesection_s0: float
    ) -> None:
        for rm in lane_node.findall("roadMark"):
            group = RoadMarkGroup(
                lane.road_id,
                lanesection_s0,
                lane.id,
                _float(rm, "width", -1.0),
                _float(rm, "height"),
                max(0.0, _float(rm, "sOffset")),
                rm.get("type", "none"),
                rm.get("weight", "standard"),
                rm.get("color", "standard"),
                rm.get("material", "standard"),
                rm.get("laneChange", "both"),
            )
            type_node = rm.find("type")
            if type_node is not None:
                name = type_node.get("name", "")
                default_width = _float(type_node, "width", -1.0)
                for line in type_node.findall("line"):
                    group.roadmark_lines.append(
                        RoadMarksLine(
                            lane.road_id,
                            lanesection_s0,
                            lane.id,
                            lanesection_s0 + group.s_offset,
                            _float(line, "width", default_width),
                            max(0.0, _float(line, "length")),
                            max(0.0, _float(line, "space")),
                            _float(line, "tOffset"),
                            max(0.0, _float(line, "sOffset")),
                            name,
                            line.get("rule", "none"),
                        )
                    )
            lane.roadmark_groups.append(group)

    def _derive_lane_borders(self, road: Road, ls: LaneSection) -> None:
        if 0 not in ls.id_to_lane:
            raise RuntimeError("lane section does not have lane #0")
        positives = sorted(i for i in ls.id_to_lane if i > 0)
        previous: Optional[CubicSpline] = None
        for lane_id in positives:
            lane = ls.id_to_lane[lane_id]
            lane.outer_border = (
                lane.lane_width if previous is None else previous.add(
                    lane.lane_width)
            )
            previous = lane.outer_border
        negatives = sorted((i for i in ls.id_to_lane if i < 0), reverse=True)
        previous = None
        for lane_id in negatives:
            lane = ls.id_to_lane[lane_id]
            width = lane.lane_width.negate()
            lane.outer_border = width if previous is None else previous.add(
                width)
            previous = lane.outer_border
        ls.id_to_lane[0].outer_border = ls.id_to_lane[0].lane_width
        for lane in ls.id_to_lane.values():
            lane.outer_border = lane.outer_border.add(road.lane_offset)

    def _validities(self, node: ET.Element) -> List[LaneValidityRecord]:
        records = []
        for valid in node.findall("validity"):
            frm, to = (
                _int(valid, "fromLane", -(2**31)),
                _int(valid, "toLane", 2**31 - 1),
            )
            if frm > to:
                frm = to = 0
            records.append(LaneValidityRecord(frm, to))
        return records

    def _parse_objects(
        self, road: Road, road_node: ET.Element, abs_z_outline: bool
    ) -> None:
        local_type = "local_abs_z" if abs_z_outline else "local_rel_z"
        for obj_node in road_node.findall("./objects/object"):
            oid = obj_node.get("id", "")
            while oid in road.id_to_object:
                oid += "_dup"
            obj = RoadObject(
                road.id,
                oid,
                max(0.0, _float(obj_node, "s")),
                _float(obj_node, "t"),
                _float(obj_node, "zOffset"),
                max(0.0, _float(obj_node, "length")),
                max(0.0, _float(obj_node, "validLength")),
                max(0.0, _float(obj_node, "width")),
                max(0.0, _float(obj_node, "radius")),
                _float(obj_node, "height"),
                _float(obj_node, "hdg"),
                _float(obj_node, "pitch"),
                _float(obj_node, "roll"),
                obj_node.get("type", ""),
                obj_node.get("name", ""),
                obj_node.get("orientation", ""),
                obj_node.get("subtype", ""),
                obj_node.get("dynamic", "no") == "yes",
            )
            for rpt in obj_node.findall("repeat"):
                obj.repeats.append(
                    RoadObjectRepeat(
                        _float(rpt, "s", math.nan),
                        max(0.0, _float(rpt, "length")),
                        max(0.0, _float(rpt, "distance")),
                        _float(rpt, "tStart", math.nan),
                        _float(rpt, "tEnd", math.nan),
                        _float(rpt, "widthStart", math.nan),
                        _float(rpt, "widthEnd", math.nan),
                        _float(rpt, "heightStart", math.nan),
                        _float(rpt, "heightEnd", math.nan),
                        _float(rpt, "zOffsetStart", math.nan),
                        _float(rpt, "zOffsetEnd", math.nan),
                    )
                )
            outlines_parent = obj_node.find("outlines")
            if outlines_parent is None:
                outlines_parent = obj_node
            for outn in outlines_parent.findall("outline"):
                outline = RoadObjectOutline(
                    _int(outn, "id", -1),
                    outn.get("fillType", ""),
                    outn.get("laneType", ""),
                    _bool(outn, "outer", True),
                    _bool(outn, "closed", True),
                )
                for corner in outn.findall("cornerLocal"):
                    outline.outline.append(
                        RoadObjectCorner(
                            _int(corner, "id", -1),
                            (
                                _float(corner, "u"),
                                _float(corner, "v"),
                                _float(corner, "z"),
                            ),
                            _float(corner, "height"),
                            local_type,
                        )
                    )
                for corner in outn.findall("cornerRoad"):
                    outline.outline.append(
                        RoadObjectCorner(
                            _int(corner, "id", -1),
                            (
                                _float(corner, "s"),
                                _float(corner, "t"),
                                _float(corner, "dz"),
                            ),
                            _float(corner, "height"),
                            "road",
                        )
                    )
                obj.outlines.append(outline)
            obj.lane_validities = self._validities(obj_node)
            road.id_to_object[oid] = obj

    def _parse_signals(self, road: Road, road_node: ET.Element) -> None:
        for sig_node in road_node.findall("./signals/signal"):
            sid = sig_node.get("id", "")
            while sid in road.id_to_signal:
                sid += "_dup"
            sig = RoadSignal(
                road.id,
                sid,
                sig_node.get("name", ""),
                max(0.0, _float(sig_node, "s")),
                _float(sig_node, "t"),
                _bool(sig_node, "dynamic"),
                _float(sig_node, "zOffset"),
                _float(sig_node, "value"),
                max(0.0, _float(sig_node, "height")),
                max(0.0, _float(sig_node, "width")),
                _float(sig_node, "hOffset"),
                _float(sig_node, "pitch"),
                _float(sig_node, "roll"),
                sig_node.get("orientation", "none"),
                sig_node.get("country", ""),
                sig_node.get("type", "none"),
                sig_node.get("subtype", "none"),
                sig_node.get("unit", ""),
                sig_node.get("text", "none"),
            )
            sig.lane_validities = self._validities(sig_node)
            road.id_to_signal[sid] = sig

    def _linked_lanesection(self, link: RoadLink) -> Optional[LaneSection]:
        if link.type != "road" or link.contact_point not in {"start", "end"}:
            return None
        road = self.id_to_road.get(link.id)
        if not road or not road.s_to_lanesection:
            return None
        sections = road.get_lanesections()
        return sections[0] if link.contact_point == "start" else sections[-1]

    def _road_to_xml(self, road: Road) -> ET.Element:
        """Serialize the small subset used by this tutorial."""

        road_node = ET.Element(
            "road",
            {
                "name": road.name,
                "length": str(road.length),
                "id": road.id,
                "junction": road.junction,
                "rule": "RHT",
            },
        )

        link_node = ET.SubElement(road_node, "link")
        if road.predecessor.type != "none":
            ET.SubElement(
                link_node,
                "predecessor",
                {
                    "elementType": road.predecessor.type,
                    "elementId": road.predecessor.id,
                    "contactPoint": road.predecessor.contact_point,
                },
            )
        if road.successor.type != "none":
            ET.SubElement(
                link_node,
                "successor",
                {
                    "elementType": road.successor.type,
                    "elementId": road.successor.id,
                    "contactPoint": road.successor.contact_point,
                },
            )

        for s, road_type in sorted(road.s_to_type.items()):
            road_type_node = ET.SubElement(
                road_node,
                "type",
                {"s": str(s), "type": road_type},
            )
            speed = road.s_to_speed.get(s)
            if speed is not None:
                ET.SubElement(
                    road_type_node,
                    "speed",
                    {"max": speed.max, "unit": speed.unit},
                )

        plan_view = ET.SubElement(road_node, "planView")
        for geom in road.ref_line.get_geometries():
            geom_node = ET.SubElement(
                plan_view,
                "geometry",
                {
                    "s": str(geom.s0),
                    "x": str(geom.x0),
                    "y": str(geom.y0),
                    "hdg": str(geom.hdg0),
                    "length": str(geom.length),
                },
            )
            if isinstance(geom, Arc):
                ET.SubElement(geom_node, "arc", {"curvature": str(geom.curvature)})
            elif isinstance(geom, Spiral):
                ET.SubElement(
                    geom_node,
                    "spiral",
                    {
                        "curvStart": str(geom.curv_start),
                        "curvEnd": str(geom.curv_end),
                    },
                )
            elif isinstance(geom, ParamPoly3):
                ET.SubElement(
                    geom_node,
                    "paramPoly3",
                    {
                        "aU": str(geom.aU),
                        "bU": str(geom.bU),
                        "cU": str(geom.cU),
                        "dU": str(geom.dU),
                        "aV": str(geom.aV),
                        "bV": str(geom.bV),
                        "cV": str(geom.cV),
                        "dV": str(geom.dV),
                        "pRange": "normalized"
                        if geom.pRange_normalized
                        else "arcLength",
                    },
                )
            else:
                ET.SubElement(geom_node, "line")

        elevation_profile = ET.SubElement(road_node, "elevationProfile")
        ET.SubElement(
            elevation_profile,
            "elevation",
            {"s": "0", "a": "0", "b": "0", "c": "0", "d": "0"},
        )

        if road.s_to_lanesection:
            lanes_node = ET.SubElement(road_node, "lanes")
            if road.lane_offset.s0_to_poly:
                for s, poly in sorted(road.lane_offset.s0_to_poly.items()):
                    ET.SubElement(
                        lanes_node,
                        "laneOffset",
                        {
                            "s": str(s),
                            "a": str(poly.get(s)),
                            "b": str(poly.get_grad(s)),
                            "c": str(poly.c),
                            "d": str(poly.d),
                        },
                    )
            for section in road.get_lanesections():
                section_node = ET.SubElement(
                    lanes_node,
                    "laneSection",
                    {"s": str(section.s0)},
                )
                lane_groups = (
                    ("left", sorted(i for i in section.id_to_lane if i > 0)),
                    ("center", [i for i in section.id_to_lane if i == 0]),
                    ("right", sorted((i for i in section.id_to_lane if i < 0), reverse=True)),
                )
                for side_name, lane_ids in lane_groups:
                    if not lane_ids:
                        continue
                    side_node = ET.SubElement(section_node, side_name)
                    for lane_id in lane_ids:
                        lane = section.get_lane(lane_id)
                        lane_node = ET.SubElement(
                            side_node,
                            "lane",
                            {
                                "id": str(lane.id),
                                "type": lane.type,
                                "level": str(lane.level).lower(),
                            },
                        )
                        if lane.id != 0:
                            for s, poly in sorted(lane.lane_width.s0_to_poly.items()):
                                ET.SubElement(
                                    lane_node,
                                    "width",
                                    {
                                        "sOffset": str(max(0.0, s - section.s0)),
                                        "a": str(poly.get(s)),
                                        "b": str(poly.get_grad(s)),
                                        "c": str(poly.c),
                                        "d": str(poly.d),
                                    },
                                )
                        for group in lane.roadmark_groups:
                            ET.SubElement(
                                lane_node,
                                "roadMark",
                                {
                                    "sOffset": str(group.s_offset),
                                    "type": group.type,
                                    "weight": group.weight,
                                    "color": group.color,
                                    "laneChange": group.lane_change,
                                },
                            )

        if road.id_to_object:
            objects_node = ET.SubElement(road_node, "objects")
            for obj in road.id_to_object.values():
                ET.SubElement(
                    objects_node,
                    "object",
                    {
                        "id": obj.id,
                        "s": str(obj.s0),
                        "t": str(obj.t0),
                        "zOffset": str(obj.z0),
                        "length": str(obj.length),
                        "width": str(obj.width),
                        "height": str(obj.height),
                        "type": obj.type,
                        "name": obj.name,
                    },
                )

        if road.id_to_signal:
            signals_node = ET.SubElement(road_node, "signals")
            for signal in road.id_to_signal.values():
                ET.SubElement(
                    signals_node,
                    "signal",
                    {
                        "id": signal.id,
                        "name": signal.name,
                        "s": str(signal.s0),
                        "t": str(signal.t0),
                        "dynamic": str(signal.is_dynamic).lower(),
                        "zOffset": str(signal.zOffset),
                        "value": str(signal.value),
                        "height": str(signal.height),
                        "width": str(signal.width),
                        "orientation": signal.orientation,
                        "country": signal.country,
                        "type": signal.type,
                        "subtype": signal.subtype,
                        "unit": signal.unit,
                        "text": signal.text,
                    },
                )

        return road_node

    def _junction_to_xml(self, junction: Junction) -> ET.Element:
        junction_node = ET.Element("junction", {"name": junction.name, "id": junction.id})
        for connection in junction.id_to_connection.values():
            connection_node = ET.SubElement(
                junction_node,
                "connection",
                {
                    "id": connection.id,
                    "incomingRoad": connection.incoming_road,
                    "connectingRoad": connection.connecting_road,
                    "contactPoint": connection.contact_point,
                },
            )
            for lane_link in connection.lane_links:
                ET.SubElement(
                    connection_node,
                    "laneLink",
                    {"from": str(lane_link.from_lane), "to": str(lane_link.to_lane)},
                )
        for priority in junction.priorities:
            ET.SubElement(junction_node, "priority", {"high": priority.high, "low": priority.low})
        for controller in junction.id_to_controller.values():
            ET.SubElement(
                junction_node,
                "controller",
                {
                    "id": controller.id,
                    "type": controller.type,
                    "sequence": str(controller.sequence),
                },
            )
        return junction_node


def readXodr(xodr_file: str | Path, **kwargs) -> OpenDriveMap:
    """Load an OpenDRIVE file into an :class:`OpenDriveMap`.

    Args:
        xodr_file: Source OpenDRIVE ``.xodr`` path.
        **kwargs: Keyword arguments forwarded to
            :meth:`OpenDriveMap.load_xodr`.

    Returns:
        The parsed :class:`OpenDriveMap`.
    """

    return OpenDriveMap().loadXodr(xodr_file, **kwargs)


__all__ = [
    "Arc",
    "CubicSpline",
    "Junction",
    "JunctionConnection",
    "JunctionController",
    "JunctionLaneLink",
    "JunctionPriority",
    "Lane",
    "LaneKey",
    "LaneSection",
    "Line",
    "Mesh3D",
    "OpenDriveMap",
    "ParamPoly3",
    "Poly3",
    "RefLine",
    "Road",
    "RoadNetworkMesh",
    "RoutingGraph",
    "RoutingGraphEdge",
    "Spiral",
    "readXodr",
]
