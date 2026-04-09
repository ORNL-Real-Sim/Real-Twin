
#!/usr/bin/env python3
"""
sumo_net2matchup_lane_config.py

Build a "matchup table" from a SUMO .net.xml file, matching junction approaches to outgoing edges
and movements, with computed bearings at each junction.

Output columns:
    - Junction ID
    - Coordinate (lat, lon)
    - Bearing (degrees, 0..360)
    - FromEdge
    - ToEdge
    - Movement  (one of: right, thru, left, Uturn; or None if unknown)
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from pyproj import Proj, Transformer


class SumoMatchupBuilder:
    def __init__(self, network_path: str | Path):
        self.network_path = Path(network_path)
        self.root = self._load_xml()
        self.utm_proj, self.wgs84_proj, self.net_offset = self._init_projection()

    # ---------- XML / Projection ----------
    def _load_xml(self) -> ET.Element:
        if not self.network_path.exists():
            raise FileNotFoundError(f"Network file not found: {self.network_path}")
        try:
            return ET.parse(str(self.network_path)).getroot()
        except ET.ParseError as e:
            raise ValueError(f"Failed to parse XML: {e}") from e

    def _init_projection(self) -> Tuple[Proj, Proj, Tuple[float, float]]:
        """Initialize UTM and WGS84 projections from <location> tag and read net offsets."""
        location = self.root.find("location")
        # Defaults
        net_offset_x, net_offset_y = 0.0, 0.0
        utm_zone: Optional[int] = None

        if location is not None:
            # netOffset="x,y"
            net_offset_x, net_offset_y = map(float, location.get("netOffset", "0,0").split(","))
            # projParameter may contain +proj=utm +zone=...
            proj_params = location.get("projParameter", "")
            for token in proj_params.split():
                if token.startswith("+zone="):
                    try:
                        utm_zone = int(token.split("=")[1])
                    except ValueError:
                        pass

        # Build Proj instances
        if utm_zone is None:
            # Fallback: attempt generic UTM zone 16 (typical for TN), but warn via comment
            utm_zone = 16
        utm_proj = Proj(proj="utm", zone=utm_zone, ellps="WGS84", datum="WGS84", units="m", no_defs=True)
        wgs84_proj = Proj(proj="latlong", datum="WGS84")
        return utm_proj, wgs84_proj, (net_offset_x, net_offset_y)

    # ---------- Helpers ----------
    def _convert_xy_to_latlon(self, x: float, y: float) -> Tuple[float, float]:
        ox, oy = self.net_offset
        lon, lat = Transformer.from_proj(self.utm_proj, self.wgs84_proj, always_xy=True).transform(x - ox, y - oy)
        return lat, lon

    @staticmethod
    def _bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        dlon = math.radians(lon2 - lon1)
        lat1r = math.radians(lat1)
        lat2r = math.radians(lat2)
        x = math.sin(dlon) * math.cos(lat2r)
        y = math.cos(lat1r) * math.sin(lat2r) - math.sin(lat1r) * math.cos(lat2r) * math.cos(dlon)
        initial = math.degrees(math.atan2(x, y))
        return (initial + 360.0) % 360.0

    # ---------- Data extraction ----------
    def _edges_df(self) -> pd.DataFrame:
        rows: List[Dict[str, str]] = []
        for edge in self.root.findall("edge"):
            edge_id = edge.get("id")
            if not edge_id:
                continue
            rows.append({
                "Edge ID": edge_id,
                "From": edge.get("from", ""),
                "To": edge.get("to", ""),
                "function": edge.get("function", ""),
            })
        return pd.DataFrame(rows)

    def _junction_center_latlon(self, junction: ET.Element) -> Tuple[Optional[float], Optional[float]]:
        """Compute an approximate centroid (lat, lon) from the junction's 'shape' polygon if present."""
        shape = junction.get("shape")
        if not shape:
            return None, None
        parts = shape.split()
        xs = [float(p.split(",")[0]) for p in parts]
        ys = [float(p.split(",")[1]) for p in parts]
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        lat, lon = self._convert_xy_to_latlon(cx, cy)
        return lat, lon

    def _incoming_edges_for_junction(self, junction: ET.Element) -> List[str]:
        inc_lanes = junction.get("incLanes", "").split()
        # Extract base edge id from lane id "edge_0", "edge_1", ... by removing the suffix after last underscore
        edges = sorted({lane.rsplit("_", 1)[0] for lane in inc_lanes if lane})
        return edges

    def _approach_bearings(self, junction_id: str, edges: List[str]) -> List[Dict]:
        """For each incoming edge, take the first lane, last two shape points, and compute bearing into the junction."""
        out: List[Dict] = []
        for edge in self.root.findall("edge"):
            edge_id = edge.get("id")
            if edge_id not in edges:
                continue
            lane = edge.find("lane")
            if lane is None:
                continue
            shape = lane.get("shape", "").split()
            if len(shape) < 2:
                continue
            # take last two points of the lane polyline
            x2, y2 = map(float, shape[-1].split(",")[:2])
            x1, y1 = map(float, shape[-2].split(",")[:2])
            lat1, lon1 = self._convert_xy_to_latlon(x1, y1)
            lat2, lon2 = self._convert_xy_to_latlon(x2, y2)
            bearing = self._bearing_deg(lat1, lon1, lat2, lon2)
            out.append({
                "Junction ID": junction_id,
                "Approach Edge": edge_id,
                "Degree": round(bearing, 2),
                # runway bearing coarse class (replicating notebook behavior)
                "Runway Bearing": int(round(bearing / 10.0) * 10 / 10),
            })
        return out

    def _connections_df(self) -> pd.DataFrame:
        """Build mapping from FromEdge -> (ToEdge, Movement) using <connection> tags and dir codes."""
        direction_mapping = {
            "s": "thru", "l": "left", "L": "left",
            "r": "right", "R": "right",
            "t": "Uturn", "invalid": "invalid"
        }
        rows = []
        for conn in self.root.findall("connection"):
            rows.append({
                "FromEdge": conn.get("from", ""),
                "ToEdge": conn.get("to", ""),
                "Movement": direction_mapping.get(conn.get("dir"), None)
            })
        return pd.DataFrame(rows)

    # ---------- Public API ----------
    def build_matchup_table(self) -> pd.DataFrame:
        """
        Build the matchup table replicating the notebook's final 'MatchupTable' structure.
        """
        edges_df = self._edges_df()

        # entrance/exit counts per junction
        # (Used to filter junctions where there are meaningful approaches/exits)
        # The notebook used: keep junctions with entrance_count >= 2 or exit_count >= 2
        entrance_counts = edges_df.groupby("To").size().rename("Entrance").reset_index()
        exit_counts = edges_df.groupby("From").size().rename("Exit").reset_index()

        # Prepare a quick lookup
        entr = dict(zip(entrance_counts["To"], entrance_counts["Entrance"]))
        ex = dict(zip(exit_counts["From"], exit_counts["Exit"]))

        # Compute approach bearings per eligible junction
        approach_rows: List[Dict] = []
        for junction in self.root.findall("junction"):
            jid = junction.get("id", "")
            entrance = entr.get(jid, 0)
            exit_ = ex.get(jid, 0)
            if entrance >= 2 or exit_ >= 2:
                inc_edges = self._incoming_edges_for_junction(junction)
                approach_rows.extend(self._approach_bearings(jid, inc_edges))

        junction_bearing = pd.DataFrame(approach_rows)
        if junction_bearing.empty:
            # Return empty table with expected columns
            return pd.DataFrame(columns=["Junction ID", "Coordinate", "Bearing", "FromEdge", "ToEdge", "Movement"])

        # Add junction coordinates
        coords: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
        for j in self.root.findall("junction"):
            lat, lon = self._junction_center_latlon(j)
            coords[j.get("id", "")] = (lat, lon)

        jb = junction_bearing.copy()
        jb["Coordinate"] = jb["Junction ID"].map(coords)
        jb = jb.rename(columns={"Approach Edge": "FromEdge", "Degree": "Bearing"})

        # Merge with connections to get ToEdge and Movement
        conn = self._connections_df()
        matchup = jb.merge(conn, on="FromEdge", how="left")

        # Clean up, sort, and final columns
        matchup["Junction ID Numeric"] = matchup["Junction ID"].astype(str)
        direction_order = {"right": 1, "thru": 2, "left": 3, "Uturn": 4}
        matchup["Direction Order"] = matchup["Movement"].map(direction_order)

        matchup = matchup.sort_values(
            by=["Junction ID Numeric", "Bearing", "Direction Order"],
            ascending=[True, True, True],
            na_position="last"
        ).reset_index(drop=True)

        # Drop helpers
        matchup = matchup.drop(columns=[c for c in ["Junction ID Numeric", "Direction Order", "Runway Bearing"] if c in matchup.columns])

        # Final column order
        desired = ["Junction ID", "Coordinate", "Bearing", "FromEdge", "ToEdge", "Movement"]
        missing = [c for c in desired if c not in matchup.columns]
        for m in missing:
            matchup[m] = None
        matchup = matchup[desired]

        matchup = matchup.drop_duplicates().reset_index(drop=True)

        matchup['approach_name'] = matchup['Bearing'].apply(self.determine_direction_guanhao)
        matchup_summary = matchup.groupby(by=['Junction ID', 'Bearing'], as_index=False).agg({'approach_name': 'first'})
        matchup_summary = matchup_summary.sort_values(by=['Junction ID', 'Bearing']).reset_index(drop=True)

        matchup_summary["approach_name_rank"] = matchup_summary.groupby(["Junction ID", "approach_name"]).cumcount()
        matchup_summary['approach_name_unique'] = np.where(matchup_summary["approach_name_rank"] >= 1,
                                                           matchup_summary['approach_name'] + matchup_summary['approach_name_rank'].astype(str),
                                                           matchup_summary['approach_name'])
        matchup_summary = matchup_summary.drop(columns=['approach_name', 'approach_name_rank'])

        matchup = pd.merge(matchup, matchup_summary, how='left', left_on=['Junction ID', 'Bearing'], right_on=['Junction ID', 'Bearing'])

        movement_dict = {'right': 'R', 'thru': 'T', 'left': 'L', 'Uturn': 'U'}

        matchup['movement_name'] = matchup['approach_name_unique'] + matchup['Movement'].map(movement_dict)

        print('Test')
        return matchup

    def determine_direction_guanhao(self, angle):
        """
        Convert an angle into a compass direction.
        0° = North, positive = clockwise.

        Parameters:
        angle (float): Angle in degrees, ideally between -360 and 360.

        Returns:
        str: Compass direction label (NB, NE, EB, etc.), or None if invalid.
        """
        if angle is None:
            return None

        if angle <= 22.5 or angle >= 337.5:
            return 'NB'
        elif 22.5 < angle < 67.5:
            return 'NE'
        elif 67.5 <= angle <= 112.5:
            return 'EB'
        elif 112.5 < angle < 157.5:
            return 'SE'
        elif 157.5 <= angle <= 202.5:
            return 'SB'
        elif 202.5 < angle < 247.5:
            return 'SW'
        elif 247.5 <= angle <= 292.5:
            return 'WB'
        elif 292.5 < angle < 337.5:
            return 'NW'
        return None


if __name__ == "__main__":
    # Example usage
    # network_path = "Nashville.net.xml"
    # network_path = "Nashville_full_0827.net.xml"
    # network_path = "Nashville_full_1007_1.net.xml"
    network_path = r'datasets/MLK/MLK_final_elevation_20251009.net.xml'
    builder = SumoMatchupBuilder(network_path)
    table = builder.build_matchup_table()
    # Print a brief preview
    print(table.head(20).to_string(index=False))




