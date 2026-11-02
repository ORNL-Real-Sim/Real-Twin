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

"""Helpers for converting between OpenDRIVE and SUMO networks.

This module uses SUMO's ``netconvert`` binary for OpenDRIVE/SUMO conversion and
returns Python objects for callers.

Attributes:
    _SOURCE_NET_FILE_ATTR: Attribute name used to remember the generated SUMO
        ``.net.xml`` file on returned ``sumolib.net.Net`` objects.
    _TEMP_DIR_ATTR: Attribute name used to retain temporary directories while
        returned objects may still need their generated files.
"""

# Copyright 2026 Xiangyong Luo
#
# This file is part of pyopendrive.
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

import contextlib
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, TYPE_CHECKING
import xml.etree.ElementTree as ET
from .xodr_reader import OpenDriveMap, readXodr

if TYPE_CHECKING:
    import sumolib.net

_SOURCE_NET_FILE_ATTR = "_pyopendrive_sumo_net_file"
_TEMP_DIR_ATTR = "_pyopendrive_temp_dir"
_ORIGINAL_NODE_ID_PARAM = "pyopendrive.original_node_id"
_ORIGINAL_LINK_ID_PARAM = "pyopendrive.original_link_id"
_ORIGINAL_LANE_ID_PARAM = "pyopendrive.original_lane_id"
_ORIGINAL_LANE_INDEX_PARAM = "pyopendrive.original_lane_index"
_FROM_NODE_ID_PARAM = "pyopendrive.from_node_id"
_TO_NODE_ID_PARAM = "pyopendrive.to_node_id"
_SUMO_SIGNALS_CODE = "realtwin.sumo_signals"


def _require_sumolib():
    try:
        import sumolib.net  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on local SUMO install
        raise RuntimeError(
            "sumolib is required for SUMO conversion. Install SUMO and make its "
            "tools directory importable, or install a compatible sumolib package."
        ) from exc
    return sumolib.net


def _netconvert_binary(binary: str | Path | None = None) -> str:
    if binary is not None:
        path = shutil.which(str(binary)) or str(binary)
    else:
        path = shutil.which("netconvert") or ""
    if not path:
        raise RuntimeError(
            "SUMO netconvert was not found on PATH. Pass netconvert_binary=... "
            "or add SUMO's bin directory to PATH."
        )
    return path


def _append_options(command: list[str], options: Mapping[str, Any] | None) -> None:
    if not options:
        return
    for key, value in options.items():
        option = key if key.startswith("-") else f"--{key}"
        if value is None or value is False:
            continue
        if value is True:
            command.append(option)
        elif isinstance(value, (list, tuple)):
            command.append(option)
            command.append(",".join(str(item) for item in value))
        else:
            command.extend([option, str(value)])


def _run_netconvert(command: Sequence[str]) -> None:
    proc = subprocess.run(
        list(command),
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        detail = "\n".join(part for part in (proc.stdout, proc.stderr) if part)
        raise RuntimeError(f"netconvert failed with code {proc.returncode}:\n{detail}")


def _make_output_path(path: str | Path | None, suffix: str) -> tuple[Path, str | None]:
    if path is not None:
        return Path(path), None
    temp_dir = tempfile.mkdtemp(prefix="pyopendrive_sumo_")
    return Path(temp_dir) / suffix, temp_dir


def xodr_to_net_xml(
    xodr_file: str | Path,
    net_file: str | Path | None = None,
    *,
    netconvert_binary: str | Path | None = None,
    netconvert_options: Mapping[str, Any] | None = None,
    read_options: Mapping[str, Any] | None = None,
    import_all_lanes: bool = True,
    with_internal: bool = True,
) -> sumolib.net.Net:
    """Convert an OpenDRIVE ``.xodr`` file to a ``sumolib.net.Net`` object.

    Files exported by :func:`xodr_from_net_xml` carry metadata for restoring
    source SUMO junction, edge and lane IDs, TLS programs and controlled lane
    connections. OpenDrive uses local integer lane IDs; SUMO lane names are
    retained separately in ``userData``. For other OpenDrive files, native
    generated IDs are retained and their OpenDrive origins are recorded as
    parameters. Roads split by netconvert retain generated segment IDs.

    Args:
        xodr_file: Source OpenDRIVE file.
        net_file: output SUMO ``.net.xml`` path.
        netconvert_binary: Optional path or executable name for SUMO
            ``netconvert``. When omitted, ``netconvert`` is resolved from
            ``PATH``.
        netconvert_options: Extra command-line options for ``netconvert`` as
            ``{"option": value}``.
        read_options: Extra keyword arguments passed to
            ``sumolib.net.readNet``.
        import_all_lanes: Whether to pass ``--opendrive.import-all-lanes``.
        with_internal: Value passed as ``withInternal`` to
            ``sumolib.net.readNet`` unless overridden by ``read_options``.

    Returns:
        sumolib.net.Net: The converted network object.
            The generated SUMO net XML file is retained and its path is attached to the returned object as an attribute.

    Raises:
        RuntimeError: If ``sumolib`` or ``netconvert`` is unavailable, or if
            ``netconvert`` exits with an error.
        ValueError: If the output would overwrite the input, original-name
            metadata is disabled, or controlled lane connections cannot be
            restored after conversion.
    """

    sumo_net = _require_sumolib()
    output_file, temp_dir = _make_output_path(net_file, "network.net.xml")
    if output_file.resolve() == Path(xodr_file).resolve():
        raise ValueError("SUMO output must differ from the source OpenDrive file.")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    command = [
        _netconvert_binary(netconvert_binary),
        "--opendrive-files",
        str(xodr_file),
        "--output-file",
        str(output_file),
    ]
    if import_all_lanes:
        command.append("--opendrive.import-all-lanes")
    options = {key.lstrip("-"): value for key, value in (netconvert_options or {}).items()}
    if options.get("output.original-names", True) is False:
        raise ValueError("output.original-names is required to preserve OpenDrive IDs.")
    options["output.original-names"] = True
    _append_options(command, options)
    # netconvert interprets a leading '-' in road IDs as a direction marker.
    # Use private numeric road IDs during import; metadata restores the source IDs.
    with tempfile.TemporaryDirectory(prefix="realtwin_xodr_import_") as import_dir:
        import_file = Path(import_dir) / "network.xodr"
        import_tree = ET.parse(xodr_file)
        import_root = import_tree.getroot()
        if any(_get_xodr_user_data(road, _ORIGINAL_LINK_ID_PARAM) for road in import_root.findall("road")):
            road_ids = {}
            for index, road in enumerate(import_root.findall("road"), start=1):
                road_ids[road.get("id")] = str(index)
                road.set("id", str(index))
            _restore_road_references(import_root, road_ids)
        _write_xml(import_tree, import_file)
        command[2] = str(import_file)
        _run_netconvert(command)
        _annotate_sumo_net_ids(output_file, import_file)

    read_kwargs = {"withInternal": with_internal}
    if read_options:
        read_kwargs |= read_options
    net = sumo_net.readNet(str(output_file), **read_kwargs)
    setattr(net, _SOURCE_NET_FILE_ATTR, str(output_file))
    if temp_dir is not None:
        setattr(net, _TEMP_DIR_ATTR, temp_dir)
    return net


def xodr_from_net_xml(
    net_file: str | Path | None = None,
    xodr_file: str | Path | None = None,
    *,
    net: sumolib.net.Net | None = None,
    netconvert_binary: str | Path | None = None,
    netconvert_options: Mapping[str, Any] | None = None,
    opendrive_map_kwargs: Mapping[str, Any] | None = None,
) -> OpenDriveMap:
    """Convert a ``.net.xml`` file or ``sumolib.net.Net`` object to :class:`pyopendrive.OpenDriveMap`.

    ``sumolib.net.Net`` does not provide a native writer.  If ``net_file`` is not
    passed, this function first reuses the source file attached by
    :func:`xodr_to_net_xml`. If no source path is available, it writes a
    minimal SUMO net XML representation from the in-memory object and passes that
    to ``netconvert``.

    The source SUMO file is never edited. Road and junction identities are
    restored from netconvert's source metadata, independently of XML order.
    Source lane IDs and signal programs are retained in OpenDrive ``userData``
    for an explicit conversion back into a separate SUMO output.

    Args:
        net_file: Optional SUMO ``.net.xml`` path to use as the source for
            ``netconvert``.
        xodr_file: Optional output OpenDRIVE ``.xodr`` path. When omitted, a
            temporary file is created and retained through an attribute on the
            returned map.
        net: Optional ``sumolib.net.Net`` object to convert. if net_file is not provided,
            this is required to resolve the source net.
        netconvert_binary: Optional path or executable name for SUMO
            ``netconvert``. When omitted, ``netconvert`` is resolved from
            ``PATH``.
        netconvert_options: Extra command-line options for ``netconvert`` as
            ``{"option": value}``.
        opendrive_map_kwargs: Extra keyword arguments passed to
            :class:`pyopendrive.OpenDriveMap`.

    Returns:
        A :class:`pyopendrive.OpenDriveMap` loaded from the generated
        OpenDRIVE file.

    Raises:
        RuntimeError: If ``netconvert`` is unavailable or exits with an error.
        ValueError: If the output would overwrite the input or required source
            edge mappings are missing.
    """

    source_net_file, temp_dir = _resolve_sumo_net_file(net, net_file)
    output_file, output_temp_dir = _make_output_path(xodr_file, "network.xodr")
    if output_file.resolve() == source_net_file.resolve():
        raise ValueError("OpenDrive output must differ from the source SUMO file.")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    command = [
        _netconvert_binary(netconvert_binary),
        "--sumo-net-file",
        str(source_net_file),
        "--opendrive-output",
        str(output_file),
    ]
    options = {key.lstrip("-"): value for key, value in (netconvert_options or {}).items()}
    if options.get("output.original-names", True) is False:
        raise ValueError("output.original-names is required to preserve SUMO IDs.")
    options["output.original-names"] = True
    _append_options(command, options)
    _run_netconvert(command)
    _restore_opendrive_ids(output_file, source_net_file)

    kwargs = dict(opendrive_map_kwargs or {})
    odr_map = readXodr(output_file, **kwargs)
    odr_map.sumo_net_file = str(source_net_file)
    odr_map.xodr_file = str(output_file)
    if temp_dir is not None:
        odr_map._pyopendrive_sumo_temp_dir = temp_dir
    if output_temp_dir is not None:
        odr_map._pyopendrive_xodr_temp_dir = output_temp_dir
    return odr_map


def _resolve_sumo_net_file(net, net_file: str | Path | None) -> tuple[Path, str | None]:
    if net_file is not None:
        return Path(net_file), None

    attached = getattr(net, _SOURCE_NET_FILE_ATTR, None)
    if attached:
        attached_path = Path(attached)
        if attached_path.exists():
            return attached_path, None

    output_file, temp_dir = _make_output_path(None, "in_memory.net.xml")
    _write_minimal_sumo_net_xml(net, output_file)
    return output_file, temp_dir


def _write_minimal_sumo_net_xml(net, output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    root = ET.Element("net", version=str(getattr(net, "getVersion", lambda: "")() or ""))

    boundary = _safe_call(net, "getBoundary") or _safe_call(net, "getBBoxXY")
    if boundary and len(boundary) >= 4:
        min_x, min_y, max_x, max_y = (float(v) for v in boundary[:4])
        ET.SubElement(
            root,
            "location",
            {
                "netOffset": "0.00,0.00",
                "convBoundary": f"{min_x:.2f},{min_y:.2f},{max_x:.2f},{max_y:.2f}",
                "origBoundary": f"{min_x:.2f},{min_y:.2f},{max_x:.2f},{max_y:.2f}",
                "projParameter": "!",
            },
        )

    for node in net.getNodes():
        x, y = node.getCoord()
        ET.SubElement(
            root,
            "junction",
            {
                "id": str(node.getID()),
                "type": str(node.getType() or "priority"),
                "x": f"{float(x):.6f}",
                "y": f"{float(y):.6f}",
            },
        )

    for edge in net.getEdges():
        if edge.getFunction():
            continue
        edge_elem = ET.SubElement(
            root,
            "edge",
            {
                "id": str(edge.getID()),
                "from": str(edge.getFromNode().getID()),
                "to": str(edge.getToNode().getID()),
                "priority": str(edge.getPriority()),
                "speed": f"{float(edge.getSpeed()):.6f}",
            },
        )
        edge_type = edge.getType()
        if edge_type:
            edge_elem.set("type", str(edge_type))
        for idx, lane in enumerate(edge.getLanes()):
            lane_shape = " ".join(
                f"{float(x):.6f},{float(y):.6f}" for x, y in lane.getShape()
            )
            attrs = {
                "id": str(lane.getID()),
                "index": str(idx),
                "speed": f"{float(lane.getSpeed()):.6f}",
                "length": f"{float(lane.getLength()):.6f}",
                "width": f"{float(lane.getWidth()):.6f}",
            }
            if lane_shape:
                attrs["shape"] = lane_shape
            ET.SubElement(edge_elem, "lane", attrs)

    for edge in net.getEdges():
        if edge.getFunction():
            continue
        for to_edge, connections in edge.getOutgoing().items():
            if to_edge.getFunction():
                continue
            for conn in connections:
                ET.SubElement(
                    root,
                    "connection",
                    {
                        "from": str(edge.getID()),
                        "to": str(to_edge.getID()),
                        "fromLane": str(conn.getFromLane().getIndex()),
                        "toLane": str(conn.getToLane().getIndex()),
                    },
                )

    ET.ElementTree(root).write(output_file, encoding="utf-8", xml_declaration=True)


def _annotate_sumo_net_ids(net_file: Path, xodr_file: Path) -> None:
    """Restore source identities when the OpenDrive file contains our metadata."""
    tree = ET.parse(net_file)
    root = tree.getroot()
    xodr_root = ET.parse(xodr_file).getroot()
    roads = {road.get("id"): road for road in xodr_root.findall("road")}
    node_id_map = {
        junction.get("id"): _get_xodr_user_data(junction, _ORIGINAL_NODE_ID_PARAM)
        for junction in xodr_root.findall("junction")
        if _get_xodr_user_data(junction, _ORIGINAL_NODE_ID_PARAM)
    }
    # origId on imported lanes identifies the OpenDrive road and local lane.
    edge_roads = {}
    for edge in root.findall("edge"):
        if edge.get("function") == "internal":
            continue
        lane = edge.find("lane")
        origin = _get_param(lane, "origId") if lane is not None else None
        if origin:
            edge_roads[edge.get("id")] = origin.split()[0].rsplit("_", 1)[0]
    road_counts = Counter(edge_roads.values())
    edge_id_map: dict[str, str] = {}
    lane_id_map: dict[str, str] = {}
    lane_index_map: dict[tuple[str, str], str] = {}
    used_edge_ids: set[str] = set()
    for road_id, road in roads.items():
        original_id = _get_xodr_user_data(road, _ORIGINAL_LINK_ID_PARAM)
        if original_id and road_counts[road_id] == 1:
            used_edge_ids.add(original_id)
    used_lane_ids: set[str] = set()
    for edge in root.findall("edge"):
        edge_id = edge.get("id")
        if not edge_id or edge.get("function") == "internal":
            continue
        road_id = edge_roads.get(edge_id)
        road = roads.get(road_id)
        original_id = _get_xodr_user_data(road, _ORIGINAL_LINK_ID_PARAM)
        one_to_one = road_counts[road_id] == 1
        new_id = original_id if original_id and one_to_one else edge_id
        if original_id and one_to_one:
            edge_id_map[edge_id] = new_id
        else:
            edge_id_map[edge_id] = _unique_id(new_id, used_edge_ids)
        edge.set("id", edge_id_map[edge_id])
        _set_param(edge, _ORIGINAL_LINK_ID_PARAM, original_id or road_id or edge_id)
        if original_id and one_to_one:
            for attr, key in (("from", _FROM_NODE_ID_PARAM), ("to", _TO_NODE_ID_PARAM)):
                source_node = _get_xodr_user_data(road, key)
                if source_node:
                    node_id_map[edge.get(attr)] = source_node
        road_lanes = {}
        if road is not None:
            road_lanes = {
                lane.get("id"): lane
                for lane in road.findall("./lanes/laneSection/*/lane")
            }
        for lane in edge.findall("lane"):
            old_lane_id = lane.get("id", "")
            origin = _get_param(lane, "origId") or ""
            local_lane = origin.split()[0].rsplit("_", 1)[-1] if origin else ""
            road_lane = road_lanes.get(local_lane)
            original_lane = _get_xodr_user_data(road_lane, _ORIGINAL_LANE_ID_PARAM)
            original_index = _get_xodr_user_data(road_lane, _ORIGINAL_LANE_INDEX_PARAM)
            new_lane_id = (
                original_lane if original_lane and one_to_one else old_lane_id
            )
            lane_id_map[old_lane_id] = _unique_id(new_lane_id, used_lane_ids)
            if original_index and one_to_one:
                lane_index_map[(edge_id, lane.get("index"))] = original_index
                lane.set("index", original_index)
            lane.set("id", lane_id_map[old_lane_id])
            _set_param(
                lane, _ORIGINAL_LANE_ID_PARAM,
                original_lane or origin or old_lane_id,
            )
            if original_id and local_lane:
                _set_param(lane, "origId", f"{original_id}_{local_lane}")
        edge[:] = sorted(edge, key=lambda child: int(child.get("index", "0")))

    # Internal edges follow their junction IDs, including lane lists and via refs.
    for edge in root.findall("edge"):
        if edge.get("function") != "internal":
            continue
        old_id = edge.get("id", "")
        new_id = old_id
        old_node, separator, connection_id = old_id[1:].rpartition("_")
        if separator and old_node in node_id_map:
            new_id = f":{node_id_map[old_node]}_{connection_id}"
        edge_id_map[old_id] = _unique_id(new_id, used_edge_ids)
        edge.set("id", edge_id_map[old_id])
        for lane in edge.findall("lane"):
            old_lane_id = lane.get("id", "")
            new_lane_id = _replace_edge_id_prefix(old_lane_id, old_id, edge_id_map[old_id])
            lane_id_map[old_lane_id] = _unique_id(new_lane_id, used_lane_ids)
            lane.set("id", lane_id_map[old_lane_id])
    for connection in root.findall("connection"):
        for attr, lane_attr in (("from", "fromLane"), ("to", "toLane")):
            key = (connection.get(attr), connection.get(lane_attr))
            if key in lane_index_map:
                connection.set(lane_attr, lane_index_map[key])
    for element in root:
        if element.tag == "edge":
            for attr in ("from", "to"):
                if element.get(attr) in node_id_map:
                    element.set(attr, node_id_map[element.get(attr)])
        elif element.tag == "junction":
            node_id = element.get("id", "")
            element.set("id", node_id_map.get(node_id, node_id))
            _set_param(element, _ORIGINAL_NODE_ID_PARAM, element.get("id"))
        elif element.tag == "connection":
            for attr in ("from", "to"):
                if element.get(attr) in edge_id_map:
                    element.set(attr, edge_id_map[element.get(attr)])
        for attr in ("incLanes", "intLanes", "via"):
            if attr in element.attrib:
                element.set(
                    attr,
                    " ".join(lane_id_map.get(value, value)
                             for value in element.get(attr).split()),
                )
    _restore_sumo_signals(root, xodr_root)
    _write_xml(tree, net_file)


def _restore_sumo_signals(root: ET.Element, xodr_root: ET.Element) -> None:
    """Restore programs only when all their controlled lane connections survive."""
    signals = xodr_root.find(f"userData[@code='{_SUMO_SIGNALS_CODE}']")
    if signals is None:
        return
    attrs = ("from", "to", "fromLane", "toLane")
    source_connections = {
        tuple(conn.get(attr) for attr in attrs): conn
        for conn in signals.findall("sumoConnection")
    }
    imported_connections = {
        tuple(conn.get(attr) for attr in attrs): conn
        for conn in root.findall("connection")
    }
    imported_controlled = {
        key for key, conn in imported_connections.items() if conn.get("tl")
    }
    if (not source_connections.keys() <= imported_connections.keys()
            or not imported_controlled <= source_connections.keys()):
        raise ValueError(
            "Cannot restore SUMO signal IDs: controlled lane connections "
            "changed during OpenDrive conversion."
        )
    for key, source in source_connections.items():
        connection = imported_connections[key]
        for attr in ("tl", "linkIndex", "linkIndex2"):
            if attr in source.attrib:
                connection.set(attr, source.get(attr))
            else:
                connection.attrib.pop(attr, None)
    used_tls = {conn.get("tl") for conn in root.findall("connection") if conn.get("tl")}
    source_tls = {logic.get("id") for logic in signals.findall("sumoTlLogic")}
    for logic in root.findall("tlLogic"):
        if logic.get("id") in source_tls or logic.get("id") not in used_tls:
            root.remove(logic)
    for logic in signals.findall("sumoTlLogic"):
        restored_logic = deepcopy(logic)
        restored_logic.tag = "tlLogic"
        root.append(restored_logic)


def _replace_edge_id_prefix(value: str, old_edge_id: str, new_edge_id: str) -> str:
    if value == old_edge_id:
        return new_edge_id
    prefix = f"{old_edge_id}_"
    if value.startswith(prefix):
        return f"{new_edge_id}_{value[len(prefix):]}"
    return value


def _restore_opendrive_ids(xodr_file: Path, source_net_file: Path) -> None:
    source_root = ET.parse(source_net_file).getroot()
    edges = {edge.get("id"): edge for edge in source_root.findall("edge")}
    tree = ET.parse(xodr_file)
    root = tree.getroot()
    roads = root.findall("road")
    # Reserve external IDs first so connector road IDs cannot steal them.
    used_ids = {
        edge_id for edge_id, edge in edges.items()
        if edge.get("function") != "internal"
    }
    road_id_map: dict[str, str] = {}
    for road in roads:
        old_id = road.get("id", "")
        source_id = road.get("name")
        if road.get("junction", "-1") == "-1":
            source_id = next(
                (data.get("value") for data in road.findall("userData")
                 if data.get("code") == "sumoId"),
                None,
            )
            if source_id not in edges:
                raise ValueError(
                    f"No source SUMO edge mapping for OpenDrive road {old_id!r}."
                )
            new_id = source_id
            _set_xodr_user_data(road, _ORIGINAL_LINK_ID_PARAM, source_id)
            for attr, key in (("from", _FROM_NODE_ID_PARAM), ("to", _TO_NODE_ID_PARAM)):
                _set_xodr_user_data(road, key, edges[source_id].get(attr, ""))
        else:
            new_id = _unique_id(source_id or old_id, used_ids)
        road.set("id", new_id)
        if old_id:
            road_id_map[old_id] = new_id
        source_edge = edges.get(source_id)
        if source_edge is not None:
            source_lanes = sorted(
                source_edge.findall("lane"),
                key=lambda lane: int(lane.get("index", "0")),
            )
            # SUMO lane 0 is rightmost; OpenDrive right lanes count outwards from -1.
            for lane in road.findall("./lanes/laneSection/*/lane"):
                if lane.get("id") == "0":
                    continue
                index = len(source_lanes) - abs(int(lane.get("id")))
                if 0 <= index < len(source_lanes):
                    source_lane = source_lanes[index]
                    _set_xodr_user_data(
                        lane, _ORIGINAL_LANE_ID_PARAM, source_lane.get("id")
                    )
                    _set_xodr_user_data(
                        lane, _ORIGINAL_LANE_INDEX_PARAM, source_lane.get("index")
                    )
    _restore_road_references(root, road_id_map)
    node_ids = {junction.get("id") for junction in source_root.findall("junction")}
    junction_id_map = {}
    for junction in root.findall("junction"):
        source_id = junction.get("name")
        if source_id in node_ids:
            junction_id_map[junction.get("id")] = source_id
            junction.set("id", source_id)
            _set_xodr_user_data(junction, _ORIGINAL_NODE_ID_PARAM, source_id)
    for road in roads:
        if road.get("junction") in junction_id_map:
            road.set("junction", junction_id_map[road.get("junction")])
        for link in road.findall("./link/*"):
            if (link.get("elementType") == "junction"
                    and link.get("elementId") in junction_id_map):
                link.set("elementId", junction_id_map[link.get("elementId")])
    signals = ET.SubElement(root, "userData", {"code": _SUMO_SIGNALS_CODE})
    for logic in source_root.findall("tlLogic"):
        saved_logic = deepcopy(logic)
        saved_logic.tag = "sumoTlLogic"
        signals.append(saved_logic)
    for connection in source_root.findall("connection"):
        if connection.get("tl"):
            saved_connection = deepcopy(connection)
            saved_connection.tag = "sumoConnection"
            signals.append(saved_connection)
    _write_xml(tree, xodr_file)


def _restore_road_references(root: ET.Element, road_id_map: Mapping[str, str]) -> None:
    for road in root.findall("road"):
        for link in road.findall("./link/*"):
            if link.get("elementType") == "road":
                element_id = link.get("elementId")
                if element_id is not None and element_id in road_id_map:
                    link.set("elementId", road_id_map[element_id])

    for connection in root.findall("./junction/connection"):
        for attr in ("incomingRoad", "connectingRoad"):
            road_id = connection.get(attr)
            if road_id is not None and road_id in road_id_map:
                connection.set(attr, road_id_map[road_id])

    for priority in root.findall("./junction/priority"):
        for attr in ("high", "low"):
            road_id = priority.get(attr)
            if road_id is not None and road_id in road_id_map:
                priority.set(attr, road_id_map[road_id])


def _get_xodr_user_data(parent: ET.Element | None, key: str) -> str | None:
    if parent is None:
        return None
    return next(
        (vector.get("value") for vector in parent.findall("./userData/vector")
         if vector.get("key") == key),
        None,
    )


def _set_param(parent: ET.Element, key: str, value: str) -> None:
    for param in parent.findall("param"):
        if param.get("key") == key:
            param.set("value", value)
            return
    ET.SubElement(parent, "param", {"key": key, "value": value})


def _get_param(parent: ET.Element, key: str) -> str | None:
    return next(
        (
            param.get("value")
            for param in parent.findall("param")
            if param.get("key") == key
        ),
        None,
    )


def _set_xodr_user_data(parent: ET.Element, key: str, value: str) -> None:
    user_data = parent.find("userData")
    if user_data is None:
        user_data = ET.SubElement(parent, "userData")
    for vector in user_data.findall("vector"):
        if vector.get("key") == key:
            vector.set("value", value)
            return
    ET.SubElement(user_data, "vector", {"key": key, "value": value})


def _unique_id(id_value: str, used_ids: set[str]) -> str:
    candidate = id_value
    suffix = 1
    while candidate in used_ids:
        suffix += 1
        candidate = f"{id_value}.{suffix}"
    used_ids.add(candidate)
    return candidate


def _write_xml(tree: ET.ElementTree[Any], output_file: Path) -> None:
    with contextlib.suppress(AttributeError):
        ET.indent(tree, space="    ")
    tree.write(output_file, encoding="utf-8", xml_declaration=True)


def _safe_call(obj, name: str):
    method = getattr(obj, name, None)
    if method is None:
        return None
    try:
        return method()
    except Exception:
        return None


__all__ = [
    "xodr_from_net_xml",
    "xodr_to_net_xml",
]
