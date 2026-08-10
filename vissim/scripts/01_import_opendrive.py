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
"""Stage 1: SUMO net -> OpenDRIVE -> Vissim, then inspect what Vissim built.

Run this first.  It converts the RealTwin updated SUMO network to OpenDRIVE with
``netconvert``, imports it into Vissim over COM, saves the ``.inpx``, and dumps
the link/connector inventory plus the derived junction-and-turn table so the
renumbering can be inspected.

Usage::

    python vissim/scripts/01_import_opendrive.py \
        --net datasets/chattanooga/updated_net/chatt.net.xml \
        --name chatt \
        --outdir vissim/work/chattanooga

Add ``--skip-netconvert`` to reuse an existing ``.xodr``, or ``--visible`` to
watch Vissim do the import.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# Make `rt_vissim` importable when running this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from rt_vissim.com import VissimSession, available_progids  # noqa: E402
from rt_vissim.network import (  # noqa: E402
    DEFAULT_JUNCTION_RADIUS, JUNCTION_SOURCES, extract_network,
    read_opendrive_junction_names)


def read_net_offset(net_path: Path) -> tuple[float, float]:
    """Return the ``netOffset`` recorded in a SUMO network.

    SUMO normalises coordinates so the network sits near the origin and stores
    the shift it applied in ``<location netOffset="-x,-y">``.  Negating it
    recovers the true projected (UTM) origin.

    Args:
        net_path: Path to the SUMO ``.net.xml``.

    Returns:
        ``(offset_x, offset_y)`` to add to bring the network back to true UTM,
        or ``(0.0, 0.0)`` when the network has no offset.
    """
    root = ET.parse(net_path).getroot()
    location = root.find("location")
    if location is None:
        return 0.0, 0.0
    raw = location.get("netOffset", "0,0")
    try:
        dx, dy = (float(v) for v in raw.split(","))
    except ValueError:
        return 0.0, 0.0
    return -dx, -dy


def run_netconvert(net_path: Path, xodr_path: Path, *, georeference: bool = True) -> Path:
    """Convert a SUMO network to OpenDRIVE.

    Two flags matter here:

    ``--output.original-names``
        Keeps the SUMO edge names in the OpenDRIVE roads.  Vissim surfaces them
        in its link names, which is how :mod:`rt_vissim.network` recovers the
        junction grouping after Vissim renumbers everything.

    ``--offset.x`` / ``--offset.y``
        Puts the network back at its true UTM coordinates.  Without this,
        netconvert writes SUMO's normalised local coordinates and parks the true
        origin in ``<header><offset>`` -- which the Vissim OpenDRIVE importer
        ignores, landing the network next to (0, 0) instead of at the site.

    Args:
        net_path: Path to the SUMO ``.net.xml``.
        xodr_path: Path to write the ``.xodr`` to.
        georeference: Whether to shift the network back to true UTM.

    Returns:
        The written ``.xodr`` path.

    Raises:
        FileNotFoundError: If ``net_path`` does not exist.
        RuntimeError: If ``netconvert`` fails or is not on PATH.
    """
    if not net_path.exists():
        raise FileNotFoundError(f"SUMO network not found: {net_path}")
    xodr_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "netconvert",
        "-s", str(net_path),
        "--opendrive-output", str(xodr_path),
        "--output.original-names", "true",
        "--junctions.scurve-stretch", "1.0",
    ]
    if georeference:
        offset_x, offset_y = read_net_offset(net_path)
        if offset_x or offset_y:
            cmd += ["--offset.x", f"{offset_x:.2f}", "--offset.y", f"{offset_y:.2f}"]
            print(f"  :Re-applying UTM offset ({offset_x:.2f}, {offset_y:.2f})")
    print(f"  :netconvert {net_path.name} -> {xodr_path.name}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        raise RuntimeError("netconvert not found on PATH; is SUMO_HOME set?") from exc

    if result.returncode != 0 or not xodr_path.exists():
        raise RuntimeError(f"netconvert failed ({result.returncode}):\n{result.stderr[-2000:]}")

    warnings = [ln for ln in result.stderr.splitlines() if ln.startswith("Warning")]
    if warnings:
        print(f"  :netconvert emitted {len(warnings)} warnings (last: {warnings[-1][:100]})")
    return xodr_path


def main(argv: list[str] | None = None) -> int:
    """Entry point.  Returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--net", default="datasets/chattanooga/updated_net/chatt.net.xml",
                        help="RealTwin updated SUMO network (.net.xml)")
    parser.add_argument("--name", default="chatt", help="Network name")
    parser.add_argument("--outdir", default="vissim/work/chattanooga", help="Output directory")
    parser.add_argument("--progid", default=None,
                        help="Vissim COM ProgID (default: newest installed)")
    parser.add_argument("--radius", type=float, default=DEFAULT_JUNCTION_RADIUS,
                        help="Junction clustering radius in metres")
    parser.add_argument("--junction-source", choices=JUNCTION_SOURCES,
                        default="geometry",
                        help="How links are tied to OpenDRIVE junctions: match "
                             "their geometry against the file's roads (default), "
                             "or parse the road ID out of the Vissim link name")
    parser.add_argument("--skip-netconvert", action="store_true",
                        help="Reuse the existing .xodr instead of regenerating it")
    parser.add_argument("--visible", action="store_true", help="Show the Vissim GUI")
    parser.add_argument("--open-gui", action="store_true",
                        help="Open the saved .inpx in a standalone Vissim GUI when done")
    args = parser.parse_args(argv)

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    xodr_path = outdir / f"{args.name}.xodr"
    inpx_path = outdir / f"{args.name}.inpx"

    if not args.skip_netconvert:
        run_netconvert(Path(args.net).resolve(), xodr_path)
    elif not xodr_path.exists():
        print(f"  :ERROR: --skip-netconvert given but {xodr_path} does not exist.")
        return 1

    print(f"  :Vissim ProgIDs available: {', '.join(available_progids()) or '<none>'}")

    with VissimSession(progid=args.progid, visible=args.visible) as sess:
        print(f"  :Started Vissim via {sess.progid}")
        sess.new_net()

        print(f"  :Importing {xodr_path.name} ...")
        sess.import_opendrive(xodr_path)

        sess.save_net_as(inpx_path)
        print(f"  :Saved Vissim network -> {inpx_path}")

        movements, links, junctions = extract_network(
            sess, radius=args.radius, xodr_path=xodr_path,
            junction_source=args.junction_source)

    # ------------------------------------------------------------------ #
    # Reports
    # ------------------------------------------------------------------ #
    link_rows = [{
        "LinkNo": ln.no,
        "Name": ln.name,
        "IsConnector": int(ln.is_connector),
        "NumLanes": ln.num_lanes,
        "Length2D": round(ln.length, 2),
        "FromLink": ln.from_link,
        "ToLink": ln.to_link,
        "NumPoints": len(ln.points),
    } for ln in sorted(links.values(), key=lambda x: x.no)]
    links_csv = outdir / f"{args.name}_links.csv"
    pd.DataFrame(link_rows).to_csv(links_csv, index=False)

    movements_csv = outdir / f"{args.name}_movements.csv"
    movements.to_csv(movements_csv, index=False)

    # The junction's OpenDRIVE name is never used to group or label -- it is
    # optional and holds whatever the producer chose -- but netconvert puts the
    # SUMO junction id there, and that is the only way to line this table up
    # against a SUMO MatchupTable once junctions carry OpenDRIVE ids.  Record it
    # rather than depending on it.
    junction_names = read_opendrive_junction_names(xodr_path)
    junction_rows = [{
        "JunctionID_OpenDrive": jid,
        "Name_OpenDrive": junction_names.get(jid, ""),
        "InternalLinks": len(members),
        "InMovementTable": int(jid in set(movements["JunctionID_OpenDrive"]))
        if not movements.empty else 0,
    } for jid, members in sorted(junctions.items(),
                                 key=lambda kv: (len(kv[0]), kv[0]))]
    junctions_csv = outdir / f"{args.name}_junctions.csv"
    pd.DataFrame(junction_rows).to_csv(junctions_csv, index=False)

    n_links = sum(not ln.is_connector for ln in links.values())
    n_conns = sum(ln.is_connector for ln in links.values())
    print()
    print(f"  :Vissim network: {n_links} links, {n_conns} connectors")
    print(f"  :Clustered into {len(junctions)} connector groups "
          f"(radius {args.radius:.0f} m)")
    if movements.empty:
        print("  :WARNING: no movements derived -- check the clustering radius and "
              "that link geometry could be read (NumPoints in the links CSV).")
    else:
        print(f"  :Derived {len(movements)} movements across "
              f"{movements['JunctionID_OpenDrive'].nunique()} junctions")
        print(f"  :Turn mix: {movements['Turn'].value_counts().to_dict()}")
    print(f"  :Wrote {links_csv}")
    print(f"  :Wrote {movements_csv}")
    print(f"  :Wrote {junctions_csv}")

    if args.open_gui:
        open_in_gui(inpx_path)
    return 0


def open_in_gui(inpx_path: Path) -> bool:
    """Open a saved network in a standalone Vissim GUI.

    A Vissim instance started over COM shuts down as soon as the Python process
    releases it, so it cannot be left open for inspection.  Launching the
    executable directly gives a normal GUI session with its own lifetime.

    Args:
        inpx_path: Path to the ``.inpx`` to open.

    Returns:
        Whether a Vissim executable was found and launched.
    """
    candidates = sorted(Path(r"C:\Program Files\PTV Vision").glob("PTV Vissim */Exe/Vissim*.exe"),
                        reverse=True)
    exe = next((c for c in candidates if c.stem.lower().startswith("vissim")), None)
    if exe is None:
        print("  :Could not find a Vissim executable to open the GUI; "
              f"open {inpx_path} by hand.")
        return False
    print(f"  :Opening {inpx_path.name} in {exe.parent.parent.name} ...")
    subprocess.Popen([str(exe), str(inpx_path)], close_fds=True)
    return True


if __name__ == "__main__":
    raise SystemExit(main())
