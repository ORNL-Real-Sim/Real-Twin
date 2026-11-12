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
"""Everything the model puts on one link: a first stop when stage 5 names it.

Stage 5 reports stranded vehicles and too-close routing decisions by link
number.  This gathers what the build knows about that link in one place:
length and lanes, its junction, the connectors in and out with their upstream
and downstream links, and every object stage 3 and 4 placed on it -- vehicle
inputs, routing decisions with their position, signal heads, stop signs.

The pattern to recognise: a short entry link (tens of metres) carrying a
vehicle input *and* a routing decision a few metres from its first connector.
Vehicles are created on it already needing a lane they cannot reach, and
Vissim strands or removes them.  That is a geometry limit of the import, not a
demand error; Chattanooga's link 43 (7.5 m) loses about 28 of 49 vehicles an
hour that way.

Usage::

    python vissim/scripts/diagnostics/describe_link.py --work vissim/work/chattanooga 43
    python vissim/scripts/diagnostics/describe_link.py --work vissim/work/phoenix 155 10060
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _work import elements, find_build  # noqa: E402

from rt_vissim.network import read_links_csv  # noqa: E402

PLACED = (("vehicleInput", "link"), ("vehicleRoutingDecisionStatic", "link"),
          ("signalHead", "lane"), ("stopSign", "lane"), ("dataCollectionPoint", "lane"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--work", required=True, help="Build folder, e.g. vissim/work/chattanooga")
    parser.add_argument("--inpx", default=None, help="Network to read objects from (default: the latest stage's)")
    parser.add_argument("links", type=int, nargs="+", help="Link or connector numbers")
    args = parser.parse_args(argv)

    build = find_build(args.work)
    inpx = Path(args.inpx) if args.inpx else next(
        (p for p in (build.inpx("signals"), build.inpx("demand"), build.inpx("import")) if p.exists()), None)
    if inpx is None or not inpx.exists():
        print(f"  :No .inpx found in {build.work}")
        return 1
    links = read_links_csv(build.links)
    placed = {tag: elements(inpx, tag) for tag, _ in PLACED}
    print(f"  :Objects read from {inpx.name}")

    for no in args.links:
        link = links.get(no)
        if link is None:
            print(f"\nlink {no}: not in {build.links.name}")
            continue
        kind = "connector" if link.is_connector else "link"
        print(f"\n{kind} {no}: {link.length:.1f} m, {link.num_lanes} lane(s), "
              f"junction {link.junction_key or '-'}, name '{link.name}'")
        if link.is_connector:
            print(f"   from link {link.from_link} to link {link.to_link}")
        for c in links.values():
            if c.is_connector and c.to_link == no:
                up = links.get(c.from_link)
                print(f"   fed by connector {c.no} from link {c.from_link} "
                      f"({up.length:.1f} m, {up.num_lanes} lanes)" if up else
                      f"   fed by connector {c.no} from link {c.from_link}")
        for c in links.values():
            if c.is_connector and c.from_link == no:
                down = links.get(c.to_link)
                print(f"   leaves by connector {c.no} to link {c.to_link} "
                      f"({down.length:.1f} m, {down.num_lanes} lanes)" if down else
                      f"   leaves by connector {c.no} to link {c.to_link}")
        for tag, attr in PLACED:
            for obj in placed[tag]:
                if obj.get(attr, "").split()[0] == str(no):
                    pos = f" at {obj['pos']} m" if "pos" in obj else ""
                    lane = f" lane {obj['lane'].split()[1]}" if attr == "lane" and " " in obj["lane"] else ""
                    print(f"   {tag}{lane}{pos}: '{obj.get('name', '')}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
