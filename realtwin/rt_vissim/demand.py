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
"""Turn GridSmart turn counts into Vissim vehicle inputs and routing decisions.

The Vissim counterpart of RealTwin's SUMO demand path
(:mod:`realtwin.func_lib._f_calibration.algo_sumo.util_cali_turn_inflow`), which
writes ``.flow.xml`` / ``.turn.xml`` and hands them to ``jtrrouter``.  Vissim has
no router: demand is expressed directly as

* **vehicle inputs** -- hourly volumes on the network's origin links, and
* **static routing decisions** -- per approach, the relative flow to each exit.

which is the same information ``jtrrouter`` consumes, minus the route expansion.

The method matches SUMO's:

1. Read each intersection's GridSmart file into 15-minute counts per movement.
2. Sum a junction's movements per approach to get the approach volume.
3. Walk upstream from each approach to the network entry link, and put a vehicle
   input there -- an intersection in the middle of the network is fed by traffic
   that entered upstream, so injecting at the approach itself would double count.
4. Turn the per-approach movement split into routing decisions.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .ir import RoutingDecision, VehicleInput

#: GridSmart reports counts in 15-minute bins.
DEMAND_INTERVAL = 15 * 60

#: Movement codes GridSmart reports, in RealTwin's canonical order.
TURN_CODES = ["NBR", "NBT", "NBL", "NBU", "EBR", "EBT", "EBL", "EBU",
              "SBR", "SBT", "SBL", "SBU", "WBR", "WBT", "WBL", "WBU"]

#: Long-form direction names GridSmart uses in its column headers.
DIRECTION_ALIASES = {"Northbound": "NB", "Southbound": "SB",
                     "Westbound": "WB", "Eastbound": "EB"}

_TIME_RE = re.compile(r"^\d{1,2}:\d{2}$")


def time_to_seconds(time_str: str) -> int:
    """Convert a ``"HH:MM"`` clock time to seconds after midnight."""
    hour, minute = (int(x) for x in str(time_str).split(":"))
    return hour * 3600 + minute * 60


# ---------------------------------------------------------------------- #
# Reading GridSmart files
# ---------------------------------------------------------------------- #
def read_gridsmart_counts(path: str | Path) -> pd.DataFrame:
    """Read one GridSmart turn-count workbook into tidy 15-minute counts.

    GridSmart exports carry a variable number of preamble rows and a two-level
    header (direction over movement).  The data is located by finding the first
    row whose first cell looks like a clock time, exactly as RealTwin's
    ``generate_turn_demand_cali`` does.

    Args:
        path: Path to the ``.xls`` / ``.xlsx`` export.

    Returns:
        Columns ``Time`` plus one per code in :data:`TURN_CODES`, with missing
        movements filled with 0.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If no time column could be located.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"GridSmart file not found: {path}")

    probe = pd.read_excel(path, header=None)
    time_rows = probe[probe[0].astype(str).str.match(_TIME_RE, na=False)]
    if time_rows.empty:
        raise ValueError(f"No 'HH:MM' time column found in {path.name}")
    start_row = time_rows.index.min() - 2
    if start_row < 0:
        raise ValueError(f"Unexpected header layout in {path.name}")

    df = pd.read_excel(path, header=[start_row, start_row + 1])

    # Flatten the two-level header, forward-filling the merged direction cells.
    df.columns = df.columns.to_frame().ffill().agg("".join, axis=1)
    df.columns = [str(c).replace(" ", "") for c in df.columns]
    df = df.rename(columns={df.columns[0]: "Time"})
    df = df.dropna(axis=1, how="all")
    df = df[df["Time"] != "Total"]
    df = df.loc[:, ~df.columns.str.contains("Unassigned", na=False)]

    for long, short in DIRECTION_ALIASES.items():
        df.columns = [c.replace(long, short) for c in df.columns]

    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    df = df.reindex(columns=["Time"] + TURN_CODES, fill_value=0)
    df = df[df["Time"].astype(str).str.match(_TIME_RE, na=False)]
    return df.reset_index(drop=True)


def build_turn_counts(matchup, traffic_dir: str | Path) -> pd.DataFrame:
    """Join GridSmart counts onto the MatchupTable's Vissim links.

    Args:
        matchup: A :class:`rt_vissim.matchup.MatchupTable`.
        traffic_dir: Directory holding the GridSmart files named in the table.

    Returns:
        One row per (interval, movement) that maps to a network turn.  Columns:
        ``IntersectionName``, ``JunctionID_OpenDrive``, ``Turn``, ``IntervalStart``,
        ``IntervalEnd``, ``Count``, ``FromLinkNo_Vissim``, ``ToLinkNo_Vissim``.
    """
    traffic_dir = Path(traffic_dir)
    lookup = matchup.turn_lookup()
    files = matchup.gridsmart_files()

    frames = []
    for junction_id, filename in files.items():
        name = filename if "." in Path(filename).name else f"{filename}.xls"
        path = traffic_dir / name
        if not path.exists():
            print(f"  :WARNING: GridSmart file missing for junction {junction_id}: {path.name}")
            continue

        junction_rows = matchup.df[matchup.df["JunctionID_OpenDrive"] == junction_id]
        names = [v for v in junction_rows["IntersectionName_GridSmart"].dropna().unique()]
        intersection = str(names[0]) if names else f"Junction {junction_id}"

        try:
            counts = read_gridsmart_counts(path)
        except (ValueError, FileNotFoundError) as exc:
            print(f"  :WARNING: could not read {path.name}: {exc}")
            continue

        long = counts.melt(id_vars=["Time"], var_name="Turn", value_name="Count")
        long["IntersectionName"] = intersection
        long["JunctionID_OpenDrive"] = junction_id
        long["IntervalStart"] = long["Time"].apply(time_to_seconds)
        long["IntervalEnd"] = long["IntervalStart"] + DEMAND_INTERVAL
        frames.append(long.drop(columns=["Time"]))

    if not frames:
        return pd.DataFrame(columns=["IntersectionName", "JunctionID_OpenDrive", "Turn",
                                     "IntervalStart", "IntervalEnd", "Count",
                                     "FromLinkNo_Vissim", "ToLinkNo_Vissim"])

    counts = pd.concat(frames, ignore_index=True)
    merged = counts.merge(lookup, on=["IntersectionName", "Turn", "JunctionID_OpenDrive"],
                          how="inner")
    merged["Count"] = pd.to_numeric(merged["Count"], errors="coerce").fillna(0).astype(int)
    return merged.reset_index(drop=True)


# ---------------------------------------------------------------------- #
# Origin links
# ---------------------------------------------------------------------- #
def find_origin_links(links: dict) -> set[int]:
    """Return the links with no upstream connector -- the network's entry points.

    A link nothing feeds into is an entry, with one exception: a link that also
    feeds nothing is a disconnected stub, not an entry, and demand injected there
    would never reach the network.  Chattanooga's OpenDRIVE import leaves two of
    them, and SUMO's flow file omits the same two of the network's 18 sources.

    Args:
        links: Output of :func:`rt_vissim.network.read_links`.

    Returns:
        Link numbers that nothing feeds into and that lead somewhere.
    """
    fed, feeds = set(), set()
    for ln in links.values():
        if not ln.is_connector:
            continue
        if ln.to_link is not None:
            fed.add(ln.to_link)
        if ln.from_link is not None:
            feeds.add(ln.from_link)
    return {no for no, ln in links.items()
            if not ln.is_connector and no not in fed and no in feeds}


def road_predecessors(links: dict) -> dict[int, set[int]]:
    """Map each road link to the road links feeding it, hopping over junctions.

    Vissim turns every OpenDRIVE road into a link, so one junction becomes
    several junction-internal links.  Walking raw connectors therefore sees an
    approach as having as many predecessors as the upstream junction has internal
    paths, which looks like a merge even when every one of them comes from the
    same road.  SUMO collapses its internal edges, which is why the same walk
    succeeds there.

    Stepping over the internal links restores that view: Chattanooga's 186
    non-connector links are 59 road links -- matching SUMO's 59 edges exactly --
    and 127 junction-internal ones.

    Args:
        links: Output of :func:`rt_vissim.network.read_links`.

    Returns:
        ``{road link: {road links feeding it}}``.
    """
    direct: dict[int, list[int]] = {}
    for ln in links.values():
        if ln.is_connector and ln.from_link is not None and ln.to_link is not None:
            direct.setdefault(ln.to_link, []).append(ln.from_link)

    def is_internal(no: int) -> bool:
        link = links.get(no)
        return link is not None and not link.is_connector and bool(link.junction_key)

    roads: dict[int, set[int]] = {}
    for no, link in links.items():
        if link.is_connector or is_internal(no):
            continue
        found: set[int] = set()
        queue, seen = list(direct.get(no, [])), set()
        while queue:
            candidate = queue.pop()
            if candidate in seen:
                continue
            seen.add(candidate)
            if is_internal(candidate):
                queue.extend(direct.get(candidate, []))
            else:
                found.add(candidate)
        roads[no] = found
    return roads


def trace_to_origin(links: dict, approach: int, max_hops: int = 50) -> int | None:
    """Walk upstream from an approach link to the network entry that feeds it.

    Follows road predecessors while each step is unambiguous.  A link fed by
    several *different roads* is a genuine merge, so the walk stops -- the demand
    for that approach is already accounted for by an upstream input.

    Args:
        links: Output of :func:`rt_vissim.network.read_links`.
        approach: Link number to start from.
        max_hops: Cycle guard.

    Returns:
        The origin link number, or ``None`` when the walk stopped at a merge.
    """
    predecessors = road_predecessors(links)

    current = approach
    seen = {current}
    for _ in range(max_hops):
        preds = predecessors.get(current, set())
        if not preds:
            return current  # nothing upstream: this is an entry link
        if len(preds) > 1:
            return None  # traffic mixes here; an upstream input covers it
        nxt = next(iter(preds))
        if nxt in seen:
            return None  # loop
        seen.add(nxt)
        current = nxt
    return None


# ---------------------------------------------------------------------- #
# Building the IR
# ---------------------------------------------------------------------- #
def build_routing_decisions(turn_counts: pd.DataFrame) -> list[RoutingDecision]:
    """Build one static routing decision per (approach, interval).

    Relative flows are the raw movement counts.  Vissim normalises them, so
    there is no need to convert to ratios first -- and keeping the counts makes
    the model easier to check against the GridSmart source.

    Args:
        turn_counts: Output of :func:`build_turn_counts`.

    Returns:
        The routing decisions, skipping approaches with no traffic in an
        interval (Vissim rejects a decision whose flows are all zero).
    """
    if turn_counts.empty:
        return []

    decisions: list[RoutingDecision] = []
    grouped = turn_counts.groupby(["JunctionID_OpenDrive", "FromLinkNo_Vissim",
                                   "IntervalStart", "IntervalEnd"], sort=True)
    for (junction_id, from_link, start, end), group in grouped:
        routes = (group.groupby("ToLinkNo_Vissim")["Count"].sum()
                  .astype(float).to_dict())
        if sum(routes.values()) <= 0:
            continue
        bound = str(group["Turn"].iloc[0])[:2]
        decisions.append(RoutingDecision(
            junction_id=int(junction_id) if str(junction_id).isdigit() else junction_id,
            from_link_no=int(from_link),
            interval_start=float(start),
            interval_end=float(end),
            routes={int(k): float(v) for k, v in routes.items()},
            name=f"{group['IntersectionName'].iloc[0]} {bound}",
        ))
    return decisions


def build_vehicle_inputs(turn_counts: pd.DataFrame, links: dict) -> tuple[list[VehicleInput], list[str]]:
    """Build vehicle inputs on the network's origin links.

    Each approach's movements are summed to an approach volume, then traced
    upstream to the entry link that feeds it.  Approaches whose traffic mixes
    before reaching an entry are dropped: an upstream input already covers them.

    Args:
        turn_counts: Output of :func:`build_turn_counts`.
        links: Output of :func:`rt_vissim.network.read_links`.

    Returns:
        ``(vehicle_inputs, warnings)``.
    """
    warnings: list[str] = []
    if turn_counts.empty:
        return [], warnings

    origins = find_origin_links(links)
    approach_totals = (turn_counts
                       .groupby(["IntersectionName", "FromLinkNo_Vissim",
                                 "IntervalStart", "IntervalEnd"], as_index=False)["Count"]
                       .sum())

    # Trace each distinct approach once; the walk is the expensive part.
    trace_cache: dict[int, int | None] = {}
    for approach in approach_totals["FromLinkNo_Vissim"].unique():
        trace_cache[int(approach)] = trace_to_origin(links, int(approach))

    approach_totals["OriginLink"] = approach_totals["FromLinkNo_Vissim"].map(
        lambda a: trace_cache.get(int(a)))

    unresolved = approach_totals[approach_totals["OriginLink"].isna()]
    if not unresolved.empty:
        n = unresolved["FromLinkNo_Vissim"].nunique()
        warnings.append(f"{n} approaches do not trace to a network entry "
                        "(traffic mixes upstream); their demand is assumed to be "
                        "covered by an upstream vehicle input.")

    resolved = approach_totals.dropna(subset=["OriginLink"]).copy()
    resolved["OriginLink"] = resolved["OriginLink"].astype(int)

    off_network = resolved[~resolved["OriginLink"].isin(origins)]
    if not off_network.empty:
        n = off_network["OriginLink"].nunique()
        warnings.append(f"{n} traced origins are not network entry links; skipped.")
    resolved = resolved[resolved["OriginLink"].isin(origins)]

    # Two approaches tracing to one entry would double count that entry.
    collisions = (resolved.groupby(["OriginLink", "IntervalStart"])["FromLinkNo_Vissim"]
                  .nunique())
    for (origin, _start), n_app in collisions.items():
        if n_app > 1:
            warnings.append(f"Origin link {origin} is fed by {n_app} approaches; "
                            "counts summed, which may over-state its demand.")
            break

    aggregated = (resolved.groupby(["OriginLink", "IntervalStart", "IntervalEnd"],
                                   as_index=False)
                  .agg(Count=("Count", "sum"),
                       IntersectionName=("IntersectionName", "first")))

    inputs: list[VehicleInput] = []
    for row in aggregated.itertuples(index=False):
        duration = float(row.IntervalEnd) - float(row.IntervalStart)
        if duration <= 0:
            continue
        inputs.append(VehicleInput(
            link_no=int(row.OriginLink),
            interval_start=float(row.IntervalStart),
            interval_end=float(row.IntervalEnd),
            volume=round(int(row.Count) * 3600.0 / duration, 2),
            count=int(row.Count),
            name=str(row.IntersectionName),
        ))

    # Every entry gets an input, at zero where no counted approach traced back to
    # it.  Omitting it instead would leave the entry silently unmodelled and
    # impossible to calibrate later; the SUMO flow file does the same, carrying
    # 16 of the network's 18 source edges.
    intervals = sorted({(float(s), float(e)) for s, e
                        in zip(approach_totals["IntervalStart"],
                               approach_totals["IntervalEnd"])})
    covered = {vi.link_no for vi in inputs}
    uncovered = sorted(set(origins) - covered)
    for link_no in uncovered:
        for start, end in intervals:
            inputs.append(VehicleInput(
                link_no=int(link_no), interval_start=start, interval_end=end,
                volume=0.0, count=0, name="no counts"))
    if uncovered:
        warnings.append(f"{len(uncovered)} of {len(origins)} entry links have no "
                        f"counted approach tracing to them: {uncovered}. "
                        "Written at volume 0 for calibration.")
    return inputs, warnings


def build_demand(matchup, links: dict, traffic_dir: str | Path, *,
                 sim_start_time: float | None = None,
                 sim_end_time: float | None = None) -> tuple[list[VehicleInput],
                                                             list[RoutingDecision],
                                                             list[str]]:
    """Build the complete demand IR from GridSmart counts.

    Args:
        matchup: A :class:`rt_vissim.matchup.MatchupTable`.
        links: Output of :func:`rt_vissim.network.read_links`.
        traffic_dir: Directory holding the GridSmart files.
        sim_start_time: Optional lower bound on demand intervals, in seconds.
        sim_end_time: Optional upper bound on demand intervals, in seconds.

    Returns:
        ``(vehicle_inputs, routing_decisions, warnings)``.
    """
    turn_counts = build_turn_counts(matchup, traffic_dir)
    warnings: list[str] = []

    if turn_counts.empty:
        warnings.append("No GridSmart counts matched the MatchupTable; "
                        "check File_GridSmart, IntersectionName_GridSmart and "
                        "Turn_GridSmart.")
        return [], [], warnings

    if sim_start_time is not None:
        turn_counts = turn_counts[turn_counts["IntervalStart"] >= sim_start_time]
    if sim_end_time is not None:
        turn_counts = turn_counts[turn_counts["IntervalEnd"] <= sim_end_time]
    if turn_counts.empty:
        warnings.append(f"No demand intervals fall inside "
                        f"[{sim_start_time}, {sim_end_time}].")
        return [], [], warnings

    inputs, input_warnings = build_vehicle_inputs(turn_counts, links)
    decisions = build_routing_decisions(turn_counts)
    return inputs, decisions, warnings + input_warnings
