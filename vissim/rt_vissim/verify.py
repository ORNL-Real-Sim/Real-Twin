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
"""Did the simulation deliver what was ingested?

Checking that every count was *written* says nothing about whether the model
*carries* it.  This compares, after a run, each counted movement and each
counted entry with what the simulation delivered -- volumes and turn shares.

The comparison is split by where an approach's traffic comes from, because the
two halves answer different questions:

* **entry** -- the approach is fed straight from a network entry whose vehicle
  input stage 3 built from this approach's own count.  Here ingestion alone
  fixes the flow, so simulated should equal counted; a gap is a modelling
  fault (capacity, a lane, a route).
* **upstream** -- the approach's traffic is whatever leaves the junctions
  before it, some of them unsurveyed, and the entries no count traces to are
  written at zero.  Its count sets only the split, so the volume can match
  only once the missing demand is estimated (stage 6).  The *shares* still
  have to match.

Volumes are scored by GEH, per movement and per approach as SUMO does; shares
by the largest gap per approach in percentage points, and only where enough
vehicles arrived for a share to mean something.
"""

from __future__ import annotations

from dataclasses import dataclass

from .calibrate import geh

#: Fewest simulated vehicles on an approach for its turn shares to be judged.
#: With 50, Vissim's random route draw alone moves a share by about 6 points.
MIN_SHARE_SAMPLE = 50

#: A share within this many percentage points of the count is agreement.
SHARE_TOLERANCE_PP = 5.0


def approach_feeds(links: dict, approaches) -> dict[int, str]:
    """Label each counted approach ``"entry"`` or ``"upstream"``.

    An approach is entry-fed when the trace stage 3 uses to build vehicle
    inputs (:func:`rt_vissim.demand.trace_to_origin`) ends at a network entry
    that no other counted approach traces to.
    """
    from .demand import find_origin_links, trace_to_origin  # noqa: PLC0415

    origins = find_origin_links(links)
    entry_of = {int(a): trace_to_origin(links, int(a)) for a in approaches}
    feeds = {}
    for a, entry in entry_of.items():
        sharing = [b for b, e in entry_of.items() if e == entry]
        feeds[a] = "entry" if entry in origins and sharing == [a] else "upstream"
    return feeds


@dataclass
class GroupResult:
    """One feed group's agreement."""

    feed: str
    approaches: int
    movements: int
    ingested: float
    simulated: float
    movement_geh_share: float
    movement_geh_mean: float
    approach_geh_share: float
    approach_geh_mean: float
    approach_failures: list[tuple[int, float, float, float]]
    shares_judged: int
    shares_within: int
    share_gap_median: float
    share_outliers: list[tuple[int, float, float]]


def compare_counts(counted: dict[tuple[int, int], float],
                   simulated: dict[tuple[int, int], float],
                   feeds: dict[int, str]) -> list[GroupResult]:
    """Compare counted and simulated volumes and shares, per feed group.

    Args:
        counted: ``{(from, to): vehicles}`` ingested for the period.
        simulated: ``{(from, to): vehicles}`` delivered, from data collection.
        feeds: Output of :func:`approach_feeds`.

    Returns:
        One :class:`GroupResult` per feed group present, ``entry`` first.
    """
    results = []
    for feed in ("entry", "upstream"):
        moves = [k for k in counted if feeds.get(k[0]) == feed]
        if not moves:
            continue
        by_approach: dict[int, list[tuple[int, int]]] = {}
        for key in moves:
            by_approach.setdefault(key[0], []).append(key)
        move_geh = [geh(simulated.get(k, 0.0), counted[k]) for k in moves]
        approach_rows, gaps, outliers = [], [], []
        for a, keys in by_approach.items():
            want = sum(counted[k] for k in keys)
            got = sum(simulated.get(k, 0.0) for k in keys)
            approach_rows.append((a, want, got, geh(got, want)))
            if got >= MIN_SHARE_SAMPLE and want > 0:
                gap = max(abs(simulated.get(k, 0.0) / got - counted[k] / want) * 100
                          for k in keys)
                gaps.append(gap)
                if gap > SHARE_TOLERANCE_PP:
                    outliers.append((a, round(gap, 1), got))
        gaps.sort()
        results.append(GroupResult(
            feed=feed, approaches=len(by_approach), movements=len(moves),
            ingested=sum(counted[k] for k in moves),
            simulated=sum(simulated.get(k, 0.0) for k in moves),
            movement_geh_share=sum(g < 5 for g in move_geh) / len(move_geh),
            movement_geh_mean=sum(move_geh) / len(move_geh),
            approach_geh_share=sum(r[3] < 5 for r in approach_rows) / len(approach_rows),
            approach_geh_mean=sum(r[3] for r in approach_rows) / len(approach_rows),
            approach_failures=sorted((r for r in approach_rows if r[3] >= 5),
                                     key=lambda r: -r[3]),
            shares_judged=len(gaps),
            shares_within=sum(g <= SHARE_TOLERANCE_PP for g in gaps),
            share_gap_median=gaps[len(gaps) // 2] if gaps else float("nan"),
            share_outliers=sorted(outliers, key=lambda o: -o[1]),
        ))
    return results


def format_report(results: list[GroupResult],
                  entries: list[tuple[int, int, float, float]] | None = None) -> list[str]:
    """The lines stage 5 prints.

    Args:
        results: Output of :func:`compare_counts`.
        entries: ``(input no, link, ingested veh/h, entered)`` per counted entry.
    """
    out = []
    if entries:
        ingested = sum(e[2] for e in entries)
        entered = sum(e[3] for e in entries)
        out.append(f"Counted entries: {len(entries)}, ingested {ingested:,.0f} veh/h, "
                   f"entered {entered:,.0f} ({entered / ingested:.1%}).")
        for no, link, want, got in entries:
            if got < 0.95 * want:
                out.append(f"   input {no} (link {link}): ingested {want:,.0f}, entered "
                           f"{got:,.0f} -- {want - got:,.0f} never got into the network")
    for r in results:
        what = ("fed straight from a counted entry -- simulated should equal counted"
                if r.feed == "entry" else
                "fed from upstream junctions -- volume needs the missing demand "
                "(stage 6); shares should still match")
        out.append(f"{r.approaches} approaches {what}:")
        out.append(f"   volume: ingested {r.ingested:,.0f}, simulated {r.simulated:,.0f} "
                   f"({r.simulated / r.ingested:.1%}); GEH<5 on {r.movement_geh_share:.1%} of "
                   f"{r.movements} movements (mean {r.movement_geh_mean:.2f}) and "
                   f"{r.approach_geh_share:.1%} of approaches (mean {r.approach_geh_mean:.2f})")
        out.append(f"   turn shares: {r.shares_within} of {r.shares_judged} approaches with "
                   f"{MIN_SHARE_SAMPLE}+ vehicles within {SHARE_TOLERANCE_PP:.0f} points "
                   f"(median largest gap {r.share_gap_median:.1f})")
        if r.feed == "entry":
            for a, want, got, g in r.approach_failures:
                out.append(f"   approach {a}: counted {want:,.0f}, simulated {got:,.0f}, GEH {g:.1f}")
        for a, gap, n in r.share_outliers[:5]:
            out.append(f"   approach {a}: a share {gap:.1f} points off, on {n:.0f} vehicles")
    return out
