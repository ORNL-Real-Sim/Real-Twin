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
"""Reference-free checks on a filled MatchupTable.

Chattanooga could be checked against RealTwin's own SUMO MatchupTable, which is
how three real defects were found.  No other corridor comes with one, so the
pipeline cannot depend on having a reference to be trusted.

Every check here works from internal consistency alone -- the network against
itself, and the counts against the network -- so it runs anywhere.  They are
deliberately advisory: each reports what looks wrong and why, and the
MatchupTable stays the place a person corrects it.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

#: Two legs closer than this are hard to order reliably, so the rank rule that
#: names directions may pair them the wrong way round.
MIN_LEG_SEPARATION = 30.0

#: Flow across a link between two counted junctions rarely balances exactly --
#: driveways, parking and camera error all contribute.  Beyond this it is worth
#: a look.
FLOW_TOLERANCE = 0.15


@dataclass
class Finding:
    """One thing worth a person's attention.

    Attributes:
        check: Which check produced it.
        severity: ``"error"`` for a provable contradiction, ``"warning"``
            for something merely suspicious.
        where: Junction, approach or link the finding concerns.
        message: What is wrong, in one line.
    """

    check: str
    severity: str
    where: str
    message: str

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"  :{self.severity.upper()}: [{self.check}] {self.where}: {self.message}"


def check_leg_separation(movements: pd.DataFrame,
                         minimum: float = MIN_LEG_SEPARATION) -> list[Finding]:
    """Flag junctions whose legs are too close together to order confidently.

    Directions are named by ranking the legs clockwise, so two legs only a few
    degrees apart can swap places on a small change in geometry and take each
    other's name.  Nothing downstream would notice.

    Args:
        movements: The stage 1 movement table.
        minimum: Smallest comfortable separation, in degrees.

    Returns:
        One finding per too-close pair.
    """
    findings: list[Finding] = []
    for junction_id, group in movements.groupby("JunctionID_Vissim", sort=True):
        legs = (group.drop_duplicates("FromLinkNo_Vissim")
                .sort_values("Bearing")[["FromLinkNo_Vissim", "Bearing"]])
        rows = list(legs.itertuples(index=False))
        for first, second in zip(rows, rows[1:]):
            gap = abs(second.Bearing - first.Bearing)
            gap = min(gap, 360.0 - gap)
            if gap < minimum:
                findings.append(Finding(
                    "leg-separation", "warning", f"junction {junction_id}",
                    f"approaches {first.FromLinkNo_Vissim} and "
                    f"{second.FromLinkNo_Vissim} are only {gap:.1f} deg apart "
                    f"({first.Bearing:.1f}, {second.Bearing:.1f}); the direction "
                    "each is given depends on that ordering being right"))
    return findings


def check_turn_order(matchup: pd.DataFrame) -> list[Finding]:
    """Verify each approach's movement codes ascend through R, T, L, U.

    Movements are sorted by turning angle, so a code that steps backwards means
    two movements were classified inconsistently.

    Args:
        matchup: A filled MatchupTable frame.

    Returns:
        One finding per approach that breaks the ordering.
    """
    order = {"R": 0, "T": 1, "L": 2, "U": 3}
    findings: list[Finding] = []
    for (junction_id, approach), group in matchup.groupby(
            ["JunctionID_Vissim", "FromLinkNo_Vissim"], sort=True):
        codes = [str(c) for c in group["Turn_GridSmart"].dropna()]
        ranks = [order.get(c[-1]) for c in codes if c[-1] in order]
        if any(b <= a for a, b in zip(ranks, ranks[1:])):
            findings.append(Finding(
                "turn-order", "error", f"junction {junction_id} approach {approach}",
                f"codes {codes} do not ascend R, T, L, U; two movements are "
                "labelled inconsistently and a count may be joined twice"))
    return findings


def check_flow_continuity(turn_counts: pd.DataFrame,
                          tolerance: float = FLOW_TOLERANCE) -> list[Finding]:
    """Compare traffic leaving one counted junction with traffic arriving at the next.

    Where two counted junctions share a link, the count leaving the first onto
    that link and the count arriving at the second from it describe the same
    traffic.  A large gap means a movement is mis-assigned, a count file is
    mismatched to its junction, or the two cameras disagree.

    Mid-block driveways make exact agreement unusual, so only a sizeable
    imbalance is reported.

    Args:
        turn_counts: Output of :func:`rt_vissim.demand.build_turn_counts`.
        tolerance: Relative difference tolerated before reporting.

    Returns:
        One finding per imbalanced pair.
    """
    if turn_counts.empty:
        return []

    outflow = turn_counts.groupby(
        ["JunctionID_Vissim", "ToLinkNo_Vissim"])["Count"].sum()
    inflow = turn_counts.groupby(
        ["JunctionID_Vissim", "FromLinkNo_Vissim"])["Count"].sum()
    approach_of = {int(link): junction for junction, link in inflow.index}

    findings: list[Finding] = []
    for (upstream, link), leaving in outflow.items():
        downstream = approach_of.get(int(link))
        if downstream is None or downstream == upstream or not leaving:
            continue
        arriving = float(inflow[(downstream, link)])
        diff = (arriving - leaving) / leaving
        if abs(diff) > tolerance:
            findings.append(Finding(
                "flow-continuity", "warning",
                f"junctions {upstream} -> {downstream} via link {link}",
                f"{leaving:.0f} vehicles counted leaving but {arriving:.0f} "
                f"arriving ({diff:+.0%}); check the count files are matched to "
                "the right junctions"))
    return findings


def validate(movements: pd.DataFrame, matchup: pd.DataFrame,
             turn_counts: pd.DataFrame) -> list[Finding]:
    """Run every reference-free check and return the findings, worst first.

    Args:
        movements: The stage 1 movement table.
        matchup: A filled MatchupTable frame.
        turn_counts: Output of :func:`rt_vissim.demand.build_turn_counts`.

    Returns:
        Findings, errors before warnings.
    """
    findings = (check_turn_order(matchup)
                + check_flow_continuity(turn_counts)
                + check_leg_separation(movements))
    return sorted(findings, key=lambda f: (f.severity != "error", f.check, f.where))
