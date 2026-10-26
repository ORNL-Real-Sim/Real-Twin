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
"""Ingested against simulated, checked without a Vissim licence."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rt_vissim.verify import MIN_SHARE_SAMPLE, compare_counts, format_report  # noqa: E402

WORK = Path(__file__).resolve().parents[1] / "work"

COUNTED = {(1, 10): 600.0, (1, 11): 300.0, (1, 12): 100.0,      # entry-fed approach 1
           (2, 20): 800.0, (2, 21): 200.0}                      # upstream-fed approach 2
FEEDS = {1: "entry", 2: "upstream"}


def test_entry_fed_and_delivered_scores_clean():
    simulated = {(1, 10): 598.0, (1, 11): 305.0, (1, 12): 97.0, (2, 20): 80.0, (2, 21): 20.0}
    entry, upstream = compare_counts(COUNTED, simulated, FEEDS)
    assert entry.feed == "entry" and entry.approach_geh_share == 1.0
    assert entry.approach_failures == [] and entry.shares_within == 1


def test_a_starved_upstream_approach_fails_volume_but_keeps_its_split():
    """A tenth of the traffic, split 80/20 as counted: volume off, shares right."""
    simulated = {(1, 10): 600.0, (1, 11): 300.0, (1, 12): 100.0, (2, 20): 80.0, (2, 21): 20.0}
    _entry, upstream = compare_counts(COUNTED, simulated, FEEDS)
    assert upstream.approach_geh_share == 0.0
    assert upstream.shares_judged == 1 and upstream.shares_within == 1


def test_shares_are_not_judged_on_too_few_vehicles():
    simulated = {(1, 10): 600.0, (1, 11): 300.0, (1, 12): 100.0,
                 (2, 20): MIN_SHARE_SAMPLE / 2, (2, 21): 0.0}
    _entry, upstream = compare_counts(COUNTED, simulated, FEEDS)
    assert upstream.shares_judged == 0


def test_an_entry_fed_shortfall_is_named():
    simulated = {(1, 10): 300.0, (1, 11): 150.0, (1, 12): 50.0, (2, 20): 800.0, (2, 21): 200.0}
    entry, _ = compare_counts(COUNTED, simulated, FEEDS)
    assert [a for a, *_ in entry.approach_failures] == [1]
    lines = format_report([entry], [(5, 6, 1000.0, 500.0)])
    assert any("approach 1" in line for line in lines)
    assert any("500 never got into the network" in line for line in lines)


@pytest.mark.skipif(not (WORK / "phoenix/phoenix_links.csv").exists(),
                    reason="Phoenix build not present")
def test_phoenix_feeds():
    """7th St southbound (5) is fed by its own entry; 88 lies between junctions."""
    import pandas as pd

    from rt_vissim.network import read_links_csv
    from rt_vissim.verify import approach_feeds
    links = read_links_csv(WORK / "phoenix/phoenix_links.csv")
    mv = pd.read_csv(WORK / "phoenix/phoenix_movements.csv")
    feeds = approach_feeds(links, set(mv.FromLinkNo_Vissim.astype(int)))
    assert feeds[5] == "entry" and feeds[88] == "upstream"
