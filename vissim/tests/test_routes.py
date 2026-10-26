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
"""Where routing decisions sit, checked on the built networks without Vissim."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rt_vissim.network import read_links_csv  # noqa: E402
from rt_vissim.routes import plan_decision_positions  # noqa: E402

WORK = Path(__file__).resolve().parents[1] / "work"


def placements(city: str, stem: str, with_demand: bool):
    links = read_links_csv(WORK / city / f"{stem}_links.csv")
    mv = pd.read_csv(WORK / city / f"{stem}_movements.csv")
    approaches = sorted({(str(r.JunctionID_OpenDrive), int(r.FromLinkNo_Vissim))
                         for r in mv.itertuples()})
    demand = set(mv.FromLinkNo_Vissim.astype(int)) if with_demand else None
    return plan_decision_positions(approaches, links, set(mv.ToLinkNo_Vissim.astype(int)),
                                   3.33, demand)


@pytest.mark.skipif(not (WORK / "phoenix/phoenix_links.csv").exists(),
                    reason="Phoenix build not present")
def test_a_decision_passes_a_fork_onto_an_entry_only_its_approach_feeds():
    """Phoenix link 6: its input is the 7th St approach's count; it forks at M003."""
    before, _ = placements("phoenix", "phoenix", with_demand=False)
    after, notes = placements("phoenix", "phoenix", with_demand=True)
    assert before[("10", 5)][0] == 5
    assert after[("10", 5)][0] == 6
    assert {k for k in before if before[k] != after[k]} == {("10", 5)}
    assert any("onto entry 6" in n for n in notes)


@pytest.mark.skipif(not (WORK / "chattanooga/chatt_links.csv").exists(),
                    reason="Chattanooga build not present")
def test_chattanooga_has_no_such_entry_and_is_unchanged():
    before, _ = placements("chattanooga", "chatt", with_demand=False)
    after, _ = placements("chattanooga", "chatt", with_demand=True)
    assert before == after
