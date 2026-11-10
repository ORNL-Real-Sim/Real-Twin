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
"""Conflict-area right of way by US rules, checked without a Vissim licence.

The cases are the ones Phoenix's import got wrong or right: a southbound right
ahead of the westbound through it merges into, a cross-street left ahead of a
through, diverges left both red, and side streets meeting an arterial.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rt_vissim.conflicts import (A_HAS_RIGHT_OF_WAY, B_HAS_RIGHT_OF_WAY,  # noqa: E402
                                 PASSIVE, UNDETERMINED, Owner, decide,
                                 from_the_right, link_turns, major_roads,
                                 plan_conflicts)

A, B = A_HAS_RIGHT_OF_WAY, B_HAS_RIGHT_OF_WAY


def owner(turn, approach, exit_, junction="7", bearing=None):
    return Owner(junction, turn, approach, exit_, bearing)


class TestDecide:
    def test_turn_rank(self):
        assert decide("thru", "right") == A
        assert decide("right", "thru") == B
        assert decide("left", "thru") == B
        assert decide("Uturn", "left") == B

    def test_left_yields_to_oncoming_right(self):
        """UVC 11-401: a left yields to oncoming traffic, right turners included."""
        assert decide("left", "right") == B

    def test_equal_rank_is_left_to_the_signal(self):
        assert decide("thru", "thru") is None
        assert decide("left", "left") is None

    def test_major_road_first(self):
        """A stop-controlled side street yields even to the arterial's left."""
        assert decide("thru", "left", major_a=False, major_b=True) == B

    def test_uturn_yields_even_to_the_minor_road(self):
        assert decide("Uturn", "right", major_a=True, major_b=False) == B


class TestPlan:
    # SB approach 10 -> exits 20 (W), 21 (S); WB approach 11 -> exit 20.
    OWNERS = {101: owner("right", 10, 20), 102: owner("thru", 10, 21),
              103: owner("thru", 11, 20), 104: owner("left", 12, 20),
              201: owner("thru", 30, 31, junction="8")}

    def test_merge_right_yields_to_thru_whatever_vissim_said(self):
        # Vissim gave the SB right (A) priority over the WB through (B).
        dec, _ = plan_conflicts([(1, 101, 103, A)], self.OWNERS)
        assert dec == {1: B}

    def test_diverge_is_both_red(self):
        dec, _ = plan_conflicts([(1, 101, 102, A), (2, 102, 101, UNDETERMINED)], self.OWNERS)
        assert dec == {1: UNDETERMINED}

    def test_unchanged_status_is_not_rewritten(self):
        dec, _ = plan_conflicts([(1, 103, 101, A)], self.OWNERS)
        assert dec == {}

    def test_left_alone(self):
        pairs = [(1, 101, 103, PASSIVE),        # Vissim made it passive
                 (2, 101, 999, A),               # approach/exit link: no owner
                 (3, 103, 201, B),               # two junctions
                 (4, 103, 103, A)]               # one movement
        dec, notes = plan_conflicts(pairs, self.OWNERS)
        assert dec == {}
        assert any("stay passive" in n for n in notes)

    def test_major_road_at_unsignalised_junction(self):
        # Approach 11 is the arterial: its through beats the side street's left
        # into the same exit, and the arterial's left beats the side right.
        major = {"7": frozenset({11})}
        owners = {**self.OWNERS, 105: owner("left", 11, 22), 106: owner("right", 12, 22)}
        dec, _ = plan_conflicts([(1, 104, 103, A), (2, 106, 105, A)], owners, major=major)
        assert dec == {1: B, 2: B}

    def test_self_conflicting_uturn_goes_passive(self):
        dec, _ = plan_conflicts([(1, 101, 102, A)], self.OWNERS, [{101, 102}])
        assert dec == {1: PASSIVE}


class TestRightHandRule:
    """Two equal-rank movements at a junction with no major road (Phoenix J33)."""

    # Northbound through (heading 0) and westbound through (heading 270, so
    # arriving from the east: the northbound driver's right).
    OWNERS = {301: owner("thru", 1, 2, junction="33", bearing=0.0),
              302: owner("thru", 3, 4, junction="33", bearing=270.0),
              303: owner("thru", 5, 6, junction="33", bearing=180.0)}

    def test_from_the_right(self):
        assert from_the_right(0.0, 270.0) == B
        assert from_the_right(270.0, 0.0) == A
        assert from_the_right(359.9, 271.4) == B      # bearings wrap at north
        assert from_the_right(0.0, 180.0) is None     # head-on: nobody's right
        assert from_the_right(None, 90.0) is None

    def test_vehicle_on_the_right_goes_first(self):
        # Vissim gave the northbound through (A) priority.
        dec, notes = plan_conflicts([(1, 301, 302, A)], self.OWNERS, signalised=())
        assert dec == {1: B}
        assert any("vehicle on the right" in n for n in notes)

    def test_already_right_is_not_rewritten(self):
        dec, _ = plan_conflicts([(1, 302, 301, A)], self.OWNERS, signalised=())
        assert dec == {}

    def test_head_on_tie_keeps_vissims_status(self):
        dec, _ = plan_conflicts([(1, 301, 303, A)], self.OWNERS, signalised=())
        assert dec == {}

    def test_signal_separates_ties(self):
        dec, _ = plan_conflicts([(1, 301, 302, A)], self.OWNERS, signalised={"33.0"})
        assert dec == {}

    def test_off_without_the_signal_list(self):
        dec, _ = plan_conflicts([(1, 301, 302, A)], self.OWNERS)
        assert dec == {}

    def test_major_road_still_comes_first(self):
        # With approach 1 the major road, the northbound through keeps priority.
        dec, _ = plan_conflicts([(1, 301, 302, A)], self.OWNERS,
                                major={"33": frozenset({1, 5})}, signalised=())
        assert dec == {}

    def test_link_turns_reads_the_bearing(self):
        mv = pd.DataFrame({"JunctionID_OpenDrive": [33.0], "Bearing": [271.4], "Turn": ["thru"],
                           "FromLinkNo_Vissim": [3], "ToLinkNo_Vissim": [4],
                           "InternalLinks_Vissim": ["302"]})
        assert link_turns(mv)[302] == Owner("33", "thru", 3, 4, 271.4)


class TestUturnJoiningItsExit:
    """Phoenix J54: the U-turn 37 -> 132 came back ahead of the traffic on exit 132."""

    LINKS = {132: SimpleNamespace(no=132, is_connector=False, junction_key=None,
                                  from_link=None, to_link=None),
             37: SimpleNamespace(no=37, is_connector=False, junction_key=None,
                                 from_link=None, to_link=None),
             9001: SimpleNamespace(no=9001, is_connector=True, junction_key=None,
                                   from_link=37, to_link=132)}
    OWNERS = {501: owner("Uturn", 37, 132, junction="54")}

    def test_uturn_yields_to_its_exit(self):
        pairs = [(201, 501, 132, A), (202, 132, 501, B), (203, 9001, 501, B)]
        dec, notes = plan_conflicts(pairs, self.OWNERS, links=self.LINKS)
        assert dec == {201: B, 202: A, 203: A}
        assert any("U-turn joining its exit" in n for n in notes)

    def test_own_approach_is_a_diverge_and_left_alone(self):
        dec, _ = plan_conflicts([(1, 501, 37, UNDETERMINED)], self.OWNERS, links=self.LINKS)
        assert dec == {}

    def test_passive_stays_passive(self):
        dec, _ = plan_conflicts([(1, 501, 132, PASSIVE)], self.OWNERS, links=self.LINKS)
        assert dec == {}


class TestSlipLaneMerge:
    """Indian School @ 7th: the SB slip lane (1 lane) rejoins WB (2 lanes) at junction J155."""

    LINKS = {1091: SimpleNamespace(no=1091, is_connector=False, junction_key="J155", num_lanes=1,
                                   from_link=None, to_link=None),
             1092: SimpleNamespace(no=1092, is_connector=False, junction_key="J155", num_lanes=2,
                                   from_link=None, to_link=None),
             1168: SimpleNamespace(no=1168, is_connector=False, junction_key="M003", num_lanes=1,
                                   from_link=None, to_link=None),
             1169: SimpleNamespace(no=1169, is_connector=False, junction_key="M003", num_lanes=3,
                                   from_link=None, to_link=None),
             11586: SimpleNamespace(no=11586, is_connector=True, junction_key=None, num_lanes=1,
                                    from_link=1091, to_link=167),
             11587: SimpleNamespace(no=11587, is_connector=True, junction_key=None, num_lanes=2,
                                    from_link=1092, to_link=167),
             11600: SimpleNamespace(no=11600, is_connector=True, junction_key=None, num_lanes=1,
                                    from_link=1168, to_link=1091),
             11601: SimpleNamespace(no=11601, is_connector=True, junction_key=None, num_lanes=3,
                                    from_link=1169, to_link=5)}

    def test_slip_lane_yields_to_the_road_it_joins(self):
        # Vissim gave the slip lane (A) priority on all three conflict areas.
        pairs = [(3081, 1091, 1092, A), (3082, 1091, 11587, A), (4757, 11586, 11587, A)]
        dec, _ = plan_conflicts(pairs, {}, links=self.LINKS)
        assert dec == {3081: B, 3082: B, 4757: B}

    def test_diverge_into_the_slip_lane_is_left_alone(self):
        dec, _ = plan_conflicts([(3274, 1168, 1169, UNDETERMINED)], {}, links=self.LINKS)
        assert dec == {}


class TestMajorRoads:
    def _links(self, lanes):
        return {no: SimpleNamespace(num_lanes=n) for no, n in lanes.items()}

    def test_wider_road_is_major(self):
        # Arterial E-W, 3 lanes each way; driveway from the north, 1 lane.
        mv = pd.DataFrame({"JunctionID_OpenDrive": [5, 5, 5], "Bearing": [90.0, 270.0, 180.0],
                           "FromLinkNo_Vissim": [1, 2, 3]})
        assert major_roads(mv, self._links({1: 3, 2: 3, 3: 1})) == {"5": frozenset({1, 2})}

    def test_tie_has_no_major_road(self):
        mv = pd.DataFrame({"JunctionID_OpenDrive": [5, 5], "Bearing": [0.0, 90.0],
                           "FromLinkNo_Vissim": [1, 2]})
        assert major_roads(mv, self._links({1: 2, 2: 2})) == {}

    def test_signalised_junction_skipped(self):
        mv = pd.DataFrame({"JunctionID_OpenDrive": [5, 5], "Bearing": [0.0, 90.0],
                           "FromLinkNo_Vissim": [1, 2]})
        assert major_roads(mv, self._links({1: 3, 2: 1}), signalised={"5": "16"}) == {}
