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
"""Signal head and detector placement, checked without a Vissim licence.

The awkward cases are all real ones from Chattanooga: a two-lane left bay whose
lanes had to be read rather than assumed, six approaches too short to hold
Synchro's detector layout, and the movements Synchro gives no phase of their own.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rt_vissim.heads import (DETECTOR_MARGIN, FEET_TO_METRES,  # noqa: E402
                             MIN_DETECTOR_LENGTH, _connectors_for, _exit_index,
                             _exits_of, _outgoing, _resolve_lanes, _shares_lane,
                             HEAD_CLEARANCE, check_coverage, stop_line,
                             stop_line_position,
                             detector_placement, group_for_code, summarise)
from rt_vissim.ir import Detector, LaneControl, SignalHead  # noqa: E402
from rt_vissim.network import VissimLink  # noqa: E402


class TestDetectorPlacement:
    """Synchro asks for 50 ft of setback and 50 ft of detector: 30.5 m in all."""

    SETBACK = 50 * FEET_TO_METRES
    LENGTH = 50 * FEET_TO_METRES

    def test_a_long_approach_takes_the_layout_as_asked(self):
        """Link 33 is 395 m, so the detector goes exactly where Synchro says."""
        pos, length, short = detector_placement(395.62, self.SETBACK, self.LENGTH)
        assert short is False
        assert length == pytest.approx(15.24, abs=0.01)
        # downstream edge one setback back from the stop line
        assert pos + length == pytest.approx(395.62 - self.SETBACK, abs=0.01)

    def test_a_short_approach_shrinks_rather_than_vanishing(self):
        """Link 43 is 7.5 m -- far too short, but must still call its phase."""
        pos, length, short = detector_placement(7.51, self.SETBACK, self.LENGTH)
        assert short is True
        assert length >= MIN_DETECTOR_LENGTH
        assert pos >= DETECTOR_MARGIN
        assert pos + length <= 7.51 + 0.01

    @pytest.mark.parametrize("approach", [7.51, 12.98, 18.47, 23.47, 24.96, 26.96,
                                          34.45, 80.66, 395.62])
    def test_the_detector_always_fits_on_the_link(self, approach):
        """Whatever the approach, the detector may not run off the end."""
        pos, length, _ = detector_placement(approach, self.SETBACK, self.LENGTH)
        assert pos >= DETECTOR_MARGIN
        assert pos + length <= approach + 0.01
        # A car length where the link allows it, otherwise whatever is left.
        assert length >= min(MIN_DETECTOR_LENGTH, approach - DETECTOR_MARGIN)

    def test_it_never_grows_beyond_what_synchro_asked_for(self):
        """Synchro's length is kept as asked when it is already long enough."""
        _, length, _ = detector_placement(500.0, self.SETBACK, self.LENGTH)
        assert length <= self.LENGTH + 0.01

    def test_a_detector_shorter_than_a_car_is_raised(self):
        """Synchro asks for 6 ft on two Chattanooga approaches.

        That is an advance detector, meant to sit 224 ft upstream and count
        vehicles passing.  At the stop bar it would drop the call as soon as a
        car crept past it, so it is raised to hold one.
        """
        six_feet = 6 * FEET_TO_METRES
        pos, length, shortened = detector_placement(25.0, self.SETBACK, six_feet)
        assert length == MIN_DETECTOR_LENGTH
        assert not shortened
        assert pos + length <= 25.0

    def test_the_floor_gives_way_on_an_approach_with_no_room(self):
        """The floor is what a car needs, not a promise the link can keep."""
        pos, length, shortened = detector_placement(7.5, self.SETBACK, 15.24)
        assert shortened
        assert pos >= DETECTOR_MARGIN
        assert pos + length <= 7.5 + 0.01

    def test_the_boundary_case_is_not_flagged_short(self):
        """Exactly enough room is not a shortening."""
        approach = self.SETBACK + self.LENGTH + DETECTOR_MARGIN
        pos, length, short = detector_placement(approach, self.SETBACK, self.LENGTH)
        assert short is False
        assert length == pytest.approx(self.LENGTH, abs=0.01)


class TestSharesLane:
    """Synchro's Shared code sits on the through movement: 1 left, 2 right, 3 both."""

    @pytest.mark.parametrize("code,side,expected", [
        ("2", "right", True), ("2", "left", False),
        ("1", "left", True), ("1", "right", False),
        ("3", "left", True), ("3", "right", True),
        ("0", "right", False), (None, "right", False), ("", "left", False),
    ])
    def test_reads_the_code(self, code, side, expected):
        assert _shares_lane({"NBT": code}, "NB", side) is expected

    def test_absent_through_movement(self):
        assert _shares_lane({}, "NB", "right") is False


class TestGroupForCode:
    def test_returns_the_phase_when_the_controller_has_it(self):
        assert group_for_code({"NBL": "3"}, "NBL", {1, 2, 3}) == 3

    def test_rejects_a_phase_the_controller_does_not_serve(self):
        """Synchro can name a phase the controller has no timings for."""
        assert group_for_code({"NBL": "7"}, "NBL", {1, 2, 3}) is None

    @pytest.mark.parametrize("value", [None, "", "nan", "0"])
    def test_blank_or_zero(self, value):
        assert group_for_code({"NBL": value}, "NBL", {1, 2, 3}) is None

    def test_missing_movement(self):
        assert group_for_code({}, "NBL", {1, 2, 3}) is None


class TestSummarise:
    def test_counts_each_kind_of_head(self):
        heads = [
            SignalHead(sc_no=1, sg_no=2, junction_id="2", from_link_no=1,
                       to_link_no=2),                                    # protected
            SignalHead(sc_no=1, sg_no=1, junction_id="2", from_link_no=1,
                       to_link_no=3, scnd_sg_no=6),                      # both
            SignalHead(sc_no=1, sg_no=6, junction_id="2", from_link_no=1,
                       to_link_no=4, permissive_only=True),              # permitted
        ]
        detectors = [
            Detector(sc_no=1, sg_no=2, junction_id="2", link_no=1, lane=1,
                     pos=1.0, length=15.0),
            Detector(sc_no=1, sg_no=2, junction_id="2", link_no=2, lane=1,
                     pos=1.0, length=4.0, shortened=True),
        ]
        text = summarise(heads, detectors)
        assert "3 signal heads" in text
        assert "1 protected" in text
        assert "1 protected-permissive" in text
        assert "1 permitted only" in text
        assert "2 detectors (1 shortened)" in text

    def test_empty(self):
        assert "0 signal heads" in summarise([], [])


class TestParallelConnectors:
    """A turn served by two lanes has two connectors, and both need a head.

    Chattanooga's link 17 is the case that shipped broken: the southbound
    approach at junction 9 has three lanes, and lanes 2 and 3 both run through
    and left.  Resolving a movement to a single connector left lane 2 crossing
    the junction with no signal at all, and no check noticed, because every
    check at the time was written in movements or signal groups -- the same
    units as the bug.
    """

    @staticmethod
    def network():
        """Link 17's shape: 3 lanes, 5 connectors, 2 of them parallel pairs."""
        def link(no, **kw):
            return VissimLink(no=no, **kw)

        def conn(no, frm, to, lanes):
            return VissimLink(no=no, is_connector=True, from_link=frm, to_link=to,
                              num_lanes=1, from_lanes=lanes)

        links = {
            17: link(17, num_lanes=3, length=350.0),   # the approach
            15: link(15), 14: link(14), 12: link(12),  # the exits
            # junction interiors carry a junction key, which is what marks them
            163: link(163, orig_name=":9_0", junction_key="9"),
            164: link(164, orig_name=":9_1", junction_key="9"),
            165: link(165, orig_name=":9_2", junction_key="9"),
            166: link(166, orig_name=":9_3", junction_key="9"),
            167: link(167, orig_name=":9_4", junction_key="9"),
            10031: conn(10031, 17, 163, [1]),          # lane 1 -> right
            10032: conn(10032, 17, 164, [2]),          # lane 2 -> through
            10033: conn(10033, 17, 165, [2]),          # lane 2 -> left
            10034: conn(10034, 17, 166, [3]),          # lane 3 -> through
            10035: conn(10035, 17, 167, [3]),          # lane 3 -> left
            10231: conn(10231, 163, 15, [1]),
            10232: conn(10232, 164, 14, [1]),
            10233: conn(10233, 165, 12, [1]),
            10234: conn(10234, 166, 14, [1]),
            10235: conn(10235, 167, 12, [1]),
        }
        return links

    @staticmethod
    def movement(from_link, to_link):
        return SimpleNamespace(FromLinkNo_Vissim=from_link, ToLinkNo_Vissim=to_link)

    def test_a_connector_reaches_the_exit_beyond_the_junction(self):
        """The exit is two connectors away, not one."""
        links = self.network()
        outgoing = _outgoing(links)
        assert _exits_of(links[10032], links, outgoing) == {14}
        assert _exits_of(links[10033], links, outgoing) == {12}

    def test_both_lanes_of_a_turn_are_found(self):
        """The through is served from lanes 2 and 3, so both connectors count."""
        links = self.network()
        outgoing = _outgoing(links)
        exits = _exit_index(links, outgoing)

        from_link, found = _connectors_for(self.movement(17, 14), outgoing, exits)
        assert from_link == 17
        assert [c.no for c in found] == [10032, 10034]

        _, found = _connectors_for(self.movement(17, 12), outgoing, exits)
        assert [c.no for c in found] == [10033, 10035]

    def test_a_single_lane_turn_still_finds_one(self):
        """The right turn has one lane, so nothing is invented for it."""
        links = self.network()
        outgoing = _outgoing(links)
        exits = _exit_index(links, outgoing)
        _, found = _connectors_for(self.movement(17, 15), outgoing, exits)
        assert [c.no for c in found] == [10031]

    def test_they_come_back_in_lane_order(self):
        """Ordering by lane keeps head numbering stable across runs."""
        links = self.network()
        outgoing = _outgoing(links)
        exits = _exit_index(links, outgoing)
        _, found = _connectors_for(self.movement(17, 14), outgoing, exits)
        assert [min(c.from_lanes) for c in found] == [2, 3]


class TestCheckCoverage:
    """The check that would have caught it: enumerate the network, not the plan.

    Link 17's southbound approach has three lanes; lane 1 turns right and lanes
    2 and 3 each run through and left.  One head per lane is what the manual
    prefers (p. 634) and what a driver sees, so the check counts lanes.
    """

    @staticmethod
    def head(lane, from_link=17):
        return SignalHead(sc_no=8, sg_no=4, junction_id="9", from_link_no=from_link,
                          link_no=from_link, lane=lane, pos=349.9,
                          movement="SBT/SBL", turn="thru/left")

    def test_a_lane_with_no_head_is_reported(self):
        """Exactly the shipped defect: lane 2 crossed the junction unsignalised."""
        links = TestParallelConnectors.network()
        problems, summary = check_coverage(links, [self.head(1), self.head(3)], [])
        assert "2/3 signalised lanes" in summary
        assert any("lane 2: no signal head" in p for p in problems)

    def test_two_heads_on_one_lane_is_reported(self):
        """A lane can show only one indication, so doubling up is also a fault."""
        links = TestParallelConnectors.network()
        heads = [self.head(1), self.head(2), self.head(2), self.head(3)]
        problems, _ = check_coverage(links, heads, [])
        assert any("lane 2: 2 signal heads" in p for p in problems)

    def test_full_coverage_is_silent(self):
        """One head on each of the three lanes and there is nothing to report."""
        links = TestParallelConnectors.network()
        heads = [self.head(1), self.head(2), self.head(3)]
        problems, summary = check_coverage(links, heads, [])
        assert problems == []
        assert "3/3 signalised lanes" in summary

    def test_a_lane_that_feeds_no_connector_is_not_demanded(self):
        """Only lanes that enter the junction need a signal."""
        links = TestParallelConnectors.network()
        links[17].num_lanes = 4          # a fourth lane that goes nowhere
        problems, summary = check_coverage(
            links, [self.head(1), self.head(2), self.head(3)], [])
        assert problems == []
        assert "3/3 signalised lanes" in summary


class TestHeadIsUpstreamOfTheStopLine:
    """A head past the diverge is never reached, so it is checked geometrically.

    Chattanooga's connectors leave their approach about 0.2 m before the link
    ends.  Placing all 69 heads at the end of the link put every one of them
    downstream of that, and traffic drove through red across the whole corridor.
    """

    @staticmethod
    def network(head_pos):
        links = TestParallelConnectors.network()
        for no in (10031, 10032, 10033, 10034, 10035):
            links[no].from_pos = 352.46          # where the connectors leave
        links[17].length = 352.7
        head = SignalHead(sc_no=8, sg_no=4, junction_id="9", from_link_no=17,
                          link_no=17, lane=2, pos=head_pos, movement="SBT/SBL")
        return links, [head]

    def test_a_head_past_the_diverge_is_reported(self):
        links, heads = self.network(352.60)     # the end of the link
        problems, summary = check_coverage(links, heads, [])
        assert any("drive through the red" in p for p in problems)
        assert "0/1 of them upstream" in summary

    def test_a_head_at_the_diverge_is_still_too_late(self):
        links, heads = self.network(352.46)
        problems, _ = check_coverage(links, heads, [])
        assert any("drive through the red" in p for p in problems)

    def test_a_head_upstream_of_the_diverge_passes(self):
        links, heads = self.network(351.46)     # one metre upstream
        problems, summary = check_coverage(links, heads, [])
        assert not any("drive through the red" in p for p in problems)
        assert "1/1 of them upstream" in summary


class TestStopLineIsWhereConnectorsLeave:
    """The stop line is the diverge, not the end of the link."""

    @staticmethod
    def control(connectors=(10032, 10033)):
        return LaneControl(sc_no=8, sg_no=4, junction_id="9", link_no=17,
                           lane=2, connectors=connectors)

    def test_it_reads_the_connector_departure(self):
        links = TestParallelConnectors.network()
        links[10032].from_pos = 352.46
        links[10033].from_pos = 352.50
        assert stop_line(self.control(), links, 352.7) == 352.46

    def test_the_head_sits_a_clearance_upstream(self):
        links = TestParallelConnectors.network()
        links[10032].from_pos = 352.46
        links[10033].from_pos = 352.50
        assert stop_line_position(self.control(), links, 352.7, []) == round(
            352.46 - HEAD_CLEARANCE, 2)

    def test_an_unknown_departure_falls_back_and_says_so(self):
        """The fallback is the assumption that caused the bug, so it is loud."""
        links = TestParallelConnectors.network()
        guessed = []
        assert stop_line(self.control(), links, 352.7, guessed) == 352.7
        assert guessed == ["link 17 lane 2"]


class TestResolveLanes:
    """A lane shows one signal, so the movements on it have to agree."""

    @staticmethod
    def entry(code, sg, scnd=None, perm_sg=None, connector=1):
        plan = SimpleNamespace(sc_no=8, junction_id="9", synchro_intid="8")
        return dict(code=code, sg=sg, scnd=scnd, perm_sg=perm_sg, plan=plan,
                    connector=connector, turn="", permissive_only=False)

    def test_movements_agreeing_give_one_control(self):
        controls, warnings = _resolve_lanes(
            {(17, 2): [self.entry("SBT", 4), self.entry("SBL", 4)]}, [])
        assert len(controls) == 1
        assert controls[0].sg_no == 4 and not controls[0].mixed
        assert controls[0].movements == ("SBT", "SBL")
        assert warnings == []

    def test_the_through_governs_a_disagreeing_lane(self):
        """A turn cannot hold the through traffic beside it in the same lane."""
        controls, _ = _resolve_lanes(
            {(17, 2): [self.entry("SBL", 1, perm_sg=4), self.entry("SBT", 4)]}, [])
        assert controls[0].sg_no == 4
        assert controls[0].mixed

    def test_a_permitted_turn_sharing_a_lane_is_noted_not_warned(self):
        """Ordinary permitted left from a shared lane: the conflict area holds it."""
        _, warnings = _resolve_lanes(
            {(17, 2): [self.entry("SBL", 1, perm_sg=4), self.entry("SBT", 4)]}, [])
        assert any("cannot be shown" in w for w in warnings)
        assert not any("Check the Synchro Lanes record" in w for w in warnings)

    def test_a_protected_only_turn_sharing_a_lane_is_a_data_error(self):
        """No signal can hold the through and release the left from one lane."""
        _, warnings = _resolve_lanes(
            {(17, 2): [self.entry("SBL", 1, perm_sg=None), self.entry("SBT", 4)]}, [])
        assert any("Check the Synchro Lanes record" in w for w in warnings)
