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

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rt_vissim.heads import (DETECTOR_MARGIN, FEET_TO_METRES,  # noqa: E402
                             MIN_DETECTOR_LENGTH, _shares_lane,
                             detector_placement, group_for_code, summarise)
from rt_vissim.ir import Detector, SignalHead  # noqa: E402


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
        assert length >= MIN_DETECTOR_LENGTH

    def test_it_never_grows_beyond_what_synchro_asked_for(self):
        _, length, _ = detector_placement(500.0, self.SETBACK, self.LENGTH)
        assert length <= self.LENGTH + 0.01

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
