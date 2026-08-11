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
"""Synchro UTDF to Ring Barrier Controller, checked without a Vissim licence.

The numbers in the integration test are Chattanooga's Synchro INTID 16, the one
intersection whose timings had not been revised since the hand-built controller
under ``VISSIM_previous/`` was made, so it can be compared field for field.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(1, str(Path(__file__).resolve().parents[2]))

from rt_vissim.ir import SignalGroupTiming, SignalPlan  # noqa: E402
from rt_vissim.rbc import build_controller, tenths  # noqa: E402
from rt_vissim.signal import (check_plan, coordinated_phases,  # noqa: E402
                              parse_brp, sequence)

ROOT = Path(__file__).resolve().parents[2]
UTDF = ROOT / "datasets/example1/Control/Synchro_signal.csv"
REFERENCE = (ROOT / "vissim/VISSIM_previous/Model/Shallowford_after_calibration"
             / "rbc_timings_16.prbc")


def group(sg_no, *, barrier=1, ring=1, position=1, max_green=20.0, yellow=4.0,
          all_red=2.0, min_green=5.0, veh_ext=3.0, **kwargs):
    """A signal group with a self-consistent split."""
    return SignalGroupTiming(
        sg_no=sg_no, phase=str(sg_no), barrier=barrier, ring=ring,
        position=position, max_green=max_green, yellow=yellow, all_red=all_red,
        min_green=min_green, veh_ext=veh_ext,
        split=max_green + yellow + all_red, **kwargs)


# ---------------------------------------------------------------------- #
# Reading Synchro
# ---------------------------------------------------------------------- #
class TestParseBRP:
    @pytest.mark.parametrize("code,expected", [
        ("111", (1, 1, 1)),
        ("212", (2, 1, 2)),
        ("122", (1, 2, 2)),
        (222, (2, 2, 2)),
        (111.0, (1, 1, 1)),
    ])
    def test_splits_barrier_ring_position(self, code, expected):
        assert parse_brp(code) == expected

    @pytest.mark.parametrize("code", ["", None, "nan", "12", "1234", "abc"])
    def test_rejects_anything_else(self, code):
        assert parse_brp(code) is None


class TestCoordinatedPhases:
    def test_reference_phase_names_two_phases(self):
        """206 is phases 2 and 6 -- the zero is a separator, not a phase."""
        assert coordinated_phases("206") == {2, 6}

    def test_single_phase(self):
        assert coordinated_phases("2") == {2}

    @pytest.mark.parametrize("value", [None, "", "0"])
    def test_blank(self, value):
        assert coordinated_phases(value) == set()

    def test_ignores_a_decimal_tail(self):
        assert coordinated_phases("206.0") == {2, 6}


# ---------------------------------------------------------------------- #
# The ring / barrier invariant
# ---------------------------------------------------------------------- #
class TestCheckPlan:
    def test_a_sound_plan_is_quiet(self):
        """Two rings of 26 s in one barrier, against a 26 s cycle."""
        plan = SignalPlan(sc_no=1, junction_id="1", synchro_intid="1",
                          cycle_time=26.0, signal_groups=[
                              group(1, ring=1), group(5, ring=2)])
        assert check_plan(plan) == []

    def test_rings_that_disagree_are_reported(self):
        """The rings cross a barrier together, so they must serve equal time."""
        plan = SignalPlan(sc_no=1, junction_id="1", synchro_intid="1",
                          cycle_time=26.0, signal_groups=[
                              group(1, ring=1),
                              group(5, ring=2, max_green=40.0)])
        problems = check_plan(plan)
        assert any("different totals" in p for p in problems)

    def test_barriers_must_add_to_the_cycle(self):
        plan = SignalPlan(sc_no=1, junction_id="1", synchro_intid="1",
                          cycle_time=90.0, signal_groups=[group(1)])
        problems = check_plan(plan)
        assert any("cycle length" in p for p in problems)

    def test_two_barriers_sum(self):
        plan = SignalPlan(sc_no=1, junction_id="1", synchro_intid="1",
                          cycle_time=52.0, signal_groups=[
                              group(1, barrier=1), group(3, barrier=2)])
        assert check_plan(plan) == []


class TestSequence:
    def test_orders_by_barrier_ring_then_position(self):
        plan = SignalPlan(sc_no=1, junction_id="1", synchro_intid="1",
                          signal_groups=[
                              group(1, barrier=1, ring=1, position=1),
                              group(2, barrier=1, ring=1, position=2),
                              group(5, barrier=1, ring=2, position=1),
                              group(3, barrier=2, ring=1, position=2),
                              group(4, barrier=2, ring=1, position=1)])
        # Phase 4 sits at position 1, so it leads phase 3: a lagging left.
        assert sequence(plan) == [[[1, 2], [5]], [[4, 3]]]


# ---------------------------------------------------------------------- #
# Writing the controller
# ---------------------------------------------------------------------- #
class TestTenths:
    @pytest.mark.parametrize("seconds,expected", [
        (100.0, 1000), (4.3, 43), (2.7, 27), (0.0, 0), (0.05, 1)])
    def test_converts(self, seconds, expected):
        assert tenths(seconds) == expected


class TestBuildController:
    def test_yellow_absorbs_the_rounding(self):
        """MaxGreen is rounded to whole seconds; the split must still hold.

        Synchro carries 6.3 s of max green with 3.0 s yellow and 2.7 s all-red,
        a 12.0 s split.  MaxGreen1 rounds to 60 tenths, so yellow takes the
        remaining 33 to keep 60 + 33 + 27 = 120.
        """
        plan = SignalPlan(sc_no=1, junction_id="1", synchro_intid="1",
                          cycle_time=12.0, signal_groups=[
                              group(1, max_green=6.3, yellow=3.0, all_red=2.7)])
        sg = build_controller(plan)["Controller"]["VehicleSignalGroups"][0]
        assert sg["MaxGreen1"] == 60
        assert sg["RedClearance"] == 27
        assert sg["Yellow"] == 33
        assert sg["MaxGreen1"] + sg["Yellow"] + sg["RedClearance"] == 120

    def test_split_survives_every_rounding(self):
        """Whatever the decimals, the three parts must add back to the split."""
        for max_green in (6.3, 21.9, 30.4, 18.3, 7.7, 20.3):
            plan = SignalPlan(sc_no=1, junction_id="1", synchro_intid="1",
                              signal_groups=[group(1, max_green=max_green,
                                                   yellow=3.6, all_red=2.1)])
            doc = build_controller(plan)["Controller"]
            sg = doc["VehicleSignalGroups"][0]
            split = doc["Patterns"][0]["VehicleSignalGroupsInPattern"][0]["Split"]
            assert sg["MaxGreen1"] + sg["Yellow"] + sg["RedClearance"] == split

    def test_a_detector_per_signal_group(self):
        plan = SignalPlan(sc_no=1, junction_id="1", synchro_intid="1",
                          signal_groups=[group(1), group(2, position=2)])
        detectors = build_controller(plan)["Controller"]["VehicleDetectors"]
        assert [d["ID"] for d in detectors] == [1, 2]
        assert detectors[0]["CalledSGs"] == [1]
        assert detectors[0]["ExtendedSGs"] == [1]

    def test_empty_plan_is_refused(self):
        plan = SignalPlan(sc_no=1, junction_id="1", synchro_intid="1")
        with pytest.raises(ValueError, match="no signal groups"):
            build_controller(plan)

    def test_times_are_tenths(self):
        plan = SignalPlan(sc_no=1, junction_id="1", synchro_intid="1",
                          cycle_time=100.0, offset=16.0,
                          signal_groups=[group(1, min_green=5.0, veh_ext=2.0)])
        doc = build_controller(plan)["Controller"]
        assert doc["Patterns"][0]["CycleLength"] == 1000
        assert doc["Patterns"][0]["Offset"] == 160
        assert doc["VehicleSignalGroups"][0]["MinGreen"] == 50
        assert doc["VehicleSignalGroups"][0]["VehExtension"] == 20


# ---------------------------------------------------------------------- #
# Against the hand-built controller
# ---------------------------------------------------------------------- #
@pytest.mark.skipif(not (UTDF.exists() and REFERENCE.exists()),
                    reason="Chattanooga Synchro export or reference .prbc absent")
class TestAgainstReference:
    """INTID 16 against the controller a person built from the same timings."""

    @pytest.fixture(scope="class")
    @classmethod
    def generated(cls):
        from rt_vissim.signal import build_signal_plan, read_synchro
        plan, _ = build_signal_plan(read_synchro(UTDF), "16")
        assert plan is not None
        return plan, build_controller(plan)

    def test_the_plan_is_sound(self, generated):
        plan, _ = generated
        assert check_plan(plan) == []

    def test_timings_match_field_for_field(self, generated):
        from rt_vissim.rbc import compare_controllers
        _, doc = generated
        reference = json.loads(REFERENCE.read_text())
        differences = [d for d in compare_controllers(doc, reference)
                       if not d.startswith("sequence")]
        assert differences == []

    def test_sequence_follows_synchro_not_the_reference(self, generated):
        """The one deliberate departure.

        Synchro's BRP puts phase 4 at position 1 of its ring, ahead of phase 3 --
        a lagging left.  The hand-built controller lists the ring in numeric
        order and loses that.  BRP is Synchro's own statement of order, so it
        wins, and this test records the disagreement rather than hiding it.
        """
        _, doc = generated
        barriers = doc["Controller"]["Sequence"]["BarrierGroups"]
        rings = [r["VehicleSignalGroups"] for b in barriers for r in b["RingGroups"]]
        assert [4, 3] in rings

        reference = json.loads(REFERENCE.read_text())
        ref_rings = [r["VehicleSignalGroups"]
                     for b in reference["Controller"]["Sequence"]["BarrierGroups"]
                     for r in b["RingGroups"]]
        assert [3, 4] in ref_rings
