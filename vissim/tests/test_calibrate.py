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
"""Turn and inflow calibration, checked without a Vissim licence.

The parts worth pinning down are the ones an optimiser would quietly exploit if
they were wrong: shares that do not sum to one, U-turns that are free to take
traffic, and knobs found at junctions that were measured.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rt_vissim.calibrate import (InflowKnob, Problem, SplitKnob,  # noqa: E402
                                 build_problem, epoch_for, geh, plot_comparison,
                                 plot_history, read_history, score,
                                 simulations, stick_shares)

WORK = Path(__file__).resolve().parents[1] / "work" / "chattanooga"


class TestGeh:
    def test_agreement_scores_zero(self):
        assert geh(100, 100) == 0.0

    def test_both_zero_scores_zero(self):
        assert geh(0, 0) == 0.0

    def test_a_starved_movement_scores_high(self):
        """J4's westbound through: 1,032 counted, nothing modelled."""
        assert geh(0, 1032) == pytest.approx(45.43, abs=0.01)

    def test_is_symmetric(self):
        assert geh(80, 120) == pytest.approx(geh(120, 80))


class TestStickShares:
    """``n - 1`` free values become ``n`` shares that always sum to one."""

    @pytest.mark.parametrize("values", [[0.0], [1.0], [0.3], [0.2, 0.7], [0.9, 0.1, 0.5]])
    def test_shares_sum_to_one(self, values):
        assert sum(stick_shares(values)) == pytest.approx(1.0)

    def test_two_exits_is_x_and_one_minus_x(self):
        assert stick_shares([0.3]) == pytest.approx([0.3, 0.7])

    def test_three_exits_follow_the_earlier_notebooks(self):
        """second = (1 - first) * x, as in the ORNL turn/inflow notebooks."""
        assert stick_shares([0.2, 0.5]) == pytest.approx([0.2, 0.4, 0.4])

    def test_out_of_range_values_are_clipped(self):
        shares = stick_shares([1.7])
        assert shares == pytest.approx([1.0, 0.0])
        assert all(s >= 0 for s in stick_shares([-0.5, 2.0]))


class TestProblem:
    @staticmethod
    def problem():
        return Problem(
            inflows=[InflowKnob(7, 42, "5"), InflowKnob(11, 46, "15")],
            splits=[SplitKnob(8, 15, "13", (2, 3), (23, 24), zero_routes=(1,)),
                    SplitKnob(99, 7, "7", (1, 2, 3), (54, 56, 57))],
            max_inflow=2000.0)

    def test_a_two_way_split_is_one_variable(self):
        assert SplitKnob(8, 15, "13", (2, 3), (23, 24)).size == 1

    def test_a_three_way_split_is_two(self):
        assert SplitKnob(99, 7, "7", (1, 2, 3), (54, 56, 57)).size == 2

    def test_u_turns_are_not_variables(self):
        """Route 1 is the U-turn: pinned at zero, not searched over."""
        knob = SplitKnob(8, 15, "13", (2, 3), (23, 24), zero_routes=(1,))
        assert knob.size == 1

    def test_size_counts_inflows_then_split_variables(self):
        assert self.problem().size == 2 + 1 + 2

    def test_every_variable_is_searched_on_zero_to_one(self):
        """SA and TS step sizes are absolute; a common scale keeps them fair."""
        lower, upper = self.problem().bounds()
        assert lower == [0.0] * 5
        assert upper == [1.0] * 5

    def test_inflows_are_written_in_vehicles_per_hour(self):
        """Inflows come first and scale by max_inflow; splits stay as they are."""
        assert self.problem().physical([0.25, 1.0, 0.25, 0.5, 0.5]) == \
            pytest.approx([500.0, 2000.0, 0.25, 0.5, 0.5])

    def test_decode_reads_back_as_shares_per_exit(self):
        decoded = dict(self.problem().decode([0.25, 0, 0.25, 0.5, 0.5]))
        split = next(v for k, v in decoded.items() if "decision 8" in k)
        assert split == {23: 0.25, 24: 0.75}
        inflow = next(v for k, v in decoded.items() if "input 7" in k)
        assert inflow == 500.0


class TestAlgorithms:
    """The three algorithms cost very different numbers of simulations."""

    def test_costs_at_the_realtwin_defaults(self):
        assert simulations("ga", {"epoch": 10, "pop_size": 10}) == 110
        assert simulations("sa", {"epoch": 10}) == 12
        assert simulations("ts", {"epoch": 10, "neighbour_size": 10}) == 102

    def test_the_proof_run_cost(self):
        """Epoch 1, population 10: the 20 evaluations the proof run logged."""
        assert simulations("ga", {"epoch": 1, "pop_size": 10}) == 20

    @pytest.mark.parametrize("algo,settings", [
        ("ga", {"pop_size": 10}), ("sa", {}), ("ts", {"neighbour_size": 10})])
    def test_a_budget_is_spent_without_overrunning(self, algo, settings):
        epoch = epoch_for(algo, {**settings, "epoch": 0}, 110)
        assert simulations(algo, {**settings, "epoch": epoch}) <= 110
        assert simulations(algo, {**settings, "epoch": epoch + 1}) > 110

    def test_an_unknown_algorithm_is_refused(self):
        with pytest.raises(ValueError):
            simulations("pso", {"epoch": 1})


class TestScore:
    def test_only_counted_movements_are_scored(self):
        result = score({(1, 2): 100, (3, 4): 50, (9, 9): 999},
                       {(1, 2): 100, (3, 4): 50})
        assert result["matched"] == 2
        assert result["share"] == 1.0
        assert result["mean"] == 0.0

    def test_share_below_five(self):
        result = score({(1, 2): 100, (3, 4): 0}, {(1, 2): 100, (3, 4): 400})
        assert result["share"] == 0.5


class TestProgressCharts:
    """The history log reads back, and both charts are drawn from it."""

    @staticmethod
    def history(tmp_path, name="run_ga_history.csv"):
        path = tmp_path / name
        path.write_text("n,seconds,mean_geh,share_below_5,modelled_total,x0\n"
                        "1,90,5.9,0.66,6500,0.1\n"
                        "2,90,5.2,0.74,7000,0.2\n"
                        "3,90,5.6,0.78,6800,0.3\n")
        return path

    def test_best_so_far_keeps_the_share_of_the_best_candidate(self, tmp_path):
        """Eval 3 has the higher share but a worse mean; the search keeps eval 2."""
        h = read_history(self.history(tmp_path))
        assert h["best"] == [5.9, 5.2, 5.2]
        assert h["best_share"] == pytest.approx([66.0, 74.0, 74.0])

    def test_charts_are_written(self, tmp_path):
        ga = self.history(tmp_path)
        sa = self.history(tmp_path, "run_sa_history.csv")
        assert plot_history(ga, tmp_path / "p.png", "GA", (5.75, 0.675)).stat().st_size
        assert plot_comparison({"ga": ga, "sa": sa}, tmp_path / "c.png", "all",
                               (5.75, 0.675)).stat().st_size


@pytest.mark.skipif(not (WORK / "chatt_demand_signals.inpx").exists(),
                    reason="Chattanooga build not present")
class TestBuildProblemOnChattanooga:
    """The knobs are read off the network, and match SUMO's 4 + 8."""

    @staticmethod
    def build(needs=None):
        from rt_vissim.network import read_links_csv
        movements = pd.read_csv(WORK / "chatt_movements.csv")
        # The six junctions with a GridSmart camera; every approach to one of
        # them has counts.
        surveyed = {2, 4, 9, 10, 11, 14}
        counted = set(movements.loc[movements.JunctionID_OpenDrive.isin(surveyed),
                                    "FromLinkNo_Vissim"].astype(int))
        return build_problem(WORK / "chatt_demand_signals.inpx",
                             read_links_csv(WORK / "chatt_links.csv"),
                             movements, counted, needs_calibration=needs)

    def test_four_inflows_one_per_unsurveyed_junction(self):
        problem, _ = self.build()
        assert sorted(k.link_no for k in problem.inflows) == [42, 46, 58, 59]
        assert sorted(k.junction for k in problem.inflows) == ["13", "15", "5", "7"]

    def test_eight_splits_on_the_decisions_sumo_calibrates(self):
        problem, _ = self.build()
        assert sorted(k.decision_no for k in problem.splits) == [8, 9, 14, 23, 24, 25, 26, 27]

    def test_decisions_moved_upstream_are_traced_to_their_approach(self):
        """Decision 25 sits on link 6 but governs approach 7."""
        problem, _ = self.build()
        knob = next(k for k in problem.splits if k.decision_no == 25)
        assert knob.approach == 7

    def test_u_turn_exits_are_never_free(self):
        problem, _ = self.build()
        knob = next(k for k in problem.splits if k.decision_no == 8)
        assert 22 not in knob.exits            # 15 -> 22 is the U-turn

    def test_twelve_variables_as_in_sumo(self):
        problem, _ = self.build()
        assert problem.size == 12

    def test_a_junction_marked_n_is_left_out(self):
        problem, notes = self.build(needs={"5", "13", "15"})   # J7 marked N
        assert all(k.junction != "7" for k in problem.inflows + problem.splits)
        assert any("Need calibration" in note for note in notes)
