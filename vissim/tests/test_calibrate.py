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
                                 build_problem, counting_sites, epoch_for, geh,
                                 plot_comparison,
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

    def test_per_approach_score_is_sumos_looser_test(self):
        """Right approach total, wrong split: passes per approach, fails per movement."""
        result = score({(1, 2): 700, (1, 3): 300}, {(1, 2): 500, (1, 3): 500})
        assert result["share"] == 0.0
        assert result["approaches"] == 1
        assert result["approach_mean"] == 0.0
        assert result["approach_share"] == 1.0

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

    def test_the_approach_level_is_read_when_logged(self, tmp_path):
        """SUMO's level by default; a log without it falls back to movements."""
        path = tmp_path / "run_ga_history.csv"
        path.write_text("n,seconds,mean_geh,share_below_5,modelled_total,"
                        "approach_mean_geh,approach_share_below_5,x0\n"
                        "1,60,3.0,0.80,9000,7.0,0.60,0.1\n"
                        "2,60,2.5,0.85,9500,8.0,0.65,0.2\n")
        h = read_history(path)
        assert h["level"] == "approach" and h["best"] == [7.0, 7.0]
        assert read_history(path, "movement")["best"] == [3.0, 2.5]
        assert read_history(self.history(tmp_path))["level"] == "movement"

    def test_charts_are_written(self, tmp_path):
        ga = self.history(tmp_path)
        sa = self.history(tmp_path, "run_sa_history.csv")
        assert plot_history(ga, tmp_path / "p.png", "GA", (5.75, 0.675)).stat().st_size
        assert plot_comparison({"ga": ga, "sa": sa}, tmp_path / "c.png", "all",
                               (5.75, 0.675)).stat().st_size


class TestCountingSites:
    """Each movement is counted on every junction link that carries it alone."""

    @staticmethod
    def links():
        from rt_vissim.network import VissimLink
        def conn(no, a, b):
            return VissimLink(no, is_connector=True, from_link=a, to_link=b, length=1.5)
        return {
            1: VissimLink(1, length=100), 10: VissimLink(10, length=100),
            11: VissimLink(11, length=100), 12: VissimLink(12, length=100),
            # a double right turn: two junction links from approach 1 to exit 10
            156: VissimLink(156, length=10.0), 157: VissimLink(157, length=15.6),
            # a U-turn stub too short to carry a point
            170: VissimLink(170, length=0.05),
            901: conn(901, 1, 156), 902: conn(902, 156, 10),
            903: conn(903, 1, 157), 904: conn(904, 157, 10),
            905: conn(905, 1, 170), 906: conn(906, 170, 12),
            907: conn(907, 1, 11),                        # straight across, no junction link
        }

    def test_both_paths_of_a_double_turn_are_counted(self):
        sites, _ = counting_sites(self.links(), [(1, 10)])
        assert sites[(1, 10)] == [156, 157]

    def test_a_stub_too_short_is_counted_on_its_connector(self):
        sites, _ = counting_sites(self.links(), [(1, 12)])
        assert sites[(1, 12)] == [905]

    def test_a_direct_connector_is_its_own_site(self):
        sites, _ = counting_sites(self.links(), [(1, 11)])
        assert sites[(1, 11)] == [907]

    def test_a_movement_with_no_path_is_reported(self):
        sites, notes = counting_sites(self.links(), [(10, 1)])
        assert (10, 1) not in sites
        assert any("10 -> 1" in note for note in notes)

    @pytest.mark.skipif(not (WORK / "chatt_links.csv").exists(),
                        reason="Chattanooga build not present")
    def test_chattanooga_double_right_and_full_coverage(self):
        from rt_vissim.network import read_links_csv
        movements = pd.read_csv(WORK / "chatt_movements.csv")
        keys = list(zip(movements.FromLinkNo_Vissim.astype(int),
                        movements.ToLinkNo_Vissim.astype(int)))
        sites, notes = counting_sites(read_links_csv(WORK / "chatt_links.csv"), keys)
        assert sites[(35, 10)] == [156, 157]
        assert len(sites) == len(keys) and not notes


class TestLaneRoutes:
    """A split searches one share per exit, divided among the exit's lane routes."""

    class _Route:
        def __init__(self, log, key):
            self.log, self.key = log, key

        def SetAttValue(self, name, value):  # noqa: N802 - COM spelling
            self.log[(self.key, name)] = value

    def session(self, log):
        route = self._Route
        class Collection:  # noqa: D106
            def __init__(self, make):
                self.make = make

            def ItemByKey(self, key):  # noqa: N802
                return self.make(key)

        class Decision:  # noqa: D106
            def __init__(self, no):
                self.VehRoutSta = Collection(lambda r: route(log, (no, r)))

        class Net:  # noqa: D106
            VehicleRoutingDecisionsStatic = Collection(Decision)
            VehicleInputs = Collection(lambda n: route(log, ("input", n)))

        class Session:  # noqa: D106
            net = Net()
        return Session()

    def test_the_through_share_is_halved_over_two_lanes(self):
        """Exits 23 and 24; 24 is a two-lane through written as routes 3 and 5."""
        from rt_vissim.calibrate import apply_solution
        knob = SplitKnob(8, 15, "13", (2, 3), (23, 24), zero_routes=(1,), intervals=1,
                         lane_routes={24: ((3, 0.5), (5, 0.5))})
        log = {}
        apply_solution(self.session(log), Problem(splits=[knob]), [0.2])
        flows = {key[1]: value for (key, name), value in log.items() if name == "RelFlow(1)"}
        assert flows == pytest.approx({2: 0.2, 3: 0.4, 5: 0.4, 1: 0.0})

    def test_lane_routes_add_no_variables(self):
        knob = SplitKnob(8, 15, "13", (2, 3), (23, 24), lane_routes={24: ((3, 0.5), (5, 0.5))})
        assert knob.size == 1


class TestJunctionPaths:
    """Every lane path through a junction, as stage 3 routes and stage 6 counts."""

    def test_two_lane_paths_and_their_lanes(self):
        from rt_vissim.network import VissimLink, junction_paths
        links = TestCountingSites.links()
        links[157] = VissimLink(157, length=15.6, num_lanes=2)
        assert junction_paths(links, 1, 10) == [(156, 1), (157, 2)]

    def test_no_path_is_empty(self):
        from rt_vissim.network import junction_paths
        assert junction_paths(TestCountingSites.links(), 10, 1) == []


class TestSavedModelRuns:
    """A calibrated model saved elsewhere takes its signal timing files along."""

    @staticmethod
    def stage6():
        import importlib.util
        path = Path(__file__).resolve().parents[1] / "scripts" / "06_calibrate_turn_inflow.py"
        spec = importlib.util.spec_from_file_location("stage6", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_data_relative_files_are_copied(self, tmp_path):
        src, dst = tmp_path / "work", tmp_path / "out"
        src.mkdir()
        dst.mkdir()
        (src / "net.inpx").write_text('<sc supplyFile1="#data#rbc_4.prbc" supplyFile2=""/>')
        (src / "rbc_4.prbc").write_text("timings")
        copied = self.stage6().copy_supply_files(src / "net.inpx", dst / "net_ga.inpx")
        assert copied == ["rbc_4.prbc"]
        assert (dst / "rbc_4.prbc").read_text() == "timings"

    def test_the_as_built_line_matches_the_charted_level(self, tmp_path):
        """A per-approach chart takes the per-approach start, not the per-movement one."""
        before = {"mean": 3.69, "share": 0.775, "approach_mean": 8.86, "approach_share": 0.591}
        new_log = tmp_path / "new_history.csv"
        new_log.write_text("n,seconds,mean_geh,share_below_5,modelled_total,"
                           "approach_mean_geh,approach_share_below_5\n")
        old_log = tmp_path / "old_history.csv"
        old_log.write_text("n,seconds,mean_geh,share_below_5,modelled_total\n")
        stage6 = self.stage6()
        assert stage6.chart_reference(before, new_log) == (8.86, 0.591)
        assert stage6.chart_reference(before, old_log) == (3.69, 0.775)
        assert stage6.chart_reference(None, new_log) is None

    def test_nothing_is_copied_into_the_same_folder(self, tmp_path):
        (tmp_path / "net.inpx").write_text('"#data#rbc_4.prbc"')
        (tmp_path / "rbc_4.prbc").write_text("timings")
        assert self.stage6().copy_supply_files(tmp_path / "net.inpx",
                                               tmp_path / "net_ga.inpx") == []


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

    def test_lane_routes_are_one_variable_shared_by_lanes(self):
        """Approach 7's through has a 3-lane and a 1-lane path: one share, split 3:1."""
        problem, _ = self.build()
        knob = next(k for k in problem.splits if k.decision_no == 25)
        assert knob.exits == (56, 57) and knob.size == 1
        shares = [share for _route, share in knob.lane_routes[56]]
        assert sorted(shares) == pytest.approx([0.25, 0.75])

    def test_a_junction_marked_n_is_left_out(self):
        problem, notes = self.build(needs={"5", "13", "15"})   # J7 marked N
        assert all(k.junction != "7" for k in problem.inflows + problem.splits)
        assert any("Need calibration" in note for note in notes)
