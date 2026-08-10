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
"""Turn classification and OpenDRIVE geometry, checked without a Vissim licence.

The turn cases are taken from Chattanooga, where the SUMO network gives an
independent answer for every movement: the angles below are measured off
``datasets/example1/updated_net/chatt.net.xml`` and the expected labels are the
``dir`` attributes ``netconvert`` wrote for the same movements.
"""
from __future__ import annotations

import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rt_vissim.network import (PART_MIN, STRAIGHT_MAX, UTURN_MIN,  # noqa: E402
                               classify_turns, parse_link_name, resample,
                               sample_geometry, signed_delta)


# ---------------------------------------------------------------------- #
# Turn classification
# ---------------------------------------------------------------------- #
class TestClassifyTurns:
    """SUMO's ``NBNode::getDirection`` rule, as ported."""

    def test_skewed_through_movement_at_junction_9(self):
        """-21.3 degrees is through, not left.

        SUMO junction 9's 235.09 degree approach.  A symmetric 20 degree cut
        called this left and so put two lefts on one approach; SUMO calls it
        through because it is inside the 44 degree band and nothing beside it is
        straighter.
        """
        assert classify_turns([56.17, -21.32, -123.45]) == ["right", "thru", "left"]

    def test_skewed_through_movement_at_junction_8(self):
        """-28.4 degrees is through -- the widest straight angle in the network."""
        assert classify_turns([65.66, -28.43, -116.45]) == ["right", "thru", "left"]

    def test_t_junction_keeps_its_turns(self):
        """An approach with no through movement must not invent one.

        The failure mode of a plain "straightest wins" rule: with nothing near
        zero it promotes a genuine right turn to through.  The 44 degree band
        prevents that.
        """
        assert classify_turns([89.34, -79.81]) == ["right", "left"]

    def test_straighter_neighbour_demotes_a_part_turn(self):
        """Inside the band, the straighter movement takes 'thru'."""
        assert classify_turns([2.0, 30.0]) == ["thru", "right"]
        assert classify_turns([2.0, -30.0]) == ["thru", "left"]

    def test_no_demotion_below_the_part_minimum(self):
        """Two nearly straight movements are both through.

        SUMO only considers demoting once the angle exceeds 6 degrees, so a pair
        of near-parallel exits does not have one of them called a turn.
        """
        assert classify_turns([1.0, 5.0]) == ["thru", "thru"]

    def test_uturn_is_near_180_either_way(self):
        assert classify_turns([180.0]) == ["Uturn"]
        assert classify_turns([-179.0]) == ["Uturn"]
        assert classify_turns([171.0]) == ["Uturn"]

    def test_uturn_does_not_absorb_a_sharp_turn(self):
        assert classify_turns([169.0]) == ["right"]

    def test_lone_movement_inside_the_band_is_through(self):
        """With no neighbour there is nothing straighter, so it stays through."""
        assert classify_turns([-21.32]) == ["thru"]

    def test_beyond_the_band_is_always_a_turn(self):
        assert classify_turns([STRAIGHT_MAX + 0.1]) == ["right"]
        assert classify_turns([-(STRAIGHT_MAX + 0.1)]) == ["left"]

    def test_empty_approach(self):
        assert classify_turns([]) == []

    def test_thresholds_match_sumo(self):
        """Guards the constants themselves against a silent edit."""
        assert (STRAIGHT_MAX, PART_MIN, UTURN_MIN) == (44.0, 6.0, 170.0)


class TestSignedDelta:
    def test_wraps_to_a_signed_half_turn(self):
        assert signed_delta(350.0, 10.0) == pytest.approx(20.0)
        assert signed_delta(10.0, 350.0) == pytest.approx(-20.0)

    def test_no_change(self):
        assert signed_delta(90.0, 90.0) == pytest.approx(0.0)

    def test_stays_within_half_a_circle(self):
        """An exact reversal lands on -180, so the range is half-open."""
        for a in range(0, 360, 17):
            for b in range(0, 360, 23):
                assert -180.0 <= signed_delta(float(a), float(b)) < 180.0

    def test_exact_reversal_is_a_uturn_either_way(self):
        assert classify_turns([signed_delta(119.0, 299.0)]) == ["Uturn"]
        assert classify_turns([signed_delta(299.0, 119.0)]) == ["Uturn"]


# ---------------------------------------------------------------------- #
# OpenDRIVE geometry
# ---------------------------------------------------------------------- #
def _geometry(xml: str):
    return ET.fromstring(xml)


class TestSampleGeometry:
    def test_line_runs_along_its_heading(self):
        geo = _geometry('<geometry s="0" x="10" y="20" hdg="0" length="100"/>')
        pts = sample_geometry(geo)
        assert pts[0] == pytest.approx((10.0, 20.0))
        # hdg 0 is +x in OpenDRIVE's convention
        assert pts[-1] == pytest.approx((110.0, 20.0))

    def test_param_poly3_is_not_a_chord(self):
        """A curved paramPoly3 must bow away from its own chord.

        Treating these as straight lines was what broke the first geometry
        prototype: netconvert writes almost every junction road this way.
        """
        geo = _geometry(
            '<geometry s="0" x="0" y="0" hdg="0" length="10">'
            '<paramPoly3 aU="0" bU="10" cU="0" dU="0"'
            '            aV="0" bV="0" cV="5" dV="0" pRange="normalized"/>'
            '</geometry>')
        pts = sample_geometry(geo)
        midpoint = pts[len(pts) // 2]
        chord_y = (pts[0][1] + pts[-1][1]) / 2.0
        assert abs(midpoint[1] - chord_y) > 0.5

    def test_arc_curves_away_from_straight(self):
        geo = _geometry(
            '<geometry s="0" x="0" y="0" hdg="0" length="10">'
            '<arc curvature="0.1"/></geometry>')
        pts = sample_geometry(geo)
        assert pts[-1][1] != pytest.approx(0.0, abs=1e-6)

    def test_zero_curvature_arc_is_a_line(self):
        geo = _geometry(
            '<geometry s="0" x="0" y="0" hdg="0" length="10">'
            '<arc curvature="0"/></geometry>')
        pts = sample_geometry(geo)
        assert pts[-1] == pytest.approx((10.0, 0.0))


class TestResample:
    def test_endpoints_are_preserved(self):
        pts = [(0.0, 0.0), (3.0, 0.0), (10.0, 0.0)]
        out = resample(pts, 5)
        assert len(out) == 5
        assert out[0] == pytest.approx((0.0, 0.0))
        assert out[-1] == pytest.approx((10.0, 0.0))

    def test_spacing_is_even_by_arc_length(self):
        out = resample([(0.0, 0.0), (10.0, 0.0)], 3)
        assert out[1] == pytest.approx((5.0, 0.0))

    def test_degenerate_polyline(self):
        assert resample([(1.0, 2.0)], 3) == [(1.0, 2.0)] * 3
        assert resample([], 3) == []

    def test_zero_length_polyline(self):
        assert resample([(1.0, 1.0), (1.0, 1.0)], 4) == [(1.0, 1.0)] * 4


# ---------------------------------------------------------------------- #
# Link names -- the optional route
# ---------------------------------------------------------------------- #
class TestParseLinkName:
    @pytest.mark.parametrize("name,road,orig", [
        ("390-0-Right", "390", ""),
        ("473: :12_0-0-Right", "473", ":12_0"),
        ("1770-0-Right", "1770", ""),
        ("123: US-27-0-Right", "123", "US-27"),   # a hyphen inside the road name
    ])
    def test_parses(self, name, road, orig):
        assert parse_link_name(name) == (road, orig)

    @pytest.mark.parametrize("name", ["", "hand drawn link", "no-trailing-side-"])
    def test_rejects_what_it_cannot_read(self, name):
        assert parse_link_name(name)[0] is None
