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
"""Every stage reads the data of the dataset its network came from."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rt_vissim.dataset import data_dir, find_dataset_dir, write_record  # noqa: E402


def dataset(tmp_path, net_sub=""):
    root = tmp_path / "city"
    (root / "Traffic").mkdir(parents=True)
    (root / "Control").mkdir()
    net_dir = root / net_sub if net_sub else root
    net_dir.mkdir(parents=True, exist_ok=True)
    net = net_dir / "city.net.xml"
    net.write_text("<net/>")
    return root, net


def test_the_network_beside_its_data(tmp_path):
    """Phoenix: Phoenix.net.xml next to Traffic/ and Control/."""
    root, net = dataset(tmp_path)
    assert find_dataset_dir(net) == root.resolve()


def test_the_network_in_a_subfolder(tmp_path):
    """Chattanooga: updated_net/chatt.net.xml, data one level up."""
    root, net = dataset(tmp_path, "updated_net")
    assert find_dataset_dir(net) == root.resolve()


def test_later_stages_read_the_recorded_folders(tmp_path):
    root, net = dataset(tmp_path)
    work = tmp_path / "work"
    work.mkdir()
    write_record(work, net)
    assert Path(data_dir("control", None, work)) == (root / "Control").resolve()
    assert Path(data_dir("traffic", None, work)) == (root / "Traffic").resolve()


def test_a_flag_overrides_the_record(tmp_path):
    _root, net = dataset(tmp_path)
    work = tmp_path / "work"
    work.mkdir()
    write_record(work, net)
    assert data_dir("control", "elsewhere/Control", work) == "elsewhere/Control"


def test_no_record_and_no_flag_stops_instead_of_guessing(tmp_path):
    with pytest.raises(SystemExit, match="--control-dir"):
        data_dir("control", None, tmp_path)


def test_a_network_outside_any_dataset_records_nothing(tmp_path):
    net = tmp_path / "loose.net.xml"
    net.write_text("<net/>")
    assert write_record(tmp_path, net) is None
