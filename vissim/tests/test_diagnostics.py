"""The diagnostics find a build's files and read objects from its network."""

import importlib
import sys
from pathlib import Path

import pytest

DIAGNOSTICS = Path(__file__).resolve().parent.parent / "scripts" / "diagnostics"
sys.path.insert(0, str(DIAGNOSTICS))
work = importlib.import_module("_work")


class TestFindBuild:
    def test_names_the_build_after_its_link_table(self, tmp_path):
        (tmp_path / "phoenix_links.csv").write_text("No\n", encoding="utf-8")
        build = work.find_build(tmp_path)
        assert build.name == "phoenix"
        assert build.links == tmp_path / "phoenix_links.csv"
        assert build.movements == tmp_path / "phoenix_movements.csv"
        assert build.matchup == tmp_path / "MatchupTable.xlsx"

    def test_inpx_by_stage(self, tmp_path):
        (tmp_path / "chatt_links.csv").write_text("No\n", encoding="utf-8")
        build = work.find_build(tmp_path)
        assert build.inpx("import").name == "chatt.inpx"
        assert build.inpx("demand").name == "chatt_demand.inpx"
        assert build.inpx("signals").name == "chatt_demand_signals.inpx"

    def test_refuses_an_ambiguous_folder(self, tmp_path):
        for name in ("a_links.csv", "b_links.csv"):
            (tmp_path / name).write_text("No\n", encoding="utf-8")
        with pytest.raises(SystemExit):
            work.find_build(tmp_path)

    def test_refuses_an_empty_folder(self, tmp_path):
        with pytest.raises(SystemExit):
            work.find_build(tmp_path)


class TestElements:
    def test_reads_attributes_and_unescapes_them(self, tmp_path):
        inpx = tmp_path / "x.inpx"
        inpx.write_text(
            '<network>\n'
            '<stopSign anmFlag="false" lane="10000 1" name="RTOR WBR (J2)" no="1" pos="0" sg="16 2"/>\n'
            '<vehicleInput link="2" name="0417) Shallowford Rd &amp; Gunbarrel Rd (link 2)" no="1">\n'
            '</network>\n', encoding="utf-8")
        signs = work.elements(inpx, "stopSign")
        assert signs == [{"anmFlag": "false", "lane": "10000 1", "name": "RTOR WBR (J2)",
                          "no": "1", "pos": "0", "sg": "16 2"}]
        inputs = work.elements(inpx, "vehicleInput")
        assert inputs[0]["name"] == "0417) Shallowford Rd & Gunbarrel Rd (link 2)"
        assert work.elements(inpx, "signalHead") == []
