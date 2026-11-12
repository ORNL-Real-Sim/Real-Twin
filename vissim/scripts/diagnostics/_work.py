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
"""Find a build's files from its work folder, so each diagnostic takes one flag.

Stage 1 writes ``<name>_links.csv`` and ``<name>_movements.csv`` into the work
folder, stage 2 ``MatchupTable.xlsx`` beside them, and the later stages the
``.inpx`` files named after the stage: ``<name>.inpx``, ``<name>_demand.inpx``,
``<name>_demand_signals.inpx``.  The diagnostics ask for the folder and work the
rest out, the same way the stage scripts do with ``dataset.json``.
"""

from __future__ import annotations

import html
import re
import sys
from dataclasses import dataclass
from pathlib import Path

VISSIM_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(VISSIM_DIR))
sys.path.insert(1, str(VISSIM_DIR.parent))


@dataclass
class Build:
    """The files of one build."""

    work: Path
    name: str

    @property
    def links(self) -> Path:
        return self.work / f"{self.name}_links.csv"

    @property
    def movements(self) -> Path:
        return self.work / f"{self.name}_movements.csv"

    @property
    def matchup(self) -> Path:
        return self.work / "MatchupTable.xlsx"

    def inpx(self, stage: str) -> Path:
        """``"demand"`` -> ``<name>_demand.inpx``; ``"signals"`` -> ``<name>_demand_signals.inpx``."""
        suffix = {"import": "", "demand": "_demand", "signals": "_demand_signals"}[stage]
        return self.work / f"{self.name}{suffix}.inpx"


def find_build(work: str | Path) -> Build:
    """The build in ``work``, named after its ``*_links.csv``.

    Raises:
        SystemExit: When the folder holds no link table, or more than one.
    """
    work = Path(work)
    tables = sorted(work.glob("*_links.csv"))
    if len(tables) != 1:
        raise SystemExit(f"  :Expected one *_links.csv in {work}, found {len(tables)}: "
                         f"{[t.name for t in tables]}")
    return Build(work, tables[0].name[: -len("_links.csv")])


def elements(inpx: Path, tag: str) -> list[dict[str, str]]:
    """Every ``<tag ...>`` element's attributes, read from the file as text.

    The ``.inpx`` is XML, but it is also large, and a regular expression over
    the text is quick and needs no COM session.
    """
    text = inpx.read_text(encoding="utf-8")
    return [{k: html.unescape(v) for k, v in re.findall(r'(\w+)="([^"]*)"', m.group(0))}
            for m in re.finditer(rf"<{tag} [^>]*>", text)]
