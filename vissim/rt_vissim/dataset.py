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
"""Which dataset a build came from, so every stage reads that dataset's data.

A RealTwin dataset keeps its counts in ``Traffic/`` and its signal timings in
``Control/``, beside or above the SUMO network.  Stage 1 finds that folder from
the ``.net.xml`` it imports and records it in ``<work>/dataset.json``; the later
stages read their data folders from there.  Defaulting instead to one example's
folders let a stage quietly use another network's signal timings.
"""

from __future__ import annotations

import json
from pathlib import Path

RECORD = "dataset.json"


def find_dataset_dir(net_path: str | Path, levels: int = 3) -> Path | None:
    """The nearest folder at or above the network holding ``Traffic/`` and ``Control/``."""
    here = Path(net_path).resolve().parent
    for _ in range(levels + 1):
        if (here / "Traffic").is_dir() and (here / "Control").is_dir():
            return here
        if here.parent == here:
            break
        here = here.parent
    return None


def write_record(work_dir: str | Path, net_path: str | Path) -> dict | None:
    """Record the dataset beside a build.  Returns it, or ``None`` if none was found."""
    found = find_dataset_dir(net_path)
    if found is None:
        return None
    record = {"net": str(Path(net_path).resolve()), "dataset": str(found),
              "traffic_dir": str(found / "Traffic"), "control_dir": str(found / "Control")}
    (Path(work_dir) / RECORD).write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def data_dir(kind: str, given: str | None, work_dir: str | Path) -> str:
    """``given`` if passed, else the build's recorded ``Traffic`` or ``Control`` folder.

    Args:
        kind: ``"traffic"`` or ``"control"``.
        given: The command-line value, or ``None``.
        work_dir: The build folder holding ``dataset.json``.

    Raises:
        SystemExit: With the flag to pass, when neither is available.
    """
    if given:
        return given
    record_path = Path(work_dir) / RECORD
    if record_path.exists():
        value = json.loads(record_path.read_text(encoding="utf-8")).get(f"{kind}_dir")
        if value:
            return value
    raise SystemExit(f"  :No --{kind}-dir given and no {RECORD} in {work_dir} "
                     f"(stage 1 writes it when the network sits in a dataset with "
                     f"Traffic/ and Control/). Pass --{kind}-dir.")
