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
"""RealTwin VISSIM pipeline (development sandbox).

Mirrors the RealTwin SUMO pipeline for PTV Vissim, driven over the COM API.
The pipeline is standalone once the OpenDRIVE network has been imported: Vissim
renumbers everything on import, so junctions, approach bearings and turn
movements are re-derived from the Vissim network itself rather than carried over
from SUMO or OpenDRIVE IDs.

Stages, mirroring ``realtwin``:

1. :mod:`rt_vissim.com` -- COM session management and OpenDRIVE import.
2. :mod:`rt_vissim.network` -- junctions, bearings and turns from the Vissim net.
3. :mod:`rt_vissim.matchup` -- generate and read the Vissim MatchupTable.
4. :mod:`rt_vissim.demand` -- GridSmart turn counts to vehicle inputs and routes.
5. :mod:`rt_vissim.signal` -- Synchro UTDF to signal controllers.
6. :mod:`rt_vissim.pipeline` -- the orchestrator tying the stages together.

See ``vissim/README.md`` for status and the development plan.
"""

__version__ = "0.0.1.dev0"

from .com import VissimSession, VissimComError, available_progids  # noqa: F401
from .ir import (  # noqa: F401
    ScenarioIR,
    VehicleInput,
    RoutingDecision,
    SignalPlan,
    SignalGroupTiming,
    SignalHead,
)

__all__ = [
    "VissimSession",
    "VissimComError",
    "available_progids",
    "ScenarioIR",
    "VehicleInput",
    "RoutingDecision",
    "SignalPlan",
    "SignalGroupTiming",
    "SignalHead",
]
