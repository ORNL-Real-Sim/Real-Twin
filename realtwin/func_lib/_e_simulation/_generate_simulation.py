##############################################################################
# Copyright (c) 2024, Oak Ridge National Laboratory                          #
# All rights reserved.                                                       #
#                                                                            #
# This file is part of RealTwin and is distributed under a MIT               #
# license. For the licensing terms see the LICENSE file in the top-level     #
# directory.                                                                 #
#                                                                            #
# Contributors: ORNL Real-Twin Team                                          #
# Contact: realtwin@ornl.gov                                                 #
##############################################################################

"""The module to prepare the simulation from the concrete scenario."""

# import four elements of AbstractScenario
from ._sumo import SUMOPrep
from ._aimsun import AimsunPrep


class SimPrep:
    '''Prepare simulation from concrete scenario'''

    def __init__(self, **kwargs):
        self.SUMOSim = SUMOPrep(**kwargs)
        self.AimsunSim = AimsunPrep(**kwargs)

    def create_sumo_sim(self,
                        ConcreteScn,
                        start_time: float = 3600 * 8,
                        end_time: float = 3600 * 10,
                        seed: list | int = 812,
                        step_length: float = 0.1):
        """Prepare SUMO documents for simulation.
        """

        # check seed type
        if isinstance(seed, int):
            seed = [seed]
        elif isinstance(seed, list):
            pass
        else:
            raise ValueError("  :seed must be an integer or a list of integers.")
        if len(seed) > 1:
            print("  :Multiple seeds are provided, the first one will be used.")
            seed = seed[0]

        self.SUMOSim.importNetwork(ConcreteScn)
        self.SUMOSim.importDemand(ConcreteScn,
                                  start_time,
                                  end_time,
                                  seed)
        self.SUMOSim.generateConfig(ConcreteScn,
                                    start_time,
                                    end_time,
                                    seed,
                                    step_length)
        self.SUMOSim.importSignal(ConcreteScn)
        # print("  :SUMO simulation is prepared.")

    def create_aimsun_sim(self,
                          ConcreteScn,
                          start_time: float = 3600 * 8,
                          end_time: float = 3600 * 10,
                          seed: list | int = 812,
                          step_length: float = 0.1):
        """Prepare Aimsun documents for simulation."""
        self.AimsunSim.importDemand(ConcreteScn, start_time, end_time)

    def create_vissim_sim(self,
                          ConcreteScn,
                          start_time: float = 3600 * 8,
                          end_time: float = 3600 * 10,
                          seed: list | int = 812,
                          step_length: float = 0.1):
        """Prepare VISSIM documents for simulation."""
        pass

#     def createSimulation(self, ConcreteScn, start_time, end_time, seed, step_length):
#
#         # SUMO
#         # NetworkName = ConcreteScn.Supply.NetworkName
#         self.Sumo.importNetwork(ConcreteScn)
#         # self.Sumo.importSignal(ConcreteScn)
#         self.Sumo.importDemand(ConcreteScn, start_time, end_time, seed)
#         self.Sumo.generateConfig(ConcreteScn, start_time, end_time, seed, step_length)
