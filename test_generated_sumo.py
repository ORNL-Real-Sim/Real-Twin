import traci
SUMO_BINARY = "sumo-gui"          # or "sumo-gui"
CONFIG = "datasets/Roosevelt/output/SUMO/roosevelt.sumocfg"

traci.start([SUMO_BINARY, "-c", CONFIG])
for _ in range(1000):
    traci.simulationStep()
traci.close()
