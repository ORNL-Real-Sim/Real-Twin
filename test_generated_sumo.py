import traci
SUMO_BINARY = "sumo-gui"          # or "sumo-gui"
CONFIG = r"datasets/MLK\output\SUMO/MLK_final_elevation_20260306.sumocfg"

traci.start([SUMO_BINARY, "-c", CONFIG])
for _ in range(1000):
    traci.simulationStep()
traci.close()
