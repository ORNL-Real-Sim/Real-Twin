=============================
Autonomous Vehicle Simulation
=============================

This section provides the simulation capabilities for autonomous vehicles (AV, CAV, EV, etc.) within the Real-Twin framework using SUMO, allowing for realistic traffic simulations and road network modeling.


Key Features in AV Simulation
==============================

- **Car-Following Models**: Implement various car-following models to simulate realistic vehicle behavior in traffic.
- **Lane-Changing Models**: Simulate lane-changing behavior of vehicles to reflect real-world driving scenarios.

Data Requirements for AV Simulation
====================================

To effectively simulate autonomous vehicles, the following data is required:

1. **SUMO Network File (.net.xml)**: A SUMO network file that defines the road network, including lanes, junctions, and traffic signals. This file is essential for setting up the simulation environment.
2. **SUMO Demand Files (.flow.xml and .turn.xml)**: SUMO flow and turn files that specify the traffic demand, including vehicle types, routes, and turning movements at intersections. These files are crucial for simulating realistic traffic patterns.
3. **SUMO Configuration File (.sumocfg)**: A SUMO configuration file that ties together the network and demand files, specifying simulation parameters such as time step and duration, output files, etc. This file is necessary to run the simulation.

Prepare AV Simulation Configuration Data
===========================================

To prepare the configuration data for AV simulation, realtwin provides a script that generate configuration file.

.. code-block:: python
    :linenos:
    :emphasize-lines: 1, 4

    import realtwin as rt

    # Generate SUMO configuration file for AV simulation
    rt.prepare_av_configs(dest_dir="")

This script will generate the necessary SUMO configuration files, including the network, demand, and configuration files, in the specified destination directory. The generated files can then be used to run the AV simulation in SUMO.


.. _Update AV Simulation Configuration Data:

:red:`Update AV Simulation Configuration Data`
===============================================

To update the AV simulation configuration data, you can modify the generated SUMO configuration files as needed. This may include car-following and lane-changing parameters.

- :orange:`input`:

  - `input_dir`:

    The directory containing the SUMO network and demand files (.net.xml. .rou.xml (or .flow.xml, .turn.xml), .sumocfg).

  - `net_file`:

    File name of the SUMO network file (.net.xml) that defines the road network. e.g. "network.net.xml".

  - `flow_file`:

    File name of the SUMO flow file (.flow.xml) that specifies the traffic demands. e.g. "network.flow.xml".

  - `turn_file`:

    File name of the SUMO turn file (.turn.xml) that defines the turning ratio between edges.  e.g. "network.turn.xml".

- :orange:`technology`:

  The technology of the vehicles to be simulated. Options include "AV" (Autonomous Vehicle), "CAV" (Connected Autonomous Vehicle), and "EV" (Electric Vehicle). Default is "AV".

- :orange:`veh_types`:

  Different vehicle types to be simulated. Default is ["Human", "AVnormal", "CAV"].

- :orange:`pct_penetration`:

  Percentage of penetration of the specified vehicle types in the traffic flow. Default is [40, 30, 30] for ["Human", "AVnormal", "CAV"] respectively. The sum of the percentages should equal 100.

- :orange:`car_following_model`:

  The car-following model to be used in the simulation. Options include "Krauss", "Wiedemann", "IDM", etc. Default is "Krauss". please refer to the SUMO documentation for a complete list of available car-following models: https://sumo.dlr.de/docs/Definition_of_Vehicles%2C_Vehicle_Types%2C_and_Routes.html.

- :orange:`lane_change_model`:

  The lane-changing model to be used in the simulation. Options include "LC2013", "SL2015", etc. Default is "LC2013". please refer to the SUMO documentation for a complete list of available lane-changing models: https://sumo.dlr.de/docs/Definition_of_Vehicles%2C_Vehicle_Types%2C_and_Routes.html.

- :orange:`sim_name`:

  The name of the simulation. The name same as the file name of the SUMO net file. This name will be used to generate SUMO configuration files.


- :orange:`sim_start`:

  The start time of the simulation in seconds. Default is 0.

- :orange:`sim_time`:

  The total duration of the simulation in seconds. Default is 3600 (1 hour).

- :orange:`Default Parameters for car-following and lane-changing models`:

  Default parameters for car-following and lane-changing models can be specified in the script. These parameters will be applied to the respective vehicle types during the simulation. For details of the parameters, please refer to the SUMO documentation at https://sumo.dlr.de/docs/Definition_of_Vehicles%2C_Vehicle_Types%2C_and_Routes.html.

  Parameters(Unit): Description
    CC0(m): Standstill distance: The desired gap between two vehicles in a stopped condition.

    CC1(s): Time headway/gap. The time headway(gap) a following driver maintains for safety when moving.

    CC2(m): Car-following distance/following variation. The variation in following distance.

    CC3(s): Threshold for entering following. The point at which a driver begins decelerating after perceiving a slower-moving leader and initiates an unconscious following behavior.

    CC4(m/s): Defines negative speed difference during the following process. Low values result in a more sensitive driver reaction to the acceleration or deceleration of the preceding vehicle.

    CC5(m/s): Defines positive speed difference during the following process. Enter a positive value for CC5 which corresponds to the negative value of CC4. Low values result in a more sensitive driver reaction to the acceleration or deceleration of the preceding vehicle.

    CC6(1/(m • s)): Speed dependency of oscillation. The influence of distance on speed oscillation during following.

    CC7(m/s2): Oscillation during acceleration. Actual acceleration during oscillation in unconscious following.

    CC8(m/s2): Standstill acceleration. Desired acceleration when starting from a standstill.

    CC9(m/s2): Desired acceleration at 80 km/h (limited by maximum acceleration defined within the acceleration curves)
