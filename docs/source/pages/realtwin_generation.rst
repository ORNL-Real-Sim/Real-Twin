=============================
Scenario Generation
=============================

.. code-block:: python
    :linenos:
    :emphasize-lines: 5

    import realtwin as rt

    if __name__ == '__main__':

        # Prepare your configuration file (in YAML format)
        CONFIG_FILE = "./public_configs.yaml"

        # initialize the realtwin object
        twin = rt.RealTwin(input_config_file=CONFIG_FILE, verbose=True)

        # NOTE optional: crate or delete a Python virtual environment for the simulation
        # twin.venv_create(venv_name=twin._venv_name, venv_dir=twin.input_config["output_dir"])
        # twin.venv_delete(venv_name=twin._venv_name, venv_dir=twin.input_config["output_dir"])

        # check simulator env: if SUMO, VISSIM, Aimsun, etc... are installed
        # twin.env_setup(sel_sim=["SUMO", "VISSIM"])
        twin.env_setup(sel_sim=["SUMO", "VISSIM"])

        # NOTE optional: check simulator including additional selection of directories
        # change the new_dir to your own directory where the SUMO is installed (multiple versions)
        # new_dir = [r"C:\Users\xh8\ornl_workspace\github_workspace\Real-Twin\SUMO\sumo-1.20.0\bin"]
        # twin.env_setup(sel_sim=["SUMO", "VISSIM"], sel_dir=new_dir)

        # NOTE optional: strict simulator check, if the version is not matched, install the required version
        # twin.env_setup(sel_sim=["SUMO", "VISSIM"], sel_dir=new_dir, strict_sumo_version="1.21.0")

        # generate MatchupTable for user to update from their customized data
        twin.generate_inputs()

        # generate abstract scenario
        twin.generate_abstract_scenario(incl_elevation_tif=True)

        # generate scenarios
        twin.generate_concrete_scenario()

        # simulate the scenario
        twin.prepare_simulation()

        # perform calibration
        # Available algorithms: GA: Genetic Algorithm, SA: Simulated Annealing, TS: Tabu Search
        twin.calibrate(sel_algo={"turn_inflow": "GA", "behavior": "GA"})

        # post-process the simulation results
        twin.post_process()  # keywords arguments can be passed to specify the post-processing options

        # visualize the simulation results
        twin.visualize()  # keywords arguments can be passed to specify the visualization options




.. _`PyPI`: https://pypi.org/project/osm2gmns
.. _`pip`: https://packaging.python.org/key_projects/#pip
.. _`Shapely`: https://github.com/Toblerity/Shapely
.. _`traci`: https://github.com/osmcode/pyosmium
.. _`Requests`: https://github.com/numpy/numpy
.. _`pyufunc`: https://github.com/xyluo25/pyufunc
.. _`mealpy`: https://mealpy.readthedocs.io/en/latest/
.. _`matplotlib`: https://matplotlib.org/
.. _`networkx`: https://networkx.org/
.. _`PyYAML`: https://pyyaml.org/
.. _`our repository`: https://github.com/ORNL-Real-Sim/Real-Twin
.. _`osmium github homepage`: https://github.com/ORNL-Real-Sim/Real-Twin
.. _`SUMO`: https://sumo.dlr.de/docs/index.html
.. _`Aimsun`: https://www.aimsun.com/
.. _`VISSIM`: https://www.ptvgroup.com/en/solutions/ptv-vissim/
.. _YAML: https://en.wikipedia.org/wiki/YAML
