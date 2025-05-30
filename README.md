- [Real-Twin](#read-twin)
  - [Summary](#summary)
  - [Installation](#installation)
  - [Documentation](#documentation)
  - [Quick Example](#quick-example)
  - [Call for Contributions](#call-for-contributions)
  - [Citation](#citation)

# Real-Twin
# 🔁 Real-Twin: A Unified Scenario Generation Tool for Mobility Simulation

**Real-Twin** is a unified, **model-agnostic scenario generation tool** designed to streamline and standardize the evaluation of emerging mobility technologies. It provides an end-to-end framework that includes robust workflows, integrated tools, and comprehensive metrics to generate, calibrate, and benchmark microscopic traffic simulation scenarios across multiple platforms.

## ✨ Key Features

- **Unified Scenario Generation**  
  Generate transferable, simulation-ready scenarios from heterogeneous data sources using a consistent workflow.

- **Automated Calibration Workflow**  
  Bridges simulation and real-world data, minimizing manual effort and making traffic simulation more accessible to researchers and engineers.

- **Model-Agnostic Compatibility**  
  Supports **SUMO**, **VISSIM**, and **AIMSUN** for cross-platform scenario generation and benchmarking. Enables reliable comparisons and reproducibility across different simulation tools.

- **Consistent Scenarios across Different Simulators**  
  Generate comparable simulation scenarios across different microscopic traffic simulators, providing users the ability to conduct benchmarking and cross-validation that are crucial for ensuring the reliability and reproducibility of simulation results.

- **Emerging Technology Support**  
  Includes a scenario database and pipeline for studying **autonomous vehicles (AVs)**, with planned extensions to **CAVs**, **EVs**, and other advanced technologies.

![Overview of RealTwin](image/README/RT_Overview.png)

## Installation

```python
pip install realtwin

# Or throught wheel:  download the latest wheel from github:
pip install *.whl
```


## Documentation

User guide and API documentation can be found at:

## Quick Example

```python

import realtwin as rt

# Please refer to the official documentation for more details on RealTwin preparation before running the simulation

if __name__ == '__main__':

    # Step 1: Prepare your configuration file (in YAML format)
    CONFIG_FILE = "./realtwin_config.yaml"
    updated_sumo_net = r"./datasets/input_dir_dummy/updated_net/chatt.net.xml"

    # Step 2: initialize the realtwin object
    twin = rt.RealTwin(input_config_file=CONFIG_FILE, verbose=True)

    # Step 3: check simulator env: if SUMO, VISSIM, Aimsun, etc... are installed
    twin.env_setup(sel_sim=["SUMO", "VISSIM"])

    # Step 4: Create Matchup Table from SUMO network
    twin.generate_inputs(incl_sumo_net=updated_sumo_net)

    # BEFORE step 5, there are three steps to be performed:
    # 1. Prepare Traffic Demand and save it to Traffic Folder in input directory
    # 2. Prepare Control Data (Signal) and save it to Control Folder in input directory
    # 3. Manually fill in the Matchup Table in the input directory

    # Step 5: generate abstract scenario
    twin.generate_abstract_scenario()

    # AFTER step 5, Double-check the Matchup Table in the input directory to ensure it is correct.

    # Step 6: generate scenarios
    twin.generate_concrete_scenario()

    # Step 7: simulate the scenario
    twin.prepare_simulation()

    # Step 8: perform calibration, Available algorithms: GA: Genetic Algorithm, SA: Simulated Annealing, TS: Tabu Search
    twin.calibrate(sel_algo={"turn_inflow": "GA", "behavior": "GA"})

    # Step 9 (ongoing): post-process the simulation results
    twin.post_process()  # keyword arguments can be passed to specify the post-processing options

    # Step 10 (ongoing): visualize the simulation results
    twin.visualize()  # keyword arguments can be passed to specify the visualization options
```

## Call for Contributions

The realtwin project welcomes your expertise and enthusiasm!

Small improvements or fixes are always appreciated. If you run into any problems, find bugs, or think of useful improvements and enhancements, feel free to open an issue. If you add a feature or fix a bug yourself and want it considered for integration, feel free to open a pull request with the changes. Please provide a detailed description of what the pull request is doing and briefly list any significant changes made. If it's in regards to a specific issue, please include or link the issue number.

Writing code isn't the only way to contribute to realtwin. You can also:

- review pull requests
- help us stay on top of new and old issues
- develop tutorials, presentations, and other educational materials
- develop graphic design for our brand assets and promotional materials
- translate website content
- help with outreach and onboard new contributors
- write grant proposals and help with other fundraising efforts

For more information about the ways you can contribute to realtwin, visit our GitHub. If you' re unsure where to start or how your skills fit in, reach out! You can ask by opening a new issue or leaving a comment on a relevant issue that is already open on GitHub.


## Funding

This work was supported by the US Department of Energy, Vehicle
Technologies Office, Energy Efficient Mobility Systems (EEMS)
program, under project Real-Twin (EEMS114).

## Citation

To cite usage of Real-Twin, please use the folowing bibtex:

```bibtex
@article{xu2025automated,
  title        = {Developing An Automated Microscopic Traffic Simulation Scenario Generation Tool},
  author       = {Xu, Guanhao and Saroj, Abhilasha and Wang, Chieh (Ross) and Shao, Yunli},
  journal      = {Transportation Research Record},
  year         = {2025},
  publisher    = {SAGE for the National Academy of Sciences: Transportation Research Board},
  note         = {In Press, DOI to be assigned}
}