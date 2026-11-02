[![PyPI version](https://badge.fury.io/py/realtwin.svg)](https://badge.fury.io/py/realtwin)[![Downloads](https://static.pepy.tech/badge/realtwin)](https://pepy.tech/project/realtwin)[![](https://img.shields.io/pypi/wheel/gensim.svg)](https://pypi.org/project/realtwin/)[![](https://img.shields.io/pypi/pyversions/realtwin.svg)](https://www.python.org/)[![](https://readthedocs.org/projects/real-twin/badge/?version=latest)](https://real-twin.readthedocs.io/en/latest/?badge=latest)[![](https://img.shields.io/github/contributors/ORNL-Real-Sim/Real-Twin)](https://img.shields.io/github/contributors/ORNL-Real-Sim/Real-Twin)[![](https://img.shields.io/badge/License-GPL-blue.svg)](https://www.gnu.org/licenses/gpl-3.0.en.html)

> - [Real-Twin](#real-twin)
>   - [🔁 Real-Twin: A Unified Simulation Scenario Generation Tool for Mobility Research](#-real-twin-a-unified-simulation-scenario-generation-tool-for-mobility-research)
>   - [Installation](#installation)
>   - [Documentation](#documentation)
>   - [Quick Example](#quick-example)
>   - [Call for Contributions](#call-for-contributions)
>   - [Funding](#funding)
>   - [Citation](#citation)

# Real-Twin

## 🔁 Real-Twin: A Unified Simulation Scenario Generation Tool for Mobility Research

**Real-Twin** is a unified, **simulation platform-agnostic scenario generation tool** designed to streamline and standardize the evaluation of emerging mobility technologies. It provides an end-to-end framework that includes robust workflows, integrated tools, and comprehensive metrics to generate, calibrate, and benchmark microscopic traffic simulation scenarios across multiple platforms.

> This is the v0.2.0 release candidate (v0.2.0rc1); the official v0.2.0 release is coming soon.

### ✨ Key Features

- **Unified Scenario Generation**: Generate transferable, simulation-ready scenarios from heterogeneous data sources using a consistent workflow.
- **Automated Calibration Workflow**: Bridges simulation and real-world data, minimizing manual effort and making traffic simulation more accessible to researchers and engineers.
- **Simulation Platform Compatibility**: Supports **SUMO**, **VISSIM**, and **AIMSUN** for cross-platform scenario generation and benchmarking. Enables reliable comparisons and reproducibility across different simulation tools.
- **Consistent Scenarios across Different Simulators**: Generate comparable simulation scenarios across different microscopic traffic simulators, providing users the ability to conduct benchmarking and cross-validation that are crucial for ensuring the reliability and reproducibility of simulation results.
- **Emerging Technology Support**: Includes a scenario database and pipeline for studying **autonomous vehicles (AVs)**, with planned extensions to **CAVs**, **EVs**, and other advanced technologies.

## Installation

```python
pip install realtwin
```

Pandas 3 is supported on Python 3.11 and newer. Python 3.10 uses pandas 2.2 or 2.3 through the dependency requirements.

To run or debug tutorials from a cloned checkout, activate the Python environment
selected in your debugger and run this command from the repository root:

```shell
python -m pip install -e .
```

This editable installation makes the local `realtwin` package importable when
launching scripts in `tutorials/`, and source edits take effect without
reinstalling. The tutorials use `os.chdir(...)` to resolve relative data paths;
changing the working directory does not add the repository to Python's import
search path (`sys.path`).

## Documentation

User guide and API documentation can be found at: [Official Documentation](https://real-twin.readthedocs.io/en/latest/)

## Quick Example - realtwin

```python

import realtwin as rt

# Please refer to the official documentation for more details on RealTwin preparation before running the simulation

if __name__ == '__main__':

    # Step 1: Prepare your configuration file (in YAML format)
    CONFIG_FILE = "./realtwin_config.yaml"

    # Step 2: initialize the realtwin object
    twin = rt.RealTwinSUMO(input_config_file=CONFIG_FILE, verbose=True)

    # Step 3: check simulator env: if SUMO, VISSIM, Aimsun, etc... are installed
    twin.env_setup(sel_sim=["SUMO", "VISSIM"])

    # Step 4: Create Matchup Table from SUMO network
    updated_sumo_net = r"./datasets/example2/chatt.net.xml"
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

    # Step 8: perform calibration, Available algorithms: GA: Genetic Algorithm, SA: Simulated Annealing, TS: Tabu Search, BO: Bayesian Optimization
    twin.calibrate(sel_algo={"turn_inflow": "GA", "behavior": "GA"})

    # Step 9 (ongoing): post-process the simulation results
    twin.post_process()  # keyword arguments can be passed to specify the post-processing options

    # Step 10 (ongoing): visualize the simulation results
    twin.visualize()  # keyword arguments can be passed to specify the visualization options
```

For SUMO, disable external signal loading in the YAML configuration with:

```yaml
SUMO:
    is_control_data: false
```

Control CSVs are then optional; abstract generation
skips Synchro loading and signal ID reassignment, and simulation preparation
keeps the source network's existing signal logic. Traffic data and
`MatchupTable.xlsx` are still required. Omitting the flag preserves signal
loading. `generate_inputs()` sets it to false when the Control folder is missing
or empty; set `twin.input_config["SUMO"]["is_control_data"] = True` before generating
the abstract scenario if you subsequently add control data.

Vissim supports the same optional external control setting:

```yaml
VISSIM:
    is_control_data: false
```

Vissim input, abstract and concrete generation disable this flag when
`<input_dir>/Control` is missing or has no signal CSV files. An explicit `false`
also skips Synchro reads and the signal-loading stage when CSVs are present.
Demand generation continues, and simulation and calibration use
`<NetworkName>_demand.inpx` instead of `<NetworkName>_demand_signals.inpx`.
Existing source-network signals are retained. The flag defaults to `true`;
after adding control data, set `twin.input_config["VISSIM"]["is_control_data"] = True`
and regenerate the abstract and concrete scenarios to load it.

Aimsun also accepts optional external control:

```yaml
AIMSUN:
    is_control_data: false
```

Aimsun checks the `Control` folder beside `AIMSUN.model_fname`, matching its
existing data lookup. Input, abstract and concrete generation set the flag to
`false` when that folder is missing or has no signal CSV files. An explicit
`false` skips Synchro reads, detector generation, signal import and RealTwin
control-plan creation even when CSVs are present. Demand generation continues,
and simulation preparation creates the scenario without requiring or assigning
the RealTwin master control plan. Existing model control objects remain
unchanged. Calibration uses the same `.ang` model. Omitting the flag keeps
loading enabled when data is available; after adding signal CSVs, set
`twin.input_config["AIMSUN"]["is_control_data"] = True` and regenerate the abstract
and concrete scenarios.

SUMO input generation keeps the original `.net.xml` in `output/OpenDrive/`
and exports a separate `.xodr` with the same road and junction IDs. Simulation
preparation copies that SUMO network into `output/SUMO/`; it no longer rebuilds
the source through an OpenDrive round trip. Newly generated matchup tables keep
exact IDs, including leading `-` characters and underscores. Existing tables
that omitted `-` are accepted when only the corresponding negative SUMO ID exists;
if both IDs exist, the exact match takes precedence.

For an explicit conversion, use the helpers with separate output paths:

```python
from realtwin.rt_sumo.xodr_sumo import xodr_from_net_xml, xodr_to_net_xml

odr_map = xodr_from_net_xml("source.net.xml", "export.xodr")
sumo_net = xodr_to_net_xml("export.xodr", "converted.net.xml")
```

The export stores original SUMO lane IDs and signal programs in OpenDrive
`userData`; OpenDrive lane IDs remain local integers. The reverse helper restores
source IDs for roads and lanes that correspond one to one, junction IDs, and
signal programs and connection indices. Geometry and internal connections can
change during native conversion, and split roads keep generated segment IDs.
Conversion raises an error if saved signal connections cannot be restored.
Arbitrary OpenDrive files without this metadata retain native generated SUMO IDs
and record their OpenDrive origins. Both helpers require `netconvert` on `PATH`
(or an explicit executable path) and importable `sumolib`.

## Bayesian optimization for calibration

BO is available for SUMO and Aimsun turn/inflow and driving-behavior calibration
through `RealTwinSUMO.calibrate()` and
`RealTwinAimsun.calibrate()`. Vissim turn/inflow and behavior calibration use the same BO
implementation through `RealTwinVISSIM.calibrate()`. Install its optional
Gaussian-process dependencies in the Python environment used to run Real-Twin:

```powershell
conda run -n rt python -m pip install -e ".[bo]"
```

For an installed release containing BO, use `python -m pip install "realtwin[bo]"`.
SUMO and `jtrrouter` must be available on `PATH`, and `SUMO_HOME` must point to
the SUMO installation. Prepare the scenario using the existing workflow first.
GA remains the default; select BO explicitly at the calibration step:

```python
twin.calibrate(
    sel_algo={"turn_inflow": "BO", "behavior": "BO"},
    sel_behavior_routes=sel_behavior_routes,
    update_turn_inflow_algo={"bo_config": {"max_evaluations": 50}},
    update_behavior_algo={"bo_config": {"max_evaluations": 30}},
)
```

Here `sel_behavior_routes` is the existing mapping of observed travel times in
seconds and SUMO edges, for example
`{"route_1": {"time": 100, "route_list": ["edge_1", "edge_2"]}}`.
Use actual routes from your network. SUMO behavior-only calibration uses existing
turn/inflow output when present, or the network and demand from `prepare_simulation()`
when turn/inflow calibration has not been run.

Shared settings belong under `Calibration.bo_config` in the YAML configuration.
Per-stage overrides above merge with these shared settings.


| Setting           | Default | Meaning                                                                                      |
| ------------------- | --------- | ---------------------------------------------------------------------------------------------- |
| `kernel_type`     | `RBF`   | `RBF`, `Matern`, `RationalQuadratic`, `ExpSineSquared`, or `Combined` (RBF plus white noise) |
| `target`          | `0`     | Desired minimum fitness                                                                      |
| `tolerance`       | `3`     | Stop when best fitness is at most `target + tolerance`                                       |
| `random_points`   | `4000`  | Candidate pool size; at least `max_evaluations`                                              |
| `max_evaluations` | `100`   | Objective evaluations per run, including initialization                                      |
| `total_run`       | `1`     | Independent runs, from 1 through 1000                                                        |
| `seed`            | `812`   | Run seeds are `seed`, `seed + 1`, and so on                                                  |

The periodic kernel uses one fixed period per parameter, equal to that parameter's
bound span. Its covariance supports multiple parameters.

Turn/inflow fitness is mean GEH. Behavior fitness is travel-time RMSE for SUMO
and travel-time MAE for Aimsun, both in seconds.
Choose each stage's tolerance accordingly. Turn ratios use bounds `[0, 1]`;
inflow bounds use `Calibration.turn_inflow.max_inflow`. Behavior uses
`Calibration.behavior.params_ranges` in the existing parameter order:
`min_gap`, `acceleration`, `deceleration`, `sigma`, `tau`, `emergencyDecel`.

The implementation adapts the GitLab Real-Twin BO workflow: space-filling
initial samples, a Gaussian-process surrogate, and expected improvement blended
with decreasing exploration. It normalizes parameters by their bounds, supports
constant objectives and fixed parameters, and does not increase the requested
evaluation budget. A fixed seed reproduces the search for a deterministic
objective; simulator randomness and existing behavior-parameter repairs can
still affect simulation results.

Each enabled stage runs its best candidate once more to restore the generated
simulation files. This final application is **one additional simulation per
stage**, outside the search budget and CSV evaluation history. Outputs are saved
under `output/SUMO/turn_inflow/turn_inflow_bo_result/` and
`output/SUMO/behavior/behavior_bo_result/`:

- `sumo_bayesopt_<N>runs_results.csv`: fitness at every evaluation.
- `sumo_bayesopt_<N>runs_points.csv`: proposed parameter values and fitness.
- `sumo_bayesopt_<N>runs_best_per_run.csv`: best candidate and elapsed search time for each run.
- Per-run and best-run convergence PNGs showing evaluated fitness and best fitness so far.

CSV parameters retain the source names `param_1`, `param_2`, etc. Turn parameters
precede inflow parameters; SUMO behavior parameters follow the order above.
Aimsun and Vissim output conventions are described below.

### Independent calibration stages

For SUMO and Aimsun, set the two switches independently in your existing YAML
configuration. For a behavior-only run:

```yaml
Calibration:
    turn_inflow:
        is_calibration: false
    behavior:
        is_calibration: true
```

Keep the rest of your configuration, including the model/network paths and BO
settings. Reverse the flags for turn/inflow-only calibration, or enable both to
run them in sequence. Disabled stages can be omitted from `sel_algo`; their
algorithm selections are ignored. If both flags are false, `calibrate()`
returns `False` without starting calibration.

For an existing `RealTwinAimsun` instance with behavior enabled:

```python
twin.calibrate(
    sel_algo={"behavior": "BO"},
    sel_behavior_routes=[("Subpath1", 2071, 2092, 60.0)],
    update_behavior_algo={"bo_config": {"max_evaluations": 30}},
)
```

Each Aimsun route is `(name, start_section_id, end_section_id, observed_seconds)`;
replace these sample IDs with sections in your model. Routes can also be stored
in `Calibration.behavior.sel_behavior_routes`. An enabled behavior stage with
no routes returns `False` and explains the missing input without changing the
enable flag. Turn/inflow-only calibration needs no behavior routes:

```python
twin.calibrate(sel_algo={"turn_inflow": "BO"})
```

Aimsun behavior-only calibration uses demand already in the model. It exports
replication metadata and creates the requested subpaths without running the
turn/inflow optimizer. Complete `env_setup()` and prepare an Aimsun model with
a replication and SQLite output before calibration. BO runs in the Python
environment hosting Real-Twin; the Aimsun console continues to run the existing
assignment and simulation scripts.

Aimsun BO searches normalized `[0, 1]` inputs. Turning weights are converted
to percentages summing to 100 per approach; normalized inflows are multiplied
by `Calibration.turn_inflow.max_inflow` in vehicles/hour. Behavior values are
mapped once to their physical ranges. `params_ranges` accepts the shared names
listed above or their Aimsun equivalents, in canonical CSV parameter order:
`MinDist`, `MaxAcc`, `NormalDec`, `SensitivityFactor`, `MinHeadway`, `MaxDec`.
The assignment CSV is reordered for Aimsun's existing script.

Aimsun exports `aimsun_bayesopt_<N>runs_results.csv`,
`aimsun_bayesopt_<N>runs_points.csv`, `aimsun_bayesopt_<N>runs_best_per_run.csv`,
and convergence PNGs under `turn_inflow_bo_result/` or
`behavior_bo_result/` beside the `.ang` model. Its `param_*` columns
contain normalized inputs. As in SUMO, the best candidate is applied in one
additional simulation per enabled stage, outside the search budget.

### Calibration code organization

The shared functions in
[`_calibration_workflow.py`](realtwin/func_lib/_f_calibration/_calibration_workflow.py)
validate stage selections, merge overrides, and dispatch GA/SA/TS/BO.
GA model construction and result exports are implemented directly in each
calibration adapter.
Each simulator keeps its stage preparation, route handling, result paths,
and cleanup in its own calibration workflow. The BO search and result exporter
remain in `algo_sumo/_bayesian_opt_util.py` and `algo_sumo/_bayesian_opt.py`,
with simulator-specific objective callbacks and parameter bounds supplied by adapters.

Aimsun's [`_console.py`](realtwin/rt_aimsun/_console.py) owns subprocess output
capture. Both Aimsun calibration modules still expose `run_aconsole` for existing
callers. The calibration regression tests cover stage independence, default
algorithms, override precedence, best-candidate application, and output exports.

### Vissim calibration

`RealTwinVISSIM.calibrate()` searches unmeasured vehicle inputs and turn splits
using GA (default), SA, TS, or BO. Prepare the demand/signals network first, fill in
the Vissim `MatchupTable.xlsx` with observed GridSmart files, and mark the
junctions to adjust with `Need calibration? = Y`. Enable
`Calibration.turn_inflow.is_calibration` for this stage. Behavior calibration
is independently enabled by `Calibration.behavior.is_calibration` and minimizes
route travel-time MAE in seconds. When both are enabled, behavior uses the model
saved by turn/inflow calibration. A demand stage with no variables passes its
input model forward, so a previous output cannot be mistaken for this run.

```python
twin.input_config["Calibration"]["turn_inflow"]["is_calibration"] = True
twin.input_config["Calibration"]["behavior"]["is_calibration"] = False
twin.calibrate(
    sel_algo={"turn_inflow": "SA"},
    update_turn_inflow_algo={
        "sa_config": {"epoch": 2, "step_size": 0.05},
        "turn_inflow": {"max_inflow": 500, "seed": 42, "optimizer_seed": 812},
    },
)
```

The package stage receives the live settings and per-call overrides without
changing the shared `Calibration` mapping or reloading the original YAML file.
`realtwin/rt_vissim/calibrate_vissim.py` imports `TurnInflowCaliVissim` from
`cali_turn_inflow.py` and `BehaviorCaliVissim` from `cali_behavior.py`, and calls
the shared `run_calibration_algorithm` dispatcher. Both adapters expose
`run_GA`, `run_SA`, `run_TS` and `run_BO`, matching the Aimsun contract.
BO uses the existing `BayesianOptimization` with normalized parameter bounds and the shared
`bo_config` keys described above. For example, select
`sel_algo={"turn_inflow": "BO"}` with
`update_turn_inflow_algo={"bo_config": {"max_evaluations": 20}}`.
It uses `<output_dir>/VISSIM/<NetworkName>_demand_signals.inpx` (or
`<NetworkName>_demand.inpx` when `VISSIM.is_control_data` is false), the adjacent
matchup/link/movement tables, and `<input_dir>/Traffic`. The effective
`scenario_config` supplies the time window (seconds after midnight), GEH target
and accepted share. `calibration_time_step` is seconds per simulation step and
must correspond to 1–20 Vissim steps per second. Counts must cover complete
15-minute intervals across the entire window without gaps, and all observed movements must have
network paths that can be measured.

Optional `turn_inflow` settings are `objective` (`approach` or `movement`),
`budget` (approximate optimizer simulation count), `seed` (Vissim),
`optimizer_seed`, and `out` (a separate calibrated `.inpx` path). `VISSIM.progid`
can pin the installed COM version. Each candidate resets the simulation and
reuses the same seed. The best demand is saved to
`<source_stem>_calibrated_<algo>.inpx` by default, with history/approach CSVs,
a summary JSON recording effective settings, and progress charts. The source
network is preserved. BO also writes `vissim_bayesopt_*` CSVs and convergence
plots to `<calibrated_stem>_bo_result/`. A stage failure raises an exception; successful or
no-variable runs return `True`, and disabled/unsupported calibration returns
`False`.

GA/SA/TS honor `max_epoch`, `max_fe`, `max_time` (seconds) and `max_early_stop`
from their stage block, falling back to `scenario_config` when present. The
algorithm's `epoch` remains a cap. BO uses `bo_config.max_evaluations`,
`total_run`, `target` and `tolerance` for its existing shared stop behavior.

Behavior requires native Vissim `params_ranges`, explicit `driving_behavior_ids`,
and routes as `[name, start_link, end_link, observed_seconds]`. Measurements run
from position zero on the start link to the end of the end link, aggregating
completed vehicles across all vehicle classes. For example, for a model using
Wiedemann 74 driving behavior 1:

```python
twin.calibrate(
    sel_algo={"behavior": "SA"},
    sel_behavior_routes=[["corridor", 33, 23, 30.0]],  # replace with your link IDs/measurement
    update_turn_inflow_algo={"turn_inflow": {"is_calibration": False}},
    update_behavior_algo={
        "behavior": {
            "is_calibration": True,
            "driving_behavior_ids": [1],
            "params_ranges": {"W74ax": [1.0, 3.0], "W74bxAdd": [1.0, 4.0]},
            "route_positions": {"corridor": {"start": 0.0, "end": 0.5}},
            "seed": 42, "optimizer_seed": 812,
        },
        "sa_config": {"epoch": 2, "step_size": 0.05},
    },
)
```

Use ranges appropriate to your model and observations. Supported continuous
parameters are `W74ax`, `W74bxAdd`, `W74bxMult`, `W99cc0` and `W99cc2`–`W99cc9`.
The selected driving behaviors must use the corresponding Wiedemann model.
`W99cc1Distr` is a distribution reference and is not a continuous headway.
SUMO/Aimsun parameter names in the shared defaults must be replaced explicitly.
For a configuration shared across simulators, put Vissim's settings under
`Calibration.behavior.vissim`, for example:

```yaml
Calibration:
    behavior:
        is_calibration: true
        vissim:
            driving_behavior_ids: [1]
            calibration_time_step: 0.1
            params_ranges:
                W74ax: [1.0, 3.0]
                W74bxAdd: [1.0, 4.0]
```

This block replaces the corresponding shared behavior settings for Vissim;
native ranges replace the whole shared range mapping. Per-call
`update_behavior_algo` settings then take precedence. The shared configuration
is unchanged. Enabled behavior inputs are validated before either calibration
stage starts, so missing ranges or routes raise a descriptive `ValueError`
directly in the calling process.
Stage `calibration_time_step` values override the shared scenario step. The
tutorial uses 0.1 seconds to match its signal controller's 10 updates per second.

Routes can also be provided in `Calibration.behavior.sel_behavior_routes`;
Vissim-specific routes can go in `Calibration.behavior.vissim.sel_behavior_routes`.
The method argument takes precedence. Missing completed vehicles receive
`missing_route_penalty + observed_seconds` (default penalty: 10,000 seconds).
If the best candidate still lacks any route measurement, calibration raises
an error without saving a calibrated model. Behavior writes
`<source_stem>_behavior_<algo>.inpx`, history/routes CSVs and a summary JSON;
BO also exports its convergence results. Behavior settings support `budget`,
`seed`, `optimizer_seed`, `out` and independent algorithm/scenario overrides.
Optional `route_positions` maps route names to `start`/`end` positions in metres
along their respective links. Match them to the field measurement locations and
place them where vehicles cross: a connector can divert vehicles before the
physical end of a link. Defaults are zero on the start link and the full end-link
length. The summary records the actual marker positions.
It also preserves an optional `reference_source` mapping, so simulated tutorial
references can be distinguished from field observations.
The Vissim tutorial selects the westbound route from link 9 to link 15, with
markers at 0.0 and 0.5 metres respectively. Its 39.198670-second target is a
Vissim baseline mean for 162 completed vehicles during 08:00–09:00, seed 42;
it is a simulated example reference, not a measured field travel time. The
reference record is saved under the generated VISSIM `behavior_reference`
folder with network/signal checksums. Replace the route target and its
`reference_source` metadata with field observations for real calibration.

For standalone validation use
`python -m realtwin.rt_vissim.06_calibrate_turn_inflow --help` and pass
`--inpx`, `--matchup`, `--links`, `--movements`, `--traffic-dir`, and optionally
`--config <yaml-or-json>`. `--dry-run` validates settings and lists variables
without starting Vissim. `--stage behavior --inpx <model> --config <config>`
selects behavior and does not require Traffic or matchup tables. Real calibration requires Windows, pywin32 and a
Vissim COM licence.

## Quick Example - Autonomous Vehicle

```python
import realtwin as rt

# Please refer to the official documentation for more details on RealTwin preparation before running the simulation

if __name__ == '__main__':

    # Step 1: Prepare/generate configuration file (in YAML format)
    rt.prepare_av_configs()
    CONFIG_FILE = "path-to-generated-config-file"

    # Step 2: Update the configuration file
    # Manually update the configuration file from User.

    # Step 3: initialize the SimAV object
    sim = rt.SimAV(input_config_file=CONFIG_FILE, verbose=True)

    # Step 4: Simulation generation
    sim.run_simulation()

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

This work is supported by the US Department of Energy, Vehicle Technologies Office, Energy Efficient Mobility Systems (EEMS) program, under project Real-Twin (EEMS114).

## Citation

To cite usage of Real-Twin, please use the folowing bibtex:

```bibtex

@article{xu2025automated,
  title        = {Developing An Automated Microscopic Traffic Simulation Scenario Generation Tool},
  author       = {Xu, Guanhao and Saroj, Abhilasha and Wang, Chieh (Ross) and Shao, Yunli},
  journal      = {Transportation Research Record},
  year         = {2025},
  doi          = {https://doi.org/10.1177/03611981251349433},
  publisher    = {SAGE for the National Academy of Sciences: Transportation Research Board},
}

@misc{ doecode_147051,
title = {Real-Twin},
author = {Wang, Chieh (Ross) and Xu, Guanhao and Saroj, Abhilasha and Luo, Xiangyong (Roy) and Yuan, Jinghui and Shao, Yunli},
doi = {10.11578/dc.20250602.3},
url = {https://doi.org/10.11578/dc.20250602.3},
howpublished = {[Computer Software] \url{https://doi.org/10.11578/dc.20250602.3}},
year = {2025},
month = {jun}
}

```
