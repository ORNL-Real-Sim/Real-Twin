# Real-Twin Tutorials

This page is the starting point for running the examples in this repository. It explains what each tutorial does, which files are used, what must be prepared manually, and the order in which the Real-Twin methods are called.

## What is included?

| Tutorial            | Simulator   | Entry point                                                                                                           | Purpose                                                                 |
| ------------------- | ----------- | --------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| SUMO workflow       | SUMO        | [`tutorial_realtwin_SUMO.py`](tutorial_realtwin_SUMO.py) or [`tutorial_realtwin_SUMO.ipynb`](tutorial_realtwin_SUMO.ipynb) | Build, prepare, and calibrate a scenario from the example data.           |
| Vissim workflow     | PTV Vissim  | [`tutorial_realtwin_Vissim.py`](tutorial_realtwin_Vissim.py)                                                             | Import a SUMO network, prepare native Vissim demand/signals, and calibrate. |
| Aimsun workflow     | Aimsun Next | [`tutorial_realtwin_Aimsun.py`](tutorial_realtwin_Aimsun.py)                                                             | Prepare an Aimsun model or calibrate an already prepared model.           |
| Autonomous vehicles | SUMO        | [`realtwin_av_config.yaml`](realtwin_av_config.yaml) plus `SimAV`                                                      | Add Human, AV, CAV, or other vehicle types to a SUMO simulation.           |

The first three tutorials use the same broad scenario workflow:

1. Load a YAML configuration.
2. Check the simulator environment.
3. Generate or import the network and create `MatchupTable.xlsx`.
4. Add traffic demand, matchup-table mappings, and optional external signal-control data.
5. Generate the abstract and concrete scenarios.
6. Prepare the simulator files; the Vissim preparation method also runs a short simulation.
7. Calibrate the enabled stages. Post-processing and visualization remain ongoing tutorial hooks.

The scripts share [`realtwin_config.yaml`](realtwin_config.yaml). Its current `input_dir` is `datasets/example1`, and an unset `output_dir` resolves to `datasets/example1/output`. The SUMO and Vissim scripts separately pass `datasets/example2/updated.net.xml` as the network input. Aimsun reads its traffic/control data and matchup table beside `AIMSUN.model_fname`.

The AV tutorial is a separate, shorter SUMO workflow. It does not use `RealTwinSUMO.generate_abstract_scenario()` or the matchup table.

## Prerequisites

- Windows. The current Real-Twin package supports Windows only.
- Python 3.10 or newer.
- A clone of this repository, or an installed Real-Twin package.
- The Python dependencies in [`requirements.txt`](../requirements.txt).
- SUMO for the SUMO and AV tutorials, and `netconvert` for the Vissim network import. SUMO calibration also requires `sumo`, `jtrrouter`, and `netconvert` on `PATH`, with `SUMO_HOME` pointing to the installation. Real-Twin can check for SUMO and, when requested, install a SUMO version.
- Windows, PTV Vissim with a valid COM license, and `pywin32` in the Real-Twin Python environment for the Vissim tutorial.
- Aimsun Next, a valid license, and a compatible Aimsun Python installation for the Aimsun tutorial. Real-Twin cannot replace the Aimsun license or model files.

### Install from the repository

Open a terminal at the repository root (`Real-Twin`) and run:

```text
python -m venv .venv
```

Activate the environment:

```text
# Windows PowerShell
.venv\Scripts\Activate.ps1

# Windows Command Prompt
.venv\Scripts\activate.bat

```

Install Real-Twin and its dependencies:

```text
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

You can also install the published package with `python -m pip install realtwin`, but the tutorial scripts and example datasets are available in the repository checkout.

The current scenario scripts select Bayesian optimization (`BO`). Install its optional dependencies in the environment that runs Real-Twin:

```text
python -m pip install -e ".[bo]"
```

For a published release containing BO, use `python -m pip install "realtwin[bo]"`. Genetic algorithm (GA), simulated annealing (SA), and tabu search (TS) do not require this extra. On Windows, install the Vissim COM dependency separately:

```text
python -m pip install pywin32
```

For the optional SUMO notebook, install Jupyter as well:

```text
python -m pip install notebook
```

## Run the core SUMO workflow

The SUMO example uses [`realtwin_config.yaml`](realtwin_config.yaml) for the `datasets/example1` demand data and imports `datasets/example2/updated.net.xml`. Run it from the repository root:

```text
python tutorials/tutorial_realtwin_SUMO.py
```

The script performs these operations:

```python
import realtwin as rt

twin = rt.RealTwinSUMO(
    input_config_file="./tutorials/realtwin_config.yaml",
    verbose=True,
)
twin.env_setup()  # Defaults to SUMO.
twin.generate_inputs(incl_sumo_net="./datasets/example2/updated.net.xml")
twin.generate_abstract_scenario()
twin.generate_concrete_scenario()
twin.prepare_simulation()
sel_behavior_routes = {
    "route_1": {
        "time": 100,
        "route_list": ["-90", "-96", "-103", "-105", "-106", "-99", "-94"],
    },
    "route_2": {
        "time": 150,
        "route_list": ["-97", "-95", "-100", "-107", "-104", "-102", "-93"],
    },
}
twin.calibrate(
    sel_algo={"turn_inflow": "BO", "behavior": "BO"},
    sel_behavior_routes=sel_behavior_routes,
)
```

The route IDs and travel times in this snippet match the script's example inputs. Replace them with routes and observed travel times in seconds for your network. The checked-in YAML enables behavior calibration and disables turn/inflow calibration, so only behavior runs even though the script selects algorithms for both stages.

The example script also calls `post_process()` and `visualize()`. Those stages are still marked as ongoing in the example and may require additional result-specific options.

### What to do when the script pauses

`generate_inputs()` creates or refreshes the input folders and generates `datasets/example1/MatchupTable.xlsx`. Before continuing:

1. Put traffic-demand turn-movement files in `datasets/example1/Traffic/`.
2. If external control loading is enabled, put the Synchro UTDF signal CSV in `datasets/example1/Control/`. Prepare it before starting the script to avoid automatic disabling when the folder is empty.
3. Open `datasets/example1/MatchupTable.xlsx` and fill in the demand filenames and required mapping values. Fill in control references only when external control is enabled.
4. Save the table, then press Enter in the terminal.
5. When asked again during abstract-scenario generation, check that the demand movements and any enabled signal movements match the network bearings, then continue.

The configuration file contains the network name, vertices, input/output directories, simulation time window, calibration targets, and algorithm settings. Change those values for a different dataset. `NetworkName` should not contain spaces.

## Optional external signal-control data

All three scenario classes support demand generation without external signal CSVs. Set `is_control_data: false` in the corresponding simulator section of your existing YAML configuration:

```yaml
SUMO:
    is_control_data: false
VISSIM:
    is_control_data: false
AIMSUN:
    is_control_data: false
```

Keep the other simulator settings, including Aimsun model/executable paths. Omitting the flag keeps external control loading enabled when data is available; an explicit `false` skips it even when CSVs are present. Traffic data and the simulator's matchup table remain required.

| Simulator | Control folder | Automatic disabling | Behavior with external control disabled |
| --------- | -------------- | ------------------- | --------------------------------------- |
| SUMO | `<input_dir>/Control` | `generate_inputs()` finds a missing or empty folder. | Skip Synchro loading, signal ID reassignment, and external signal import; preserve the source network's existing signals. |
| Vissim | `<input_dir>/Control` | Input, abstract, or concrete generation finds no signal CSV files. | Skip signal-column derivation and signal loading; simulation and calibration use `<NetworkName>_demand.inpx`. |
| Aimsun | `Control` beside `AIMSUN.model_fname` | Input, abstract, or concrete generation finds no signal CSV files. | Skip detector/signal/control-plan creation; prepare the scenario without assigning the RealTwin master control plan. Existing model control objects remain unchanged. |

Automatic disabling updates the live `twin.input_config`. Adding CSVs afterwards does not re-enable it. For example, when adding Vissim control data during an interactive workflow:

```python
twin.input_config["VISSIM"]["is_control_data"] = True
twin.generate_abstract_scenario()
twin.generate_concrete_scenario()
twin.prepare_simulation()
```

Use `"SUMO"` or `"AIMSUN"` for the other classes, and complete the control references in their matchup tables before regenerating. SUMO's current prompts may still mention control data even when its flag is false; only demand preparation is required in that mode.

## SUMO tutorial

Use the SUMO script when you want a repeatable run:

```text
python tutorials/tutorial_realtwin_SUMO.py
```

Use the notebook when you want to inspect each stage interactively:

```text
python -m jupyter notebook tutorials/tutorial_realtwin_SUMO.ipynb
```

The SUMO workflow uses the same input conventions as the general example. A SUMO network can be supplied through `generate_inputs(incl_sumo_net="path/to/network.net.xml")`. If that argument is omitted, Real-Twin creates a network from the vertices in the YAML configuration.

To check or select a SUMO installation explicitly:

```python
twin.env_setup(sel_sim=["SUMO"])

# Search additional directories for SUMO executables.
twin.env_setup(sel_sim=["SUMO"], sel_dir=[r"C:\path\to\SUMO\bin"])

# Require a particular SUMO version. Installation may be attempted if it is missing.
twin.env_setup(
    sel_sim=["SUMO"],
    strict_sumo_version="1.21.0",
)
```

SUMO input generation keeps the supplied network in `<output_dir>/OpenDrive/<NetworkName>.net.xml` and exports a separate `.xodr`. Simulation preparation uses the SUMO network in `<output_dir>/SUMO/` without rebuilding it through an OpenDRIVE round trip. Matchup tables and route definitions retain exact edge IDs, including leading `-` characters and underscores; keep those IDs when entering your observations.

For a deliberate conversion, use separate output paths:

```python
from realtwin.rt_sumo.xodr_sumo import xodr_from_net_xml, xodr_to_net_xml

xodr_from_net_xml("source.net.xml", "export.xodr")
xodr_to_net_xml("export.xodr", "converted.net.xml")
```

Both helpers require `netconvert` and importable `sumolib`. Real-Twin's OpenDRIVE export stores source identities and signal metadata for reverse conversion. Native conversion can change geometry/internal topology, and split roads retain generated segment IDs; an arbitrary OpenDRIVE file without that metadata cannot restore the original SUMO identities.

## Vissim tutorial

Run the Vissim script from the repository root after installing SUMO's conversion tools, Vissim, and `pywin32`:

```text
python tutorials/tutorial_realtwin_Vissim.py
```

The workflow uses the native Vissim class:

```python
import realtwin as rt

twin = rt.RealTwinVISSIM(
    input_config_file="./tutorials/realtwin_config.yaml",
    verbose=True,
)
twin.env_setup(sel_sim=["VISSIM"])
twin.generate_inputs(incl_sumo_net="./datasets/example2/updated.net.xml")

# Complete the generated Vissim matchup table at the preparation prompt.
twin.generate_abstract_scenario()
twin.generate_concrete_scenario()
twin.prepare_simulation()
twin.calibrate(sel_algo={"turn_inflow": "BO", "behavior": "BO"})
```

`generate_inputs()` converts the selected SUMO network to OpenDRIVE, imports it into Vissim, and exports native link, movement, and junction tables under `<output_dir>/VISSIM/`. Its `incl_sumo_net` argument takes precedence over `VISSIM.sumo_xml_file` for this non-demo example. If you omit the argument, update `VISSIM.sumo_xml_file` to a valid network path; the checked-in value is machine-specific.

### Prepare the Vissim matchup table

1. Put GridSmart turning-count workbooks in `<input_dir>/Traffic/` and, when enabled, Synchro signal CSVs in `<input_dir>/Control/` before running the script.
2. Edit `<output_dir>/VISSIM/MatchupTable.xlsx` (`datasets/example1/output/VISSIM/MatchupTable.xlsx` with the current configuration).
3. Use `<NetworkName>_junctions.csv` to identify each surveyed junction. Fill in `File_GridSmart` and, when control is enabled, `File_Synchro` and `IntersectionID_Synchro`.
4. Keep the generated `FromLinkNo_Vissim`, `ToLinkNo_Vissim`, and `InternalLinks_Vissim` values. Save and close the workbook, then continue. Abstract generation derives the remaining demand/signal fields.

A SUMO workbook cannot replace the native Vissim table. To copy its observation references through the exported junction mapping, run stage 2 as a package module:

```text
python -m realtwin.rt_vissim.02_matchup_table --movements datasets/example1/output/VISSIM/chatt_movements.csv --outdir datasets/example1/output/VISSIM --traffic-dir datasets/example1/Traffic --control-dir datasets/example1/Control --seed-from datasets/example1/MatchupTable.xlsx
```

Append `--no-control` for demand-only preparation. A table with missing native link numbers or movement IDs from another network is rejected. Back up that table before using `--regenerate`, then seed or enter its observation references again. Use package-module commands (`python -m realtwin.rt_vissim.<stage>`) for standalone stages so imports resolve correctly.

Concrete generation writes `<NetworkName>_demand.inpx` and, with control enabled, `<NetworkName>_demand_signals.inpx`. `prepare_simulation()` runs the appropriate network for 900 seconds. This short run is separate from the YAML calibration window, currently 08:00–09:00.

### Vissim calibration inputs and results

Turn/inflow calibration uses the native matchup table and observed counts covering the complete calibration window. Mark junctions to adjust with `Need calibration? = Y` and enable `Calibration.turn_inflow.is_calibration`.

Behavior calibration requires native Vissim parameter ranges, driving-behavior IDs, and routes as `[name, start_link, end_link, observed_seconds]`. The current YAML places these settings under `Calibration.behavior.vissim`: Wiedemann 74 behavior ID 1, `W74ax`/`W74bxAdd` ranges, seed 42, and route `Shallowford_WB_9_to_15` from link 9 to link 15. Its markers are 0.0 and 0.5 metres along those links, and its calibration step is 0.1 seconds for the example's 10/s controller.

The route's 39.198670-second target comes from a simulated baseline, with provenance in [`Shallowford_WB_baseline.json`](../datasets/example1/output/VISSIM/behavior_reference/Shallowford_WB_baseline.json). Replace the target, native IDs, marker locations, ranges, and `reference_source` metadata as appropriate for your model and field observations. Calling `calibrate()` without `sel_behavior_routes` uses the configured route; passing routes explicitly takes precedence.

When both stages are enabled, behavior uses the demand stage's saved network. The prepared source network is preserved. Demand calibration writes `<source_stem>_calibrated_<algo>.inpx`; behavior writes `<source_stem>_behavior_<algo>.inpx`, together with history/measurement CSVs and summary JSONs. BO also exports `vissim_bayesopt_*` CSVs and convergence plots in `<calibrated_stem>_bo_result/`. Missing completed route measurements can cause calibration to fail before saving a model.

## Aimsun tutorial

Aimsun uses a separate class and configuration because it imports and drives an Aimsun model. Start with:

```text
python tutorials/tutorial_realtwin_Aimsun.py
```

Before running it, edit the `AIMSUN` section of [`realtwin_config.yaml`](realtwin_config.yaml):

- `model_fname`: the absolute path to the `.ang` Aimsun model.
- `model_xdor`: the path to the model's OpenDRIVE file, when required by the model.
- `site_packages`: the site-packages directory for the Python version used by Aimsun.
- `aimsun_exe_path`: the full path to the licensed Aimsun console executable, such as `C:\Program Files\Aimsun\Aimsun Next 23\aconsole.exe`. `sel_dir` can help discover the installation, but the version check and subsequent stages use this configured path.

The Aimsun model and OpenDRIVE paths in the checked-in configuration are machine-specific examples. They must be replaced on another computer. Aimsun Next 23, for example, uses Python 3.10; install packages such as `numpy`, `pandas`, `openpyxl`, and `xlrd` into the site-packages directory used by that Aimsun installation.

The checked-in script currently calls `env_setup()` and BO calibration on an already prepared model. Its input, abstract, concrete, and simulation-preparation calls are commented out. For a new model, enable those calls in the script or run the following sequence interactively before calibration:

```python
import realtwin as rt

twin = rt.RealTwinAimsun(
    input_config_file="./tutorials/realtwin_config.yaml",
    verbose=True,
)
twin.env_setup(sel_sim=["AIMSUN"])
twin.generate_inputs()
twin.generate_abstract_scenario()
twin.generate_concrete_scenario()
twin.prepare_simulation()
```

`generate_inputs()` opens the configured Aimsun model, creates `Control/` and `Traffic/` folders beside it, and generates `MatchupTable.xlsx` there. Enter the traffic workbook for each surveyed native node and add Synchro file/intersection references when control is enabled. Prepare signal CSVs before input generation or re-enable the live flag afterwards as described above. The integration pauses for manual preparation and movement checks. Work on a copy of your `.ang` model because preparation stages save changes to it.

`prepare_simulation()` creates the RealTwin scenario, experiment, and replication and configures SQLite output. Aimsun calibration requires a prepared model with a replication and SQLite output. The script passes this behavior route:

```python
twin.calibrate(
    sel_algo={"turn_inflow": "BO", "behavior": "BO"},
    sel_behavior_routes=[("Subpath1", 2071, 2092, 60.0)],
)
```

Routes are `(name, start_section_id, end_section_id, observed_seconds)` tuples. Replace the example section IDs and 60-second target with your own observations, or configure routes under `Calibration.behavior.sel_behavior_routes`. They are required only when behavior calibration is enabled. A behavior-only run uses demand already in the model. Aimsun BO outputs go to `turn_inflow_bo_result/` or `behavior_bo_result/` beside the `.ang` model, with `aimsun_bayesopt_*` CSVs and convergence plots. Post-processing and visualization remain commented out in the script.

## Calibration settings shared by the scenario tutorials

SUMO, Vissim, and Aimsun support `GA` (genetic algorithm, the default), `SA` (simulated annealing), `TS` (tabu search), and `BO` (Bayesian optimization). Select an algorithm through `sel_algo`; stage switches in the existing YAML independently determine what runs:

```yaml
Calibration:
    turn_inflow:
        is_calibration: false
    behavior:
        is_calibration: true
```

This is the current tutorial's behavior-only selection. Keep the remaining calibration settings when editing these flags. Reverse the flags for turn/inflow only or set both to `true` to run both stages. Disabled stages may be omitted from `sel_algo`; if both are disabled, `calibrate()` returns `False` without running an optimizer.

For a prepared Vissim `twin` with behavior enabled and routes configured in the YAML, a smaller BO search can be selected with:

```python
twin.calibrate(
    sel_algo={"behavior": "BO"},
    update_behavior_algo={"bo_config": {"max_evaluations": 30}},
)
```

SUMO and Aimsun accept the same `update_behavior_algo` override; retain their `sel_behavior_routes` argument when supplying observations in Python. Aimsun can instead read routes from the YAML. The current `Calibration.bo_config` uses `kernel_type: RBF`, `target: 0`, `tolerance: 3`, `random_points: 4000`, `max_evaluations: 100`, `total_run: 1`, and `seed: 812`. The candidate pool must contain at least `max_evaluations` points, and the evaluation budget includes initialization. BO can stop early when fitness reaches `target + tolerance`. SUMO/Aimsun apply the best candidate in an additional simulation outside the search budget.

Calibration start/end values are seconds after midnight; the current window is 28800–32400 (08:00–09:00). SUMO's no-argument `prepare_simulation()` defaults to 08:00–10:00. To prepare the same window as calibration, pass `start_time=28800, end_time=32400` explicitly. For objective definitions and further optimizer settings, see the [README calibration guide](../README.md#bayesian-optimization-for-calibration).

## Autonomous-vehicle tutorial

The AV example runs a SUMO simulation using vehicle-type penetration rates and car-following/lane-changing parameters. It uses the files in `datasets/autonomous_veh`:

- `chatt.net.xml`: SUMO network.
- `chatt.flow.xml`: traffic flow input.
- `chatt.turn.xml`: turn input.
- [`realtwin_av_config.yaml`](realtwin_av_config.yaml): example AV configuration.

Run it from Python:

```python
import realtwin as rt

sim = rt.SimAV(
    path_config="./tutorials/realtwin_av_config.yaml",
    verbose=True,
)
sim.run_simulation()
```

The configuration controls `veh_types`, their `pct_penetration`, the SUMO car-following and lane-changing models, and the simulation duration. The penetration percentages must add up to 100. Results and generated SUMO files are placed in `datasets/autonomous_veh/output_AV/`.

To create a starter AV configuration instead of using the checked-in example:

```python
import realtwin as rt

rt.prepare_av_configs(dest_dir=".")
```

Edit the generated `config_av.yaml`, especially the input filenames and vehicle parameters, before passing it to `SimAV`.

## Virtual environment helpers

Real-Twin also exposes helpers for creating and deleting a project virtual environment:

```python
import realtwin as rt

rt.venv_create(venv_name="venv_rt", venv_dir="")
rt.venv_delete(venv_name="venv_rt", venv_dir="")
```

In most cases, the standard `python -m venv .venv` commands in the installation section are easier to inspect and activate manually.

## Common problems

- **`ModuleNotFoundError`**: activate the intended virtual environment and install `requirements.txt`.
- **SUMO is not found**: install SUMO, add its `bin` directory with `sel_dir`, or use `strict_sumo_version`.
- **BO dependency errors**: install the `bo` extra in the environment running Real-Twin, or select GA, SA, or TS.
- **The script stops for input**: this is expected. Complete the traffic and matchup-table preparation, plus control references when enabled. Vissim's workbook is under `<output_dir>/VISSIM`; Aimsun's is beside the model.
- **Signal data is skipped after adding CSVs**: reset the simulator's live `is_control_data` flag to `True`, update the matchup references, and regenerate abstract/concrete scenarios and simulation preparation.
- **Vissim COM errors**: verify Windows, `pywin32`, and a valid Vissim COM license. `VISSIM.progid` can select a specific installed COM version for calibration.
- **Vissim rejects the matchup table**: preserve a backup, regenerate from the current native movement table, and seed or enter references again. Keep native link IDs.
- **Calibration does not run the selected stage**: check its `is_calibration` flag. Algorithm selection alone does not enable a stage. Aimsun behavior also needs routes with observed times; Vissim behavior needs native ranges and driving-behavior IDs.
- **Aimsun cannot be found**: verify that `aconsole.exe` is installed, licensed, and discoverable, and update `AIMSUN.aimsun_exe_path` to its full path.
- **Aimsun Python import errors**: use the Python version expected by the Aimsun release and point `site_packages` to that environment.
- **Missing network or input files**: check that paths in the YAML file are relative to the repository root or are valid absolute paths.

For API details and additional scenario formats, see the [Real-Twin documentation](https://real-twin.readthedocs.io/en/latest/).
