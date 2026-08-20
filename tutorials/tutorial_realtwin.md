# Real-Twin Tutorials

This page is the starting point for running the examples in this repository. It explains what each tutorial does, which files are used, what must be prepared manually, and the order in which the Real-Twin methods are called.

## What is included?

| Tutorial               | Simulator   | Entry point                                                                                                                  | Purpose                                                                      |
| ---------------------- | ----------- | ---------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| Core scenario workflow | SUMO        | [`tutorial_realtwin_SUMO.py`](tutorial_realtwin_SUMO.py)                                                                    | Build, calibrate, and prepare a traffic scenario from the example data.      |
| SUMO workflow          | SUMO        | [`tutorial_realtwin_SUMO.py`](tutorial_realtwin_SUMO.py) or [`tutorial_realtwin_SUMO.ipynb`](tutorial_realtwin_SUMO.ipynb) | Run the SUMO-specific workflow interactively or as a Python script.          |
| Aimsun workflow        | Aimsun Next | [`tutorial_realtwin_Aimsun.py`](tutorial_realtwin_Aimsun.py)                                                                | Import an Aimsun model, create its matchup table, and generate the scenario. |
| Autonomous vehicles    | SUMO        | [`realtwin_av_config.yaml`](realtwin_av_config.yaml) plus `SimAV`                                                         | Add Human, AV, CAV, or other vehicle types to a SUMO simulation.             |

The first three tutorials use the same broad scenario workflow:

1. Load a YAML configuration.
2. Check the simulator environment.
3. Generate or import the network and create `MatchupTable.xlsx`.
4. Add traffic demand, signal-control data, and matchup-table mappings.
5. Generate the abstract and concrete scenarios.
6. Prepare the simulator files.
7. Optionally calibrate, post-process, or visualize the results.

The AV tutorial is a separate, shorter SUMO workflow. It does not use `RealTwin.generate_abstract_scenario()` or the matchup table.

## Prerequisites

- Python 3.10 or newer.
- A clone of this repository, or an installed Real-Twin package.
- The Python dependencies in [`requirements.txt`](../requirements.txt).
- SUMO for the SUMO tutorials. Real-Twin can check for SUMO and, when requested, install a SUMO version.
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

# macOS or Linux
source .venv/bin/activate
```

Install Real-Twin and its dependencies:

```text
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

You can also install the published package with `python -m pip install realtwin`, but the tutorial scripts and example datasets are available in the repository checkout.

For the optional SUMO notebook, install Jupyter as well:

```text
python -m pip install notebook
```

## Run the core SUMO workflow

The general example uses [`realtwin_config.yaml`](realtwin_config.yaml), which points to `datasets/example2`. Run it from the repository root:

```text
python tutorials/tutorial_realtwin_SUMO.py
```

The script performs these operations:

```python
import realtwin as rt

twin = rt.RealTwin(
    input_config_file="./tutorials/realtwin_config.yaml",
    verbose=True,
)
twin.env_setup()  # Defaults to SUMO.
twin.generate_inputs(incl_sumo_net="./datasets/example2/updated.net.xml")
twin.generate_abstract_scenario()
twin.generate_concrete_scenario()
twin.prepare_simulation()
twin.calibrate(sel_algo={"turn_inflow": "GA", "behavior": "GA"})
```

The example script also calls `post_process()` and `visualize()`. Those stages are still marked as ongoing in the example and may require additional result-specific options.

### What to do when the script pauses

`generate_inputs()` creates or refreshes the input folders and generates `datasets/example2/MatchupTable.xlsx`. Before continuing:

1. Put traffic-demand turn-movement files in `datasets/example2/Traffic/`.
2. Put the signal-control file, such as a Synchro UTDF file, in `datasets/example2/Control/`.
3. Open `datasets/example2/MatchupTable.xlsx` and fill in the demand and control filenames and the required mapping values.
4. Save the table, then press Enter in the terminal.
5. When asked again after abstract-scenario generation, check that the demand and signal movements match the network bearings, then continue.

The configuration file contains the network name, vertices, input/output directories, simulation time window, calibration targets, and algorithm settings. Change those values for a different dataset. `NetworkName` should not contain spaces.

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

After `prepare_simulation()`, SUMO preparation files are written below the configured output directory. Calibration supports `GA` (genetic algorithm), `SA` (simulated annealing), and `TS` (tabu search), subject to the settings in the YAML file.

## Aimsun tutorial

Aimsun uses a separate class and configuration because it imports and drives an Aimsun model. Start with:

```text
python tutorials/tutorial_realtwin_Aimsun.py
```

Before running it, edit the `AIMSUN` section of [`realtwin_config.yaml`](realtwin_config.yaml):

- `model_fname`: the absolute path to the `.ang` Aimsun model.
- `model_xdor`: the path to the model's OpenDRIVE file, when required by the model.
- `site_packages`: the site-packages directory for the Python version used by Aimsun.

The Aimsun model and OpenDRIVE paths in the checked-in configuration are machine-specific examples. They must be replaced on another computer. Aimsun Next 23, for example, uses Python 3.10; install packages such as `numpy`, `pandas`, `openpyxl`, and `xlrd` into the site-packages directory used by that Aimsun installation.

The Aimsun script runs this workflow:

```python
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

`generate_inputs()` opens the configured Aimsun model, creates `Control/` and `Traffic/` folders beside the model, and generates an Aimsun matchup table. Add the demand and signal files to those folders and complete the matchup table before continuing. The Aimsun integration may pause for confirmation at the same points as the SUMO workflow.

The current Aimsun example prepares the scenario files. Calibration, post-processing, and visualization calls are commented out in the script and should be treated as follow-up work rather than required steps for this tutorial.

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
- **The script stops for input**: this is expected. Complete the `Control/`, `Traffic/`, and `MatchupTable.xlsx` preparation described above.
- **Aimsun cannot be found**: verify that `aconsole.exe` is installed, licensed, and discoverable, or pass its directory through `sel_dir`.
- **Aimsun Python import errors**: use the Python version expected by the Aimsun release and point `site_packages` to that environment.
- **Missing network or input files**: check that paths in the YAML file are relative to the repository root or are valid absolute paths.

For API details and additional scenario formats, see the [Real-Twin documentation](https://real-twin.readthedocs.io/en/latest/).
