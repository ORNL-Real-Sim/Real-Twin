Installation
============

The current Real-Twin package supports Windows only and requires Python 3.10
or newer. Install it in the Windows Python environment that will run your
scenario scripts.

Published package
-----------------

.. code-block:: text

    python -m pip install realtwin

This installs the package's required Python dependencies. Simulator executables
and commercial simulator licenses must be prepared separately; see
:doc:`realtwin_simulator`.

Repository tutorials
--------------------

The tutorial scripts and datasets are available in the repository checkout.
From the repository root, create and activate an environment:

.. code-block:: text

    python -m venv .venv

.. code-block:: text

    # Windows PowerShell
    .venv\Scripts\Activate.ps1

    # Windows Command Prompt
    .venv\Scripts\activate.bat

Then install the checkout and its dependencies:

.. code-block:: text

    python -m pip install --upgrade pip
    python -m pip install -e .

An editable installation makes the current checkout available to the tutorial
scripts and their subprocesses. Activate an existing conda environment instead
if you already maintain one for Real-Twin.

Optional calibration and simulator dependencies
-----------------------------------------------

The current SUMO, Vissim, and Aimsun scenario scripts select Bayesian
optimization (BO). Install its Gaussian-process dependencies in the same
environment:

.. code-block:: text

    python -m pip install -e ".[bo]"

For a published release containing BO, use:

.. code-block:: text

    python -m pip install "realtwin[bo]"

Genetic algorithm (GA), simulated annealing (SA), and tabu search (TS) do not
require the BO extra. Vissim's native Windows integration
also requires:

.. code-block:: text

    python -m pip install pywin32

For the interactive SUMO notebook:

.. code-block:: text

    python -m pip install notebook
    python -m jupyter notebook tutorials/tutorial_realtwin_SUMO.ipynb

Aimsun uses its own compatible Python runtime. Set
``AIMSUN.site_packages`` to packages installed for that runtime, and set
``AIMSUN.aimsun_exe_path`` to the licensed console executable. See
:doc:`realtwin_prepare` before running the Aimsun example.

Build the documentation
-----------------------

From the repository root, install the documentation dependencies and build HTML:

.. code-block:: text

    python -m pip install -r docs/source/requirements_dev.txt
    python -m sphinx -b html -W --keep-going docs/source docs/_build/html

Open ``docs/_build/html/index.html`` to review the output. Build artifacts belong
under ``docs/_build/``, outside the source pages.

Common installation problems
----------------------------

* **Import errors:** activate the intended environment and install the checkout
  there. Use ``python -m pip`` so installation targets the selected interpreter.
* **BO dependency errors:** install the ``bo`` extra or select GA, SA, or TS.
* **SUMO tools are missing:** make ``sumo``, ``netconvert``, and ``jtrrouter``
  available on ``PATH``; calibration also expects ``SUMO_HOME``.
* **Vissim COM errors:** check Windows, ``pywin32``, and your Vissim COM license.
* **Aimsun import/version errors:** match ``site_packages`` to the Python version
  used by the installed Aimsun release.
* **Shapely DLL errors in a conda environment:** check the Shapely installation
  in that environment. A conda-provided installation can be installed with
  ``conda install shapely``.

Package releases are available on `PyPI <https://pypi.org/project/realtwin/>`_,
and source examples are in the
`Real-Twin repository <https://github.com/ORNL-Real-Sim/Real-Twin>`_.
