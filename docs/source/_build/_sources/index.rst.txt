
Welcome to realtwin documentation!
==================================

.. only:: html

   .. image:: https://badge.fury.io/py/realtwin.svg
      :target: https://badge.fury.io/py/realtwin

   .. image:: https://static.pepy.tech/badge/realtwin
      :target: https://pepy.tech/project/realtwin

   .. image:: https://img.shields.io/pypi/wheel/realtwin.svg
      :target: https://pypi.python.org/pypi/realtwin

   .. image:: https://img.shields.io/pypi/pyversions/realtwin.svg
      :target: https://www.python.org/

   .. image:: https://readthedocs.org/projects/Real-Twin/badge/?version=latest
      :target: https://real-twin.readthedocs.io/en/latest/?badge=latest

   .. image:: https://img.shields.io/github/contributors/ORNL-Real-Sim/real-twin.svg
      :target: https://img.shields.io/github/contributors/ORNL-Real-Sim/Real-Twin

   .. image:: https://img.shields.io/badge/License-GPL-blue.svg
      :target: https://www.gnu.org/licenses/gpl-3.0.en.html

Real-Twin provides scenario generation and calibration through
``RealTwinSUMO``, ``RealTwinVISSIM``, and ``RealTwinAimsun``. The current
workflows support traffic demand with optional external signal control and
independent turn/inflow and behavior calibration using genetic algorithm (GA),
simulated annealing (SA), tabu search (TS), or Bayesian optimization (BO).

.. important::

   The current Real-Twin package supports Windows only.

Start with :doc:`pages/install`, configure your data using
:doc:`pages/realtwin_prepare`, and follow :doc:`pages/realtwin_generation`.
The SUMO vehicle-type workflow is described in
:doc:`pages/autonomous_vehicles_simulation`.


.. toctree::
   :maxdepth: 3
   :caption: Real-Twin Navigation

   pages/intro.rst

.. toctree::
   :maxdepth: 3

   pages/install.rst

.. toctree::
   :maxdepth: 3

   pages/realtwin_simulator.rst

.. toctree::
   :maxdepth: 3

   pages/realtwin_prepare.rst

.. toctree::
   :maxdepth: 3

   pages/realtwin_generation.rst

.. toctree::
   :maxdepth: 3

   pages/autonomous_vehicles_simulation.rst

.. toctree::
   :maxdepth: 2

   pages/api.rst

.. toctree::
   :maxdepth: 3

   pages/support.rst
   pages/acknowledgements.rst

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
