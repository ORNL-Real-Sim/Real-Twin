============
Installation
============

You can install the latest stable release of the package at `PyPI`_ using `pip`_:

.. code-block:: bash

    pip install realtwin

By running the command above, the realtwin package along with required dependency packages (`PyYAML`_, `Shapely`_, `traci`_, `Requests`_, `pyufunc`_, `mealpy`_, `matplotlib`_, `networkx`_) will be installed to your running environment (if they have not been installed yet).

Potential Issues
=================

- Shapely

If you install realtwin in a conda environment, you may get an error message: "OSError: [WinError 126]
The specified module could not be found" when importing realtwin. To resolve this issue, you need to uninstall
the `Shapely`_ package first, and reinstall it manually using the command below.

.. code-block:: bash

    conda install shapely


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
