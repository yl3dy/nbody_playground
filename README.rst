=================
n-body playground
=================

Requires Python 3.5+.


Components
----------

* Initial body config generator (from oscullating elements or state vectors);
* Main simulation;
* Analysis toolkit.


Simulation engines and methods
------------------------------

* ``dummy``: for testing only
    * ``dummy_method``
* ``naive``: straightforward Python implementation without massive Numpy/Scipy usage
    * ``semi_explicit_euler``
    * ``explicit_rk2``
    * ``explicit_rk4``
    * ``ralston``
    * ``bogacki_shampine`` (TODO)
    * ``dormand_prince`` (TODO)
    * ``explicit_rk16`` (TODO)
    * ``crank_nicolson`` (TODO)
    * ``adams_bashforth_2``
    * ``adams_bashforth_5``
    * ``velocity_verlet``
    * ``ruth3``
    * ``ruth4``
* ``scipy``: uses ``scipy.integrate.solve_ivp``
* ``numpy`` (TODO): massive Numpy/Scipy usage
* ``cython`` (TODO): cythonized integration loop
* ``numba`` (TODO)
* ``cpp`` (TODO): main loop in C++


Usage
-----

Simulation parameters are set in a single YAML config file. See ``sample_configs/`` for examples and description (TODO). Simulation name is the stem (filename without extension) of the config file, it is used in the intermediate file name generation.

To run simulation:

#. Create YAML config (e.g. ``simulation.yaml``)
#. Run ``prepare_config.py``, which will generate ``simulation.global`` and ``simulation.bodies``.
#. Run ``nbody.py`` (using *simulation name* as a parameter, here ``simulation``), which will output to a directory ``simulation/``.
#. Do some analysis using ``plot.py`` (using simulation name as a parameter).


Issues and roadmap
------------------

* Gather accuracy info: build energy error graphs for ``sample_configs/oscullating.yaml`` for different dt and methods.
* Gather performance info: create a table with per-iteration speeds for each method.
* Add plotting of effective oscullating elements for each body
* Upgrade position plot to use oscullating element "smoothing" to produce nice plots when T is large but ``output_point_num`` is low.
* More methods for ``naive`` engine.
* Create a list of useful methods for other engines.
* How ``numpy`` can be implemented?
* Implement ``cython`` engine.
* Better documentation
* Tests
