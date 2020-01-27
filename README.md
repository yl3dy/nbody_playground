# n-body playground

Requires Python 3.5+.

## Components

* Initial body config generator (from oscullating elements);
* Main simulation;
* Analysis toolkit.

## Simulation engines and methods

* `dummy`: for testing only
    * `dummy_method`
* `naive`: straightforward Python implementation without massive Numpy/Scipy usage
    * `semi_explicit_euler`
    * `explicit_rk2`
    * `explicit_rk4`
    * `ralston`
    * `bogacki_shampine`
    * `dormand_prince`
    * `explicit_rk16`
    * `crank_nicolson`
    * `adams_bashforth_2`
    * `adams_bashforth_5`
    * `velocity_verlet`
    * `ruth_3`
    * `ruth_4`
* `numpy` (TODO): massive Numpy/Scipy usage
* `cython` (TODO): cythonized integration loop
* `numba` (TODO)
* `cpp` (TODO): main loop on C++


## CLI

Simulation dataset is specified with a prefix. Global config filename is `{prefix}.global`, body config is `{prefix}.bodies`, output dir is `{prefix}/`.

## Data/config files

Below CSV means CSV with whitespace delimiter.

Global config file (one-entry CSV):

```
dt iter_num output_point_num engine method
{one line with global parameters}
```

Body config file (CSV) should be used for any time step. Custom name for initial step, for others `{output_dir}/{iteration_number:09d}.bodies`. It should be possible to reuse a snapshot at every step to restart simulation.

```
name x y z vx vy vz
{a line for each body}
```

Body name must not consist of whitespace or non-ASCII symbols.
