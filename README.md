# n-body playground

## Components

* Initial body config generator (from oscullating elements);
* Main simulation;
* Analysis toolkit.

## CLI

Simulation dataset is specified with a prefix. Global config filename is `{prefix}.global`, body config is `{prefix}.bodies`, output dir is `{prefix}/`.

## Data/config files

Below CSV means CSV with whitespace delimiter.

Global config file (one-entry CSV):

```
dt iternum output_point_num engine method
{one line with global parameters}
```

Body config file (CSV) should be used for any time step. Custom name for initial step, for others `{output_dir}/{iteration_number:09d}.bodies`. It should be possible to reuse a snapshot at every step to restart simulation.

```
name x y z vx vy vz
{a line for each body}
```

Body name must not consist of whitespace or non-ASCII symbols.
