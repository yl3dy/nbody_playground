from pathlib import Path
import collections
import csv
import numpy as np
import progressbar


GlobalConfig = collections.namedtuple('GlobalConfig', ['dt', 'iter_num', 'output_point_num', 'engine', 'method'])
SingleBodyConfig = collections.namedtuple('SingleBodyConfig', ['name', 'r', 'v'])
SystemState = collections.namedtuple('SystemState', ['names', 'r', 'v'])


def read_global_config(run_name : str) -> GlobalConfig:
    """Return global config."""
    p = Path(run_name + '.global')
    with open(p, newline='') as f:
        reader = csv.DictReader(f, delimiter=' ')
        cfg_dict = next(reader)
        cfg = GlobalConfig(
            dt=float(cfg_dict['dt']),
            iter_num=int(cfg_dict['iter_num']),
            output_point_num=int(cfg_dict['output_point_num']),
            engine=cfg_dict['engine'],
            method=cfg_dict['method']
        )
    return cfg


def get_body_config_path(run_name : str, iter_num : int = 0) -> Path:
    assert iter_num >= 0
    if iter_num == 0:
        p = Path(run_name + '.bodies')
    else:
        p = Path(run_name) / Path(f'{iter_num:09d}.bodies')

    return p


def read_body_config(run_name : str, iter_num : int = 0) -> SystemState:
    """Return body config path."""
    config_path = get_body_config_path(run_name, iter_num)

    bodies = []
    with open(config_path, newline='') as f:
        reader = csv.DictReader(f, delimiter=' ')
        for row in reader:
            bodies.append(SingleBodyConfig(
                name=row['name'],
                r=(float(row['x']), float(row['y']), float(row['z'])),
                v=(float(row['vx']), float(row['vy']), float(row['vz']))
            ))

    system_state = SystemState(
        names=[b.name for b in bodies],
        r=np.array([b.r for b in bodies], dtype=np.float64),
        v=np.array([b.r for b in bodies], dtype=np.float64)
    )
    return system_state


def write_body_config(run_name : str, state : SystemState, iter_num : int) -> None:
    assert iter_num >= 1

    fieldnames = ['name', 'x', 'y', 'z', 'vx', 'vy', 'vz']
    body_count = len(state.names)
    output_path = get_body_config_path(run_name, iter_num)

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=' ')
        writer.writeheader()
        for i in range(body_count):
            writer.writerow({
                'name': state.names[i],
                'x': state.r[i, 0], 'y': state.r[i, 1], 'z': state.r[i, 2],
                'vx': state.v[i, 0], 'vy': state.v[i, 1], 'vz': state.v[i, 2],
            })


def get_iter_indices(iter_num : int, output_num : int):
    """Generator to get iteration index and write flag."""
    assert iter_num > output_num
    assert iter_num % output_num == 0

    for iter_idx in range(1, iter_num+1):
        do_write = (iter_idx % output_num) == 0
        yield do_write, iter_idx


def get_my_progressbar(max_value):
    """Get customized progressbar object."""
    bar_widgets = [
        progressbar.Percentage(), ' ',
        progressbar.Bar(), ' ',
        progressbar.Timer(format='Elapsed: %(elapsed)s'), ' ',
        progressbar.AdaptiveETA()
    ]
    return progressbar.ProgressBar(max_value=max_value, widgets=bar_widgets)

