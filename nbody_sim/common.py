from pathlib import Path
import collections
import csv
import numpy as np


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
        p = Path(run_name) / Path(f'{run_name:09d}.bodies')

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


def write_body_config(state : SystemState, iter_num : int) -> None:
    assert iter_num >= 1
    raise NotImplementedError
