from typing import Optional, Tuple, List
import logging
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas

from . import common


def load_simulation_data(run_name : str):
    global_config = common.read_global_config(run_name)
    actual_iter_indices = [
        iter_idx for do_write, iter_idx
        in common.get_iter_indices(global_config.iter_num, global_config.output_point_num)
        if do_write
    ]
    body_names = common.read_body_config(run_name).names

    # Preallocate body info tables
    body_info_dtype = [(field, np.float64) for field in ('t', 'm', 'x', 'y', 'z', 'vx', 'vy', 'vz')]
    body_info_tables = {name: np.empty(len(actual_iter_indices), dtype=body_info_dtype) for name in body_names}

    for i, iter_idx in enumerate(actual_iter_indices):
        body_info = common.read_body_config(run_name, iter_idx)
        assert body_info.names == body_names
        for body_idx, body_name in enumerate(body_names):
            row = body_info_tables[body_name][i]
            row['t'] = iter_idx * global_config.dt
            row['m'] = body_info.m[body_idx]
            for field_idx, field_name in enumerate(['x', 'y', 'z']):
                row[field_name] = body_info.r[body_idx, field_idx]
            for field_idx, field_name in enumerate(['vx', 'vy', 'vz']):
                row[field_name] = body_info.v[body_idx, field_idx]

    return body_info_tables


def plot_positions(run_name : str, save_path : Optional[Path], body_list : Optional[List[str]]) -> None:
    body_infos = load_simulation_data(run_name)

    if not body_list:
        body_list = body_infos.keys()

    for body_name in body_list:
        body_info = body_infos[body_name]
        plt.plot(body_info['x'], body_info['y'], label=body_name)
    plt.xlabel('x, m')
    plt.ylabel('y, m')
    plt.legend()
    if save_path:
        plt.savefig(str(save_path))
    else:
        plt.show()


def get_body_energies(body_info):
    return 0.5 * body_info['m'] * (body_info['vx']**2 + body_info['vy']**2 + body_info['vz']**2)


def plot_energy(run_name : str, save_path : Optional[Path], body_list : Optional[List[str]], is_cumulative : bool) -> None:
    body_infos = load_simulation_data(run_name)

    if not body_list:
        body_list = body_infos.keys()

    body_energies = {name: get_body_energies(body_infos[name]) for name in body_list}

    if is_cumulative:
        body_times = next(iter(body_infos.values()))['t']
        cumulative_energy = sum(body_energies.values())
        plt.plot(body_times, cumulative_energy, label='Cumulative')
    else:
        for body_name in body_list:
            body_info = body_infos[body_name]
            plt.plot(body_info['t'], body_energies[body_name], label=body_name)

    plt.ylabel('Energy, J')
    plt.xlabel('Time, s')
    plt.legend()
    if save_path:
        plt.savefig(str(save_path))
    else:
        plt.show()



def plot_momentum(args):
    raise NotImplementedError