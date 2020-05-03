from typing import Optional, Tuple, List
import logging
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import G
import scipy.linalg
from numpy.lib.recfunctions import structured_to_unstructured

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


def _get_times_from_body_infos(body_infos):
    return next(iter(body_infos.values()))['t']


def plot_positions(run_name : str, save_path : Optional[Path], axes : str, body_list : Optional[List[str]], relative_to : str) -> None:
    body_infos = load_simulation_data(run_name)

    if not body_list:
        body_list = body_infos.keys()

    if relative_to:
        reference = {coord: body_infos[relative_to][coord] for coord in ('x', 'y', 'z')}
    else:
        reference = {'x': 0., 'y': 0., 'z': 0.}

    axis_1, axis_2 = axes
    times = _get_times_from_body_infos(body_infos)
    for body_name in body_list:
        body_info = body_infos[body_name]
        plt.scatter(body_info[axis_1] - reference[axis_1], body_info[axis_2] - reference[axis_2], c=times, label=body_name, s=2, alpha=0.5, cmap='inferno')
    plt.axvline(0, linestyle='--', color='gray', alpha=0.5)
    plt.axhline(0, linestyle='--', color='gray', alpha=0.5)
    plt.xlabel(f'{axis_1}, m')
    plt.ylabel(f'{axis_2}, m')
    plt.legend()
    if save_path:
        plt.savefig(str(save_path))
    else:
        plt.show()


def get_body_energies(body_infos, body_name):
    bi = body_infos[body_name]
    K = 0.5 * bi['m'] * (bi['vx']**2 + bi['vy']**2 + bi['vz']**2)

    W = 0
    r_mine = structured_to_unstructured(bi[['x', 'y', 'z']])
    for other_body_name in body_infos.keys():
        if other_body_name == body_name:
            continue
        body_other = body_infos[other_body_name]
        r_other = structured_to_unstructured(body_other[['x', 'y', 'z']])
        W += body_other['m'] / scipy.linalg.norm(r_other - r_mine, axis=1)
    W *= -G*bi['m']

    return K + W

def _get_energy_stats(energies):
    return {
        'max_diff': np.abs(energies - energies[0]).max() / abs(energies[0]),
        'rel_stdev': energies.std() / abs(energies[0]),
        'avg': energies.mean()
    }


def plot_energy(run_name : str, save_path : Optional[Path], body_list : Optional[List[str]], is_cumulative : bool) -> None:
    body_infos = load_simulation_data(run_name)

    if not body_list:
        body_list = body_infos.keys()

    body_energies = {name: get_body_energies(body_infos, name) for name in body_list}

    if is_cumulative:
        body_times = _get_times_from_body_infos(body_infos)
        cumulative_energy = sum(body_energies.values())
        plt.plot(body_times, cumulative_energy, label='Cumulative')

        print('--- Cumulative energy stats ---')
        print('avg {avg:.5e}, relative stdev {rel_stdev:.3}, max diff {max_diff:.3}'.format(**_get_energy_stats(cumulative_energy)))
    else:
        print('--- Per body energy stats ---')
        for body_name in body_list:
            body_info = body_infos[body_name]
            plt.plot(body_info['t'], body_energies[body_name], label=body_name)
            print('{body_name}: avg {avg:.5e}, relative stdev {rel_stdev:.3}, max diff {max_diff:.3}'\
                  .format(body_name=body_name, **_get_energy_stats(body_energies[body_name])))

    plt.ylabel('Energy, J')
    plt.xlabel('Time, s')
    plt.legend()
    if save_path:
        plt.savefig(str(save_path))
    else:
        plt.show()



def plot_momentum(args):
    raise NotImplementedError
