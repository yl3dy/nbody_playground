from typing import Optional, Tuple, List
import logging
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import G
import scipy.linalg
from numpy.lib.recfunctions import structured_to_unstructured
import astropy.units, astropy.constants
import poliastro.twobody, poliastro.bodies, poliastro.core.angles

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
    body_info_dtype = [(field, np.float64) for field in ('t', 'm', 'x', 'y', 'z', 'vx', 'vy', 'vz')] + [('parent_names', object)]
    body_info_tables = {name: np.empty(len(actual_iter_indices), dtype=body_info_dtype) for name in body_names}

    for i, iter_idx in enumerate(actual_iter_indices):
        body_info = common.read_body_config(run_name, iter_idx)
        assert body_info.names == body_names
        for body_idx, body_name in enumerate(body_names):
            row = body_info_tables[body_name][i]
            row['parent_names'] = body_info.parent_names[body_idx]
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


def plot_elements(run_name : str, save_path : Optional[Path], body_name : str, elements : str):
    parameter_name_map = {'sma': 'a', 'ecc': 'ecc', 'inc': 'inc', 'raan': 'raan', 'arg_pe': 'argp', 'M0': 'nu'}

    body_infos = load_simulation_data(run_name)
    this_body = body_infos[body_name]
    parent_name = this_body['parent_names'][0]
    parent_body = body_infos[parent_name]
    if body_name == parent_name:
        raise ValueError('this is the main body in the system, cannot calculate elements')

    def get_relative_vec(this, parent, prefix=''):
        vec_this = np.array([this[prefix+'x'], this[prefix+'y'], this[prefix+'z']])
        vec_parent = np.array([parent[prefix+'x'], parent[prefix+'y'], parent[prefix+'z']])
        return vec_this - vec_parent

    # Build element sequence
    parent_mass = parent_body['m'][0]
    iteration_count = len(this_body['x'])
    parameter_list = np.empty(iteration_count, dtype=np.float64)
    for idx in range(iteration_count):
        r_rel = get_relative_vec(this_body[idx], parent_body[idx])
        v_rel = get_relative_vec(this_body[idx], parent_body[idx], prefix='v')
        if not np.all(np.isfinite(r_rel)) or not np.all(np.isfinite(v_rel)):
            print('Invalid values!', 'r', r_rel, 'v', v_rel)
        if np.linalg.norm(r_rel) < 1e7:
            print('Distance is too close!', r_rel)

        parent_mass_units = parent_mass*astropy.units.kg
        parent_body_orbital = poliastro.bodies.Body(None, astropy.constants.G*parent_mass_units, parent_name, mass=parent_mass_units*astropy.units.kg)
        orbit = poliastro.twobody.Orbit.from_vectors(attractor=parent_body_orbital, r=r_rel*astropy.units.m, v=v_rel*astropy.units.m/astropy.units.s)
        if elements != 'M0':
            parameter_value = getattr(orbit, parameter_name_map[elements])
            # Fuck units
            parameter_value = float(parameter_value.si / parameter_value.unit)
        else:
            parameter_value = poliastro.core.angles.E_to_M(poliastro.core.angles.nu_to_E(orbit.nu, orbit.ecc), orbit.ecc)
        parameter_list[idx] = parameter_value

    # Actual plotting
    times = _get_times_from_body_infos(body_infos)
    plt.plot(times, parameter_list)
    plt.xlabel('Time')
    plt.ylabel(elements)
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
        'max_diff': abs(energies.min() - energies.max()) / abs(energies[0]),
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


def get_body_momenti(body_infos : dict, body_name : str):
    bi = body_infos[body_name]
    mass = bi['m'][0]   # assume constant mass
    p = mass * structured_to_unstructured(bi[['vx', 'vy', 'vz']])
    return p


def get_rel_momenti(body_momenti : dict, cumulative : bool = False):
    if cumulative:
        integral_momentum = sum(body_momenti.values())

        # Reference momentum: the length of the smallest momentum of all bodies at a given time
        N_idx = body_momenti[list(body_momenti.keys())[0]].shape[0]  # FIXME awful hack to get the shape of a momentum array for some body
        ref_momentum_abs = []
        for idx in range(N_idx):
            ref_momentum_abs.append(
                min(scipy.linalg.norm(body_momenti[body_name][idx, :]) for body_name in body_momenti.keys())
            )

        rel_integral_momentum = scipy.linalg.norm(integral_momentum, axis=1) / ref_momentum_abs
        return rel_integral_momentum
    else:
        return {body_name: scipy.linalg.norm(body_momentum, axis=1) for body_name, body_momentum in body_momenti.items()}


def plot_momentum(run_name : str, save_path : Optional[Path], body_list : Optional[List[str]], is_cumulative : bool) -> None:
    body_infos = load_simulation_data(run_name)

    if not body_list:
        body_list = body_infos.keys()

    body_momenti = {name: get_body_momenti(body_infos, name) for name in body_list}

    if is_cumulative:
        body_times = _get_times_from_body_infos(body_infos)
        cumulative_momenti = get_rel_momenti(body_momenti, cumulative=True)
        plt.plot(body_times, cumulative_momenti, label='Cumulative')
    else:
        noncumulative_momenti = get_rel_momenti(body_momenti, cumulative=False)
        for body_name in body_list:
            body_info = body_infos[body_name]
            plt.plot(body_info['t'], noncumulative_momenti[body_name], label=body_name)

    plt.ylabel('Relative momentum length')
    plt.xlabel('Time, s')
    plt.legend()
    if save_path:
        plt.savefig(str(save_path))
    else:
        plt.show()
