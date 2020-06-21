from typing import List
import argparse
from pathlib import Path
import collections
import logging

from ruamel.yaml import YAML
import numpy as np
import scipy.linalg
from scipy.constants import G
import orbital, orbital.bodies

from nbody_sim import common


def setup_logging(is_debug : bool) -> None:
    loglevel = logging.DEBUG if is_debug else logging.INFO
    logging.basicConfig(level=loglevel)


BarycenterData = collections.namedtuple('BarycenterData', ['m', 'r', 'v', 'children'])
OscBarycenterData = collections.namedtuple('OscBarycenterData', ['m', 'sma', 'ecc', 'inc', 'raan', 'arg_pe', 'M0', 'children'])


def to_arr(iterable):
    return np.array(iterable, dtype=np.float64)


def osc_to_state_barycenter(osc_barycenter, parent_mass):
    """Convert oscullating barycenter to a normal (state-vector) one."""
    if isinstance(osc_barycenter, BarycenterData):
        return osc_barycenter

    parent_body = orbital.bodies.Body(parent_mass, G*parent_mass, 0, 0, 0)
    barycenter_orbit = orbital.KeplerianElements(
        a=osc_barycenter.sma, e=osc_barycenter.ecc, i=osc_barycenter.inc,
        raan=osc_barycenter.raan, arg_pe=osc_barycenter.arg_pe, M0=osc_barycenter.M0,
        body=parent_body
    )

    def orbital_vector_to_arr(vector):
        return np.array([vector.x, vector.y, vector.z], dtype=np.float64)

    return BarycenterData(
        m=osc_barycenter.m,
        r=orbital_vector_to_arr(barycenter_orbit.r),
        v=orbital_vector_to_arr(barycenter_orbit.v),
        children=osc_barycenter.children
    )



def build_body_list(subpart):
    logger = logging.getLogger(__name__)

    if isinstance(subpart, list):
        logger.debug(f'Summing barycenter lists, src: {subpart}')
        return sum((build_body_list(subitem) for subitem in subpart), [])
    else:
        if not subpart['satellites']:
            logger.debug(f'Dealing with a single barycenter, no satellites, src: {subpart}')
            if 'r' in subpart and 'v' in subpart:
                barycenter = BarycenterData(
                    m=float(subpart['m']),
                    r=to_arr(subpart['r']),
                    v=to_arr(subpart['v']),
                    children=[common.SingleBodyConfig(
                        name=subpart['name'],
                        m=float(subpart['m']),
                        r=np.array([0., 0., 0.]),
                        v=np.array([0., 0., 0.]),
                        parent_name=subpart['name']
                    )]
                )
            else:
                barycenter_args = {k: float(subpart[k]) for k in subpart.keys() if k not in ('name', 'satellites')}
                barycenter_args['children'] = [
                    common.SingleBodyConfig(
                        name=subpart['name'],
                        m=float(subpart['m']),
                        r=np.array([0., 0., 0.]),
                        v=np.array([0., 0., 0.]),
                        parent_name=subpart['name']
                    )
                ]
                barycenter = OscBarycenterData(**barycenter_args)
            logger.debug(f'Processed barycenter, got this: {barycenter}')
            return [barycenter]
        else:
            logger.debug(f'Dealing with a single barycenter, has satellites, src: {subpart}')
            barycenters = build_body_list(subpart['satellites'])
            logger.debug(f'Barycenters: {barycenters}')
            N_sats = len(barycenters)
            my_mass = float(subpart['m'])
            my_name = subpart['name']


            # Convert other barycenters to state-vectors
            barycenters = [osc_to_state_barycenter(bary, my_mass) for bary in barycenters]
            logger.debug(f'Got the following barycenters after conversion: {barycenters}')

            # Find radii w.r.t. current barycenter
            logger.debug('Solving a linear system for radii')
            A_radii = np.eye(N_sats+1)
            A_radii[:, 0] = -1
            A_radii[0, 0] = my_mass
            A_radii[0, 1:] = [b.m for b in barycenters]
            logger.debug(f'A = {A_radii}')
            b_radii = np.zeros([N_sats+1, 3], dtype=np.float64)
            b_radii[1:, :] = np.array([b.r for b in barycenters], dtype=np.float64)
            logger.debug(f'b = {b_radii}')
            r_barys = scipy.linalg.solve(A_radii, b_radii)
            logger.debug(f'Solutions: {r_barys}')

            # Find velocities w.r.t. current barycenter
            logger.debug('Solving a linear system for velocities')
            A_vels = np.eye(N_sats+1)
            A_vels[:, 0] = -1
            A_vels[0, 0] = my_mass
            A_vels[0, 1:] = [b.m for b in barycenters]
            logger.debug(f'A = {A_vels}')
            b_vels = np.zeros([N_sats+1, 3], dtype=np.float64)
            b_vels[1:, :] = np.array([b.v for b in barycenters], dtype=np.float64)
            logger.debug(f'b = {b_vels}')
            v_barys = scipy.linalg.solve(A_vels, b_vels)
            logger.debug(f'Solutions: {v_barys}')

            # Start building a list of "children" of this barycenter
            new_children_cfg = [common.SingleBodyConfig(
                name=my_name, m=my_mass,
                r=r_barys[0, :], v=v_barys[0, :],
                parent_name=my_name
            )]
            # Apply r,v updates to each of "satellite" barycenter children
            for satellite_idx in range(N_sats):
                r_fix = r_barys[satellite_idx+1, :]
                v_fix = v_barys[satellite_idx+1, :]
                satellite_bary_cfg = barycenters[satellite_idx]
                for child in satellite_bary_cfg.children:
                    new_children_cfg += [common.SingleBodyConfig(
                        name=child.name, m=child.m,
                        r=child.r + r_fix,
                        v=child.v + v_fix,
                        parent_name=my_name if child.parent_name == child.name else child.parent_name
                    )]

            # Build current barycenter data object
            current_bary_m = sum(bc.m for bc in barycenters) + my_mass
            if 'r' in subpart and 'v' in subpart:
                # State-vector definition
                current_bary_r = to_arr(subpart['r'])
                current_bary_v = to_arr(subpart['v'])
                current_barycenter = BarycenterData(m=current_bary_m, r=current_bary_r, v=current_bary_v, children=new_children_cfg)
            else:
                # Oscullating element definition
                barycenter_args = {k: float(subpart[k]) for k in ('sma', 'ecc', 'inc', 'raan', 'arg_pe', 'M0')}
                barycenter_args['m'] = current_bary_m
                barycenter_args['children'] = new_children_cfg
                current_barycenter = OscBarycenterData(**barycenter_args)
            logger.debug(f'Processed barycenter, got this: {current_barycenter}')

            return [current_barycenter]


def generate_system_state(body_config):
    logger = logging.getLogger(__name__)

    full_system_barycenter = build_body_list(body_config)
    logger.debug(f'Got the following full barycenter: {full_system_barycenter}')

    # FIXME: correct behaviour (coordinate/velocity transform) when barycenter(s) are not at (0, 0, 0)
    bodies = full_system_barycenter[0].children
    return common.SystemState(
        names=[b.name for b in bodies],
        m=np.array([b.m for b in bodies], dtype=np.float64),
        r=np.array([b.r for b in bodies], dtype=np.float64),
        v=np.array([b.v for b in bodies], dtype=np.float64),
        parent_names=[b.parent_name for b in bodies]
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_config', type=Path)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    setup_logging(args.debug)
    run_name = args.input_config.stem

    yaml= YAML(typ='safe')
    yaml_cfg = yaml.load(args.input_config)

    global_cfg = common.GlobalConfig(**yaml_cfg['global_config'])
    system_state = generate_system_state(yaml_cfg['body_config'])

    common.write_global_config(run_name, global_cfg)
    common.write_body_config(run_name, system_state, 0)
    run_dir = Path(run_name)
    if not run_dir.exists():
        run_dir.mkdir()
    else:
        for p in run_dir.glob('*.csv'):
            p.unlink()


if __name__ == '__main__':
    main()
