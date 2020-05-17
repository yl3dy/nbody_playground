"""
Integration using Scipy-provided tool

ODEs in the system go as follows: first all coordinate (x, y, z) equations in the order of body_config, then all velocity (vx, vy, vz) ones.

"""
import time
from math import sqrt
import numpy as np
import scipy.integrate
from scipy.constants import G

from ..common import SystemState, GlobalConfig
from .. import common


METHODS = ('RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA')


def initial_condition(body_cfg : SystemState):
    y0 = []
    body_num = len(body_cfg.names)

    # First, set coordinates
    for body_idx in range(body_num):
        y0.extend(list(body_cfg.r[body_idx]))
    # Set velocities
    for body_idx in range(body_num):
        y0.extend(list(body_cfg.v[body_idx]))

    return y0


def get_r_from_y(y, body_idx, body_num):
    assert body_idx < body_num
    start_idx = body_idx*3
    stop_idx = (body_idx+1)*3
    return [y[i] for i in range(start_idx, stop_idx)]


def get_v_from_y(y, body_idx, body_num):
    assert body_idx < body_num
    start_idx = body_num*3 + body_idx*3
    stop_idx = body_num*3 + (body_idx+1)*3
    return [y[i] for i in range(start_idx, stop_idx)]


def norm3(r):
    return sqrt(r[0]**2 + r[1]**2 + r[2]**2)**3


def f(t, y, m):
    f_val = []
    body_num = len(m)

    # Coordinate ODEs
    for body_idx in range(body_num):
        f_val.extend(get_v_from_y(y, body_idx, body_num))

    # Velocity ODEs
    for body_idx in range(body_num):
        my_r = np.array(get_r_from_y(y, body_idx, body_num))
        accel = np.zeros(3, dtype=np.float64)
        for other_body_idx in range(body_num):
            if other_body_idx == body_idx:
                continue
            other_r = np.array(get_r_from_y(y, other_body_idx, body_num))
            accel += m[other_body_idx] * (other_r - my_r) / norm3(other_r - my_r)
        f_val.extend(G*accel)

    return f_val


def get_state(integration_result, initial_body_cfg, iter_idx : int):
    body_num = len(initial_body_cfg.names)

    y_t = integration_result.y[:, iter_idx - 1]
    r, v = [], []
    for body_idx in range(body_num):
        r.append(get_r_from_y(y_t, body_idx, body_num))
        v.append(get_v_from_y(y_t, body_idx, body_num))
    r, v = np.array(r), np.array(v)

    return SystemState(names=initial_body_cfg.names, m=initial_body_cfg.m, r=r, v=v)


def simulate(run_name : str, global_config : GlobalConfig, body_config : SystemState) -> None:
    t_span = (0, global_config.dt*global_config.iter_num)
    t_eval = np.linspace(global_config.dt, global_config.dt*global_config.iter_num, num=global_config.iter_num)
    y0 = initial_condition(body_config)

    assert global_config.method in METHODS

    t0 = time.time()
    result = scipy.integrate.solve_ivp(f, t_span, y0, method=global_config.method, t_eval=t_eval, args=(body_config.m,), first_step=global_config.dt)
    # result = scipy.integrate.solve_ivp(f, t_span, y0, method=global_config.method, t_eval=t_eval, args=(body_config.m,))
    print('Integration time: {:.3} s'.format(time.time() - t0))
    print(result)
    assert result.success

    # Write down output
    for do_write, iter_idx in common.get_iter_indices(global_config.iter_num, global_config.output_point_num):
        if do_write:
            state = get_state(result, body_config, iter_idx)
            common.write_body_config(run_name, state, iter_idx)
