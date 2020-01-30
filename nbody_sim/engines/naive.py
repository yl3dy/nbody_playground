import collections
from abc import ABC, abstractmethod
import math
import numpy as np
from scipy.constants import G
import scipy.linalg

from ..common import SystemState, GlobalConfig
from .. import common


def _norm(r):
    """Custom Euler norm calculator for 3-vectors

    Seems slightly faster than `scipy.linalg.norm`.

    """
    return math.sqrt(r[0]**2 + r[1]**2 + r[2]**2)


class BaseIntegrator(ABC):
    """Base integrator class for timestep-constant methods."""
    NUM_PREVIOUS_STATES = None

    def __init__(self, global_config : GlobalConfig, initial_system_state : SystemState):
        self._global_cfg = global_config
        self._m = initial_system_state.m.copy()
        self._names = initial_system_state.names[:]
        self._previous_r = collections.deque([initial_system_state.r.copy()], self.NUM_PREVIOUS_STATES)
        self._previous_v = collections.deque([initial_system_state.v.copy()], self.NUM_PREVIOUS_STATES)
        self._current_r = initial_system_state.r.copy()
        self._current_v = initial_system_state.v.copy()

        assert len(self._names) == self._m.shape[0] == self._current_r.shape[0] == self._current_v.shape[0]

    @property
    def dt(self):
        return self._global_cfg.dt

    def _starting_method(self):
        pass

    @abstractmethod
    def _main_method(self):
        pass

    def _grav_accel(self, r, accel=None):
        """Calculate gravitational acceleration."""
        N_bodies = len(self._names)
        if accel is None:
            accel = np.empty_like(r)

        for i in range(N_bodies):
            # Calculation of force for i-th body
            r_current = r[i, :]
            accel_accumulator = 0
            for j in range(N_bodies):
                if i != j:
                    #accel_accumulator += self._m[j] * (r[j, :] - r_current) / scipy.linalg.norm(r[j, :] - r_current)**3
                    accel_accumulator += self._m[j] * (r[j, :] - r_current) / _norm(r[j, :] - r_current)**3
            accel[i, :] = G * accel_accumulator
        return accel

    def advance(self):
        """Perform next iteration."""
        if len(self._previous_r) < self._previous_r.maxlen:
            self._starting_method()
        else:
            self._main_method()
        self._previous_r.appendleft(self._current_r.copy())
        self._previous_v.appendleft(self._current_v.copy())

    @property
    def state(self):
        """Current system state."""
        return SystemState(names=self._names, m=self._m, r=self._current_r, v=self._current_v)


class SemiExplicitEulerIntegrator(BaseIntegrator):
    NUM_PREVIOUS_STATES = 1
    def _main_method(self):
        self._current_v = self._previous_v[0] + self.dt*self._grav_accel(self._previous_r[0])
        self._current_r = self._previous_r[0] + self.dt*self._current_v


INTEGRATOR_LIST = {
    'semi_explicit_euler': SemiExplicitEulerIntegrator
}
METHODS = list(INTEGRATOR_LIST.keys())

def simulate(run_name : str, global_config : GlobalConfig, body_config : SystemState) -> None:
    integrator = INTEGRATOR_LIST[global_config.method](global_config, body_config)

    with common.get_my_progressbar(global_config.iter_num) as bar:
        for do_write, iter_idx in common.get_iter_indices(global_config.iter_num, global_config.output_point_num):
            integrator.advance()
            if do_write:
                common.write_body_config(run_name, integrator.state, iter_idx)
                bar.update(iter_idx)
        bar.update(global_config.iter_num)
