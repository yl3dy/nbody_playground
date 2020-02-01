import collections
from abc import ABC, abstractmethod
import math
import time
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

class ExplicitRK2Integrator(BaseIntegrator):
    NUM_PREVIOUS_STATES = 1
    def _main_method(self):
        dt, a = self.dt, self._grav_accel
        prev_r, prev_v = self._previous_r[0], self._previous_v[0]

        k1_r = prev_v*dt
        k1_v = a(prev_r)*dt
        k2_r = (prev_v + 0.5*k1_v)*dt
        k2_v = a(prev_r + 0.5*k1_r)*dt
        self._current_r = prev_r + k2_r
        self._current_v = prev_v + k2_v

class RalstonIntegrator(BaseIntegrator):
    NUM_PREVIOUS_STATES = 1
    def _main_method(self):
        dt, a = self.dt, self._grav_accel
        prev_r, prev_v = self._previous_r[0], self._previous_v[0]

        k1_r = prev_v*dt
        k1_v = a(prev_r)*dt
        k2_r = (prev_v + (2/3)*k1_v)*dt
        k2_v = a(prev_r + (2/3)*k1_r)*dt
        self._current_r = prev_r + 0.25*k1_r + 0.75*k2_r
        self._current_v = prev_v + 0.25*k1_v + 0.75*k2_v

class ExplicitRK4Integrator(BaseIntegrator):
    NUM_PREVIOUS_STATES = 1
    def _main_method(self):
        dt, a = self.dt, self._grav_accel
        prev_r, prev_v = self._previous_r[0], self._previous_v[0]

        k1_r = prev_v*dt
        k1_v = a(prev_r)*dt
        k2_r = (prev_v + 0.5*k1_v)*dt
        k2_v = a(prev_r + 0.5*k1_r)*dt
        k3_r = (prev_v + 0.5*k2_v)*dt
        k3_v = a(prev_r + 0.5*k2_r)*dt
        k4_r = (prev_v + k3_v)*dt
        k4_v = a(prev_r + k3_r)*dt
        self._current_r = prev_r + (k1_r + 2*k2_r + 2*k3_r + k4_r) / 6
        self._current_v = prev_v + (k1_v + 2*k2_v + 2*k3_v + k4_v) / 6

class VelocityVerletIntegrator(BaseIntegrator):
    NUM_PREVIOUS_STATES = 1
    def _main_method(self):
        dt, a = self.dt, self._grav_accel
        prev_r, prev_v = self._previous_r[0], self._previous_v[0]

        if not hasattr(self, '_a_prev'):
            self._a_prev = a(prev_r)

        self._current_r = prev_r + prev_v*dt + 0.5*self._a_prev*dt**2
        a_new = a(self._current_r)
        self._current_v = prev_v + 0.5*(self._a_prev + a_new)*dt
        self._a_prev = a_new

class AdamsBashforth2Integrator(BaseIntegrator):
    NUM_PREVIOUS_STATES = 2
    def _starting_method(self):
        """Semi-explicit Euler as a starting method."""
        self._current_v = self._previous_v[0] + self.dt*self._grav_accel(self._previous_r[0])
        self._current_r = self._previous_r[0] + self.dt*self._current_v
    def _main_method(self):
        dt, a = self.dt, self._grav_accel
        prev_r, prev_v = self._previous_r, self._previous_v

        self._current_v = prev_v[0] + 1.5*dt*a(prev_r[0]) - 0.5*dt*a(prev_r[1])
        self._current_r = prev_r[0] + 1.5*dt*prev_v[0] - 0.5*dt*prev_v[1]

class AdamsBashforth5Integrator(BaseIntegrator):
    NUM_PREVIOUS_STATES = 5
    def _starting_method(self):
        """Semi-explicit Euler as a starting method."""
        self._current_v = self._previous_v[0] + self.dt*self._grav_accel(self._previous_r[0])
        self._current_r = self._previous_r[0] + self.dt*self._current_v
    def _main_method(self):
        dt, a = self.dt, self._grav_accel
        prev_r, prev_v = self._previous_r, self._previous_v

        self._current_v = prev_v[0] + dt*(1901/720*a(prev_r[0]) - 2774/720*a(prev_r[1]) + 2616/720*a(prev_r[2]) - 1274/720*a(prev_r[3]) + 251/720*a(prev_r[4]))
        self._current_r = prev_r[0] + dt*(1901/720*prev_v[0] - 2774/720*prev_v[1] + 2616/720*prev_v[2] - 1274/720*prev_v[3] + 251/720*prev_v[4])

class BaseRuthIntegrator(BaseIntegrator):
    NUM_PREVIOUS_STATES = 1
    C_COEFFS = None
    D_COEFFS = None
    def _main_method(self):
        dt, a = self.dt, self._grav_accel
        prev_r, prev_v = self._previous_r[0], self._previous_v[0]

        if self.C_COEFFS is None or self.D_COEFFS is None:
            raise NotImplementedError

        self._current_r = prev_r.copy()
        self._current_v = prev_v.copy()
        for c, d in zip(self.C_COEFFS, self.D_COEFFS):
            self._current_v += d*a(self._current_r)*dt
            self._current_r += c*self._current_v*dt

class Ruth3Integrator(BaseRuthIntegrator):
    C_COEFFS = [2/3, -2/3, 1]
    D_COEFFS = [7/24, 3/4, -1/24]

class Ruth4Integrator(BaseRuthIntegrator):
    C_COEFFS = [1 / (2*(2 - 2**(1/3))), (1 - 2**(1/3)) / (2*(2 - 2**(1/3))), (1 - 2**(1/3)) / (2*(2 - 2**(1/3))), 1 / (2*(2 - 2**(1/3)))]
    D_COEFFS = [0, 1/(2 - 2**(1/3)), -2**(1/3) / (2 - 2**(1/3)), 1/(2 - 2**(1/3))]


INTEGRATOR_LIST = {
    'semi_explicit_euler': SemiExplicitEulerIntegrator,
    'explicit_rk2': ExplicitRK2Integrator,
    'ralston': RalstonIntegrator,
    'explicit_rk4': ExplicitRK4Integrator,
    'velocity_verlet': VelocityVerletIntegrator,
    'adams_bashforth_2': AdamsBashforth2Integrator,
    'adams_bashforth_5': AdamsBashforth5Integrator,
    'ruth3': Ruth3Integrator,
    'ruth4': Ruth4Integrator,
}
METHODS = list(INTEGRATOR_LIST.keys())

def simulate(run_name : str, global_config : GlobalConfig, body_config : SystemState) -> None:
    integrator = INTEGRATOR_LIST[global_config.method](global_config, body_config)

    t0 = time.time()
    with common.get_my_progressbar(global_config.iter_num) as bar:
        for do_write, iter_idx in common.get_iter_indices(global_config.iter_num, global_config.output_point_num):
            integrator.advance()
            if do_write:
                common.write_body_config(run_name, integrator.state, iter_idx)
                bar.update(iter_idx)
        bar.update(global_config.iter_num)
    print('Integration time: {:.3} s'.format(time.time() - t0))
