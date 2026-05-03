import math
import time

import jax.numpy as jnp
import numpy as np
import scipy as sp

import hj_reachability as hj


class ClosedLoopTrajectory:
    """
    Docstring for ClosedLoopTrajectory

    This class represents a closed-loop trajectory for a sample-and-hold controller
    designed from the output of hjpy.
    """

    def __init__(
        self, model, grid, times, value_function, initial_state, steps=100, **kwargs
    ):
        """
        Docstring for __init__

        :param self: ClosedLoopTrajctory object
        :param model: dynamics object from hjpy
        :param grid: spatial grid object from hjpy
        :param times: times over which the value function was computed
        :param value_function: value function tensor
        :param initial_state: numpy vector representing the initial state
        :param steps: (int) number of sample and hold steps
        :param **kwags: optional arguments to be passed to scipy.integrate.solve_ivp
        """
        self._model = model
        self._grid = grid
        if times[0] < times[-1]:
            raise ValueError(
                "Time axis should start at the final time and move backwards."
            )
        elif times[-1] < times[0]:
            self._times = times[::-1]
            self._V = value_function[::-1, ...]

        self._initial_state = np.array(initial_state)
        self._steps = max(int(steps), 1)

        self._u = None
        self._d = None

        self._us = [None] * self._steps
        self._ds = [None] * self._steps
        self._sols = [None] * self._steps

        self._gradV = np.array([self._grid.grad_values(self._V[i, ...]) for i in range(len(self._times))])

        self._solve_ivp(**kwargs)

    def x(self, t):
        return self._sols[self._get_sol_index(t)]

    def u(self, t):
        return self._us[self._get_sol_index(t)]

    def d(self, t):
        return self._ds[self._get_sol_index(t)]

    def gradient(self, t):
        return self._gradient(t, self.x(t))

    def value(self, t):
        return self._value(t, self.x(t))

    def _gradient(self, t, state):
        j, k = self._get_time_indexes(t)

        if j == k:
            gradient = self._grid.interpolate(
                self._gradV[j], state=state
            )

        else:
            gradient_left = self._grid.interpolate(
                self._gradV[j], state=state
            )

            gradient_right = self._grid.interpolate(
                self._gradV[j], state=state
            )

            gradient = (
                (t - self._times[j]) * gradient_right
                + (self._times[k] - t) * gradient_left
            ) / (self._times[k] - self._times[j])

        return gradient

    def _value(self, t, state):
        j, k = self._get_time_indexes(t)

        if j == k:
            value = self._grid.interpolate(self._V[j, ...], state=state)

        else:
            value_left = self._grid.interpolate(self._V[j, ...], state=state)
            value_right = self._grid.interpolate(self._V[k, ...], state=state)

            value = (
                (t - self._times[j]) * value_right + (self._times[k] - t) * value_left
            ) / (self._times[k] - self._times[j])

        return value

    def _dynamics(self, time, state):
        return self._model.__call__(state, self._u, self._d, time)

    def _get_time_indexes(self, t):
        i = np.searchsorted(self._times, t, side="right") - 1
        if t == self._times[i]:
            return i, i
        else:
            return i, i + 1

    def _get_sol_index(self, t):
        if t >= self._times[-1]:
            return -1
        elif t <= self._times[0]:
            return 0
        else:
            return math.floor(
                (t - self._times[0]) / (self._times[-1] - self._times[0]) * self._steps
            )


    def _rk4_step(self, f, t, x, dt):
        k1 = f(t, x)
        k2 = f(t + dt/2, x + dt/2 * k1)
        k3 = f(t + dt/2, x + dt/2 * k2)
        k4 = f(t + dt, x + dt * k3)
        return np.array(x + dt/6 * (k1 + 2*k2 + 2*k3 + k4))


    def _solve_ivp(self, substeps=1):
        state = self._initial_state

        T0 = self._times[0]
        Tf = self._times[-1]
        total_time = Tf - T0

        for i in range(self._steps):

            # Time interval for this control step
            t = total_time * (i / self._steps) + T0
            t_plus = total_time * ((i + 1) / self._steps) + T0
            dt = t_plus - t

            # Compute control (sample-and-hold)
            gradient = self._gradient(t, state)
            self._u = self._model.optimal_control(state, t, gradient)
            self._d = self._model.optimal_disturbance(state, t, gradient)

            # Optional: subdivide RK4 for better accuracy
            dt_sub = dt / substeps
            t_local = t

            for _ in range(substeps):
                state = self._rk4_step(self._dynamics, t_local, state, dt_sub)
                t_local += dt_sub

            # Store results
            self._sols[i] = state.copy()
            self._us[i] = self._u.copy()
            self._ds[i] = self._d.copy()



if __name__ == "__main__":
    
    from maze_builder import Smaze
    from roomba import RoombaDynamics

    # specify the dynamics we are considering
    model = RoombaDynamics()
    maze = Smaze()

    # specify the time horizon of the problem
    T = 200
    res = 80

    # specify the number of voxels to divide the spatial and temporal axes
    x_voxels = res
    y_voxels = res
    t_voxels = 2 * res

    # Specify PDE grid corners
    x_min = -10
    y_min = -10

    x_max = +10
    y_max = +10

    # discretize state-space and the time to solve the HJ Partial Differential Equation
    grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
        hj.sets.Box([x_min, y_min], [x_max, y_max]),
        [x_voxels + 1, y_voxels + 1],
    )
    times = np.linspace(0.0, -T, t_voxels + 1)

    # Initial state
    x0 = np.array((-9.25, 9.25))

    # specify the target and obstacle
    q = maze.obstacle_sdf(grid)
    r = maze.target_sdf(grid)

    def value_postprocessor(t, V):
        return jnp.minimum(jnp.maximum(V, .5 * r), .5 * q)

    # specify the accuracy with which to solve the HJ Partial Differential Equation
    solver_settings = hj.SolverSettings.with_accuracy(
        "very_high", value_postprocessor=value_postprocessor
    )

    # solve for the value function
    V = hj.solve(
        solver_settings,
        model,
        grid,
        times,
        jnp.minimum(.5 * r, .5 * q),
    )

    cl = ClosedLoopTrajectory(model, grid, times, V, initial_state=x0, steps = 1000)
