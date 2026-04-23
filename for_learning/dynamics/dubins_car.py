import jax.numpy as jnp
from hj_reachability import dynamics, sets


class DubinsCarDynamics(dynamics.ControlAndDisturbanceAffineDynamics):
    def __init__(
        self,
        control_mode="min",
        disturbance_mode="max",
        uMax=+1.0,
        uMin=-1.0,
    ):
        self.uMax = uMax
        self.uMin = uMin

        control_space = sets.Box(
            jnp.array([self.uMin, 0.0]), jnp.array([self.uMax, 1.0])
        )
        disturbance_space = sets.Box(
            jnp.array([0.0]), jnp.array([0.0])
        )  # no disturbance

        super().__init__(
            control_mode, disturbance_mode, control_space, disturbance_space
        )

    def open_loop_dynamics(self, state, time):
        f = jnp.array([[0.0], [0.0], [0.0]])

        return f.reshape([3])

    def control_jacobian(self, state, time):
        x, y, theta = state

        g = jnp.array(
            [
                [0.0, jnp.cos(theta)],
                [0.0, jnp.sin(theta)],
                [1.0, 0.0],
            ]
        )

        return g.reshape([3, 2])

    def disturbance_jacobian(self, state, time):
        h = jnp.array([[0.0], [0.0], [0.0]])

        return h.reshape([3, 1])


class DubinsCarDynamicsFixedVelocity(dynamics.ControlAndDisturbanceAffineDynamics):
    def __init__(
        self,
        control_mode="min",
        disturbance_mode="max",
        uMax=+1.0,
        uMin=-1.0,
    ):
        self.uMax = uMax
        self.uMin = uMin

        control_space = sets.Box(jnp.array([self.uMin]), jnp.array([self.uMax]))
        disturbance_space = sets.Box(
            jnp.array([0.0]), jnp.array([0.0])
        )  # no disturbance

        super().__init__(
            control_mode, disturbance_mode, control_space, disturbance_space
        )

    def open_loop_dynamics(self, state, time):
        x, y, theta = state

        f = jnp.array([[jnp.cos(theta)], [jnp.sin(theta)], [0.0]])

        return f.reshape([3])

    def control_jacobian(self, state, time):
        g = jnp.array([[0.0], [0.0], [1.0]])

        return g.reshape([3, 1])

    def disturbance_jacobian(self, state, time):
        h = jnp.array([[0.0], [0.0], [0.0]])

        return h.reshape([3, 1])
