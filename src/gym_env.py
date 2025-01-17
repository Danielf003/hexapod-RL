# __credits__ = ["Kallinteris-Andreas"]

from typing import Dict, Tuple, Union
import os

import mujoco
import numpy as np
import roboticstoolbox as rtb

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from traj_calculation import param_traj
from traj_calculation import thetas_traj

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,
    "distance": 4.5,
    "lookat": np.array((0.0, 0.0, 1.15)),
    "elevation": -20.0,
}

class InputHolder:
    def __init__(self, timestep, input_func):
        self.timestep = timestep
        self.input_func = input_func
    def update(self, action, current_time):
        if current_time == 0:
            self.Tupd = 2*action[0]
            self.nsteps = int(self.Tupd/self.timestep)
            self.cnt = 0
            self.memory = self.input_func(action)
        if self.cnt >= self.nsteps:
            self.memory = self.input_func(action)
            self.cnt = 0
            self.Tupd = 2*action[0]
            self.nsteps = int(self.Tupd/self.timestep)
        else:
            self.cnt += 1

    def get_output(self):
        # print(f'cnt: {self.cnt}')
        return self.memory

def func(action):
    T_f, T_b, L, alpha = action[:4]
    delta_thetas = np.asarray(action[4:]).reshape((4,))
    # print(delta_thetas.shape)
    # C_x, C_y, C_z, a = param_traj(T_f, T_b, L, alpha, delta_thetas)
    C_x_left, C_y_left, C_z_left, a_left = param_traj(True, T_f, T_b, L, alpha, delta_thetas)
    C_x_right, C_y_right, C_z_right, a_right = param_traj(False, T_f, T_b, L, alpha, delta_thetas)

    return T_f, T_b, C_x_left, C_y_left, C_z_left, a_left, C_x_right, C_y_right, C_z_right, a_right

class LegRTB:
    def __init__(self):
        l1 = 0.3  # m length of the first link
        l2 = 0.848   # m length of the second link
        l3 = 1.221   # m length of the third link
        l4 = 0.6  # m length of the fourth link

        E1 = rtb.ET.Rz()

        E2 = rtb.ET.tx(l1)
        E3 = rtb.ET.Ry(flip=True)

        E4 = rtb.ET.tx(l2)
        E5 = rtb.ET.Ry(flip=True)

        E6 = rtb.ET.tx(l3)
        E7 = rtb.ET.Ry(flip=True)

        E8 = rtb.ET.tx(l4)

        self.robot = E1 * E2 * E3 * E4 * E5 * E6 * E7 * E8
        self.chosen_coords = (0,1,2,4)
        self.q0 = [0, 1.22, 4.01-2*np.pi, 5.76-2*np.pi]
    
    def calc_Jinv(self, q):
        J = self.robot.jacob0(q) #representation='eul' #jacob0 faster than analyt
        return np.linalg.inv(J[self.chosen_coords,:])

    def calc_Jdot(self, q, dq):
        # Jdot = self.robot.jacob0_dot(q, dq) #representation='eul'
        Jdot = np.tensordot(self.robot.hessian0(q), dq, (0, 0))
        return Jdot[self.chosen_coords,:]

class HexapodEnv(MujocoEnv, utils.EzPickle):
    r"""
    ## Description
    This environment is based on the work of Erez, Tassa, and Todorov in ["Infinite Horizon Model Predictive Control for Nonlinear Periodic Tasks"](http://www.roboticsproceedings.org/rss07/p10.pdf).
    The environment aims to increase the number of independent state and control variables compared to classical control environments.
    The hopper is a two-dimensional one-legged figure consisting of four main body parts - the torso at the top, the thigh in the middle, the leg at the bottom, and a single foot on which the entire body rests.
    The goal is to make hops that move in the forward (right) direction by applying torque to the three hinges that connect the four body parts.


    ## Action Space
    ```{figure} action_space_figures/hopper.png
    :name: hopper
    ```

    The action space is a `Box(-1, 1, (3,), float32)`. An action represents the torques applied at the hinge joints.

    | Num | Action                             | Control Min | Control Max | Name (in corresponding XML file) | Joint | Type (Unit)  |
    |-----|------------------------------------|-------------|-------------|----------------------------------|-------|--------------|
    | 0   | Torque applied on the thigh rotor  | -1          | 1           | thigh_joint                      | hinge | torque (N m) |
    | 1   | Torque applied on the leg rotor    | -1          | 1           | leg_joint                        | hinge | torque (N m) |
    | 2   | Torque applied on the foot rotor   | -1          | 1           | foot_joint                       | hinge | torque (N m) |


    ## Observation Space
    The observation space consists of the following parts (in order):

    - *qpos (5 elements by default):* Position values of the robot's body parts.
    - *qvel (6 elements):* The velocities of these individual body parts (their derivatives).

    By default, the observation does not include the robot's x-coordinate (`rootx`).
    This can  be included by passing `exclude_current_positions_from_observation=False` during construction.
    In this case, the observation space will be a `Box(-Inf, Inf, (12,), float64)`, where the first observation element is the x-coordinate of the robot.
    Regardless of whether `exclude_current_positions_from_observation` is set to `True` or `False`, the x- and y-coordinates are returned in `info` with the keys `"x_position"` and `"y_position"`, respectively.

    By default, however, the observation space is a `Box(-Inf, Inf, (11,), float64)` where the elements are as follows:

    | Num | Observation                                        | Min  | Max | Name (in corresponding XML file) | Joint | Type (Unit)              |
    | --- | -------------------------------------------------- | ---- | --- | -------------------------------- | ----- | ------------------------ |
    | 0   | z-coordinate of the torso (height of hopper)       | -Inf | Inf | rootz                            | slide | position (m)             |
    | 1   | angle of the torso                                 | -Inf | Inf | rooty                            | hinge | angle (rad)              |
    | 2   | angle of the thigh joint                           | -Inf | Inf | thigh_joint                      | hinge | angle (rad)              |
    | 3   | angle of the leg joint                             | -Inf | Inf | leg_joint                        | hinge | angle (rad)              |
    | 4   | angle of the foot joint                            | -Inf | Inf | foot_joint                       | hinge | angle (rad)              |
    | 5   | velocity of the x-coordinate of the torso          | -Inf | Inf | rootx                          | slide | velocity (m/s)           |
    | 6   | velocity of the z-coordinate (height) of the torso | -Inf | Inf | rootz                          | slide | velocity (m/s)           |
    | 7   | angular velocity of the angle of the torso         | -Inf | Inf | rooty                          | hinge | angular velocity (rad/s) |
    | 8   | angular velocity of the thigh hinge                | -Inf | Inf | thigh_joint                      | hinge | angular velocity (rad/s) |
    | 9   | angular velocity of the leg hinge                  | -Inf | Inf | leg_joint                        | hinge | angular velocity (rad/s) |
    | 10  | angular velocity of the foot hinge                 | -Inf | Inf | foot_joint                       | hinge | angular velocity (rad/s) |
    | excluded | x-coordinate of the torso                     | -Inf | Inf | rootx                            | slide | position (m)             |


    ## Rewards
    The total reward is: ***reward*** *=* *healthy_reward + forward_reward - ctrl_cost*.

    - *healthy_reward*:
    Every timestep that the Hopper is healthy (see definition in section "Episode End"),
    it gets a reward of fixed value `healthy_reward` (default is $1$).
    - *forward_reward*:
    A reward for moving forward,
    this reward would be positive if the Hopper moves forward (in the positive $x$ direction / in the right direction).
    $w_{forward} \times \frac{dx}{dt}$, where
    $dx$ is the displacement of the "torso" ($x_{after-action} - x_{before-action}$),
    $dt$ is the time between actions, which depends on the `frame_skip` parameter (default is $4$),
    and `frametime` which is $0.002$ - so the default is $dt = 4 \times 0.002 = 0.008$,
    $w_{forward}$ is the `forward_reward_weight` (default is $1$).
    - *ctrl_cost*:
    A negative reward to penalize the Hopper for taking actions that are too large.
    $w_{control} \times \|action\|_2^2$,
    where $w_{control}$ is `ctrl_cost_weight` (default is $10^{-3}$).

    `info` contains the individual reward terms.


    ## Starting State
    The initial position state is $[0, 1.25, 0, 0, 0, 0] + \mathcal{U}_{[-reset\_noise\_scale \times I_{6}, reset\_noise\_scale \times I_{6}]}$.
    The initial velocity state is $\mathcal{U}_{[-reset\_noise\_scale \times I_{6}, reset\_noise\_scale \times I_{6}]}$.

    where $\mathcal{U}$ is the multivariate uniform continuous distribution.

    Note that the z-coordinate is non-zero so that the hopper can stand up immediately.


    ## Episode End
    ### Termination
    If `terminate_when_unhealthy is True` (the default), the environment terminates when the Hopper is unhealthy.
    The Hopper is unhealthy if any of the following happens:

    1. An element of `observation[1:]` (if  `exclude_current_positions_from_observation=True`, otherwise `observation[2:]`) is no longer contained in the closed interval specified by the `healthy_state_range` argument (default is $[-100, 100]$).
    2. The height of the hopper (`observation[0]` if  `exclude_current_positions_from_observation=True`, otherwise `observation[1]`) is no longer contained in the closed interval specified by the `healthy_z_range` argument (default is $[0.7, +\infty]$) (usually meaning that it has fallen).
    3. The angle of the torso (`observation[1]` if  `exclude_current_positions_from_observation=True`, otherwise `observation[2]`) is no longer contained in the closed interval specified by the `healthy_angle_range` argument (default is $[-0.2, 0.2]$).

    ### Truncation
    The default duration of an episode is 1000 timesteps.


    ## Arguments
    Hopper provides a range of parameters to modify the observation space, reward function, initial state, and termination condition.
    These parameters can be applied during `gymnasium.make` in the following way:

    ```python
    import gymnasium as gym
    env = gym.make('Hopper-v5', ctrl_cost_weight=1e-3, ....)
    ```

    | Parameter                                    | Type      | Default               | Description                                                                                                                                                                                                 |
    | -------------------------------------------- | --------- | --------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
    | `xml_file`                                   | **str**   | `"hopper.xml"`        | Path to a MuJoCo model                                                                                                                                                                                      |
    | `forward_reward_weight`                      | **float** | `1`                   | Weight for _forward_reward_ term (see `Rewards` section)                                                                                                                                                    |
    | `ctrl_cost_weight`                           | **float** | `1e-3`                | Weight for _ctrl_cost_ reward (see `Rewards` section)                                                                                                                                                       |
    | `healthy_reward`                             | **float** | `1`                   | Weight for _healthy_reward_ reward (see `Rewards` section)                                                                                                                                                  |
    | `terminate_when_unhealthy`                   | **bool**  | `True`                | If `True`, issue a `terminated` signal is unhealthy (see `Episode End` section)                                                                                                                                |
    | `healthy_state_range`                        | **tuple** | `(-100, 100)`         | The elements of `observation[1:]` (if `exclude_current_positions_from_observation=True`, else `observation[2:]`) must be in this range for the hopper to be considered healthy (see `Episode End` section)  |
    | `healthy_z_range`                            | **tuple** | `(0.7, float("inf"))` | The z-coordinate must be in this range for the hopper to be considered healthy (see `Episode End` section)                                                                                                  |
    | `healthy_angle_range`                        | **tuple** | `(-0.2, 0.2)`         | The angle given by `observation[1]` (if `exclude_current_positions_from_observation=True`, else `observation[2]`) must be in this range for the hopper to be considered healthy (see `Episode End` section) |
    | `reset_noise_scale`                          | **float** | `5e-3`                | Scale of random perturbations of initial position and velocity (see `Starting State` section)                                                                                                               |
    | `exclude_current_positions_from_observation` | **bool**  | `True`                | Whether or not to omit the x-coordinate from observations. Excluding the position can serve as an inductive bias to induce position-agnostic behavior in policies(see `Observation Space` section)          |

    ## Version History
    * v5:
        - Minimum `mujoco` version is now 2.3.3.
        - Added support for fully custom/third party `mujoco` models using the `xml_file` argument (previously only a few changes could be made to the existing models).
        - Added `default_camera_config` argument, a dictionary for setting the `mj_camera` properties, mainly useful for custom environments.
        - Added `env.observation_structure`, a dictionary for specifying the observation space compose (e.g. `qpos`, `qvel`), useful for building tooling and wrappers for the MuJoCo environments.
        - Return a non-empty `info` with `reset()`, previously an empty dictionary was returned, the new keys are the same state information as `step()`.
        - Added `frame_skip` argument, used to configure the `dt` (duration of `step()`), default varies by environment check environment documentation pages.
        - Fixed bug: `healthy_reward` was given on every step (even if the Hopper was unhealthy), now it is only given when the Hopper is healthy. The `info["reward_survive"]` is updated with this change (related [GitHub issue](https://github.com/Farama-Foundation/Gymnasium/issues/526)).
        - Restored the `xml_file` argument (was removed in `v4`).
        - Added individual reward terms in `info` (`info["reward_forward"]`, `info["reward_ctrl"]`, `info["reward_survive"]`).
        - Added `info["z_distance_from_origin"]` which is equal to the vertical distance of the "torso" body from its initial position.
    * v4: All MuJoCo environments now use the MuJoCo bindings in mujoco >= 2.1.3.
    * v3: Support for `gymnasium.make` kwargs such as `xml_file`, `ctrl_cost_weight`, `reset_noise_scale`, etc. rgb rendering comes from tracking camera (so agent does not run away from screen)
    * v2: All continuous control environments now use mujoco-py >= 1.50.
    * v1: max_time_steps raised to 1000 for robot based tasks. Added reward_threshold to environments.
    * v0: Initial versions release.
    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
            # "rgbd_tuple",
        ],
    }

    def __init__(
        self,
        xml_file: str =  os.path.join(os.getcwd(),"hexapod.xml"),
        # xml_file: str = r"D:\hexapod.xml",
        frame_skip: int = 1,
        default_camera_config: Dict[str, Union[float, int]] = DEFAULT_CAMERA_CONFIG,

        forward_reward_weight: float = 300.0,
        ctrl_cost_weight: float = 1e-2,
        healthy_reward: float = 0*2e-3,
        stab_reward_weight_z = 1.0,
        # stab_reward_weight_y = 1.0,

        terminate_when_unhealthy: bool = True,

        healthy_state_range: Tuple[float, float] = (-100.0, 100.0),
        healthy_z_range: Tuple[float, float] = (0.7, float("inf")),

        healthy_sin_z_range: Tuple[float, float] = (-0.25, 0.25),
        healthy_tau_range: Tuple[float, float] = (-10**4-500, 10**4+500),

        healthy_angle_range: Tuple[float, float] = (-100.0, 100.0),
        

        reset_noise_scale: float = 5e-3,
        exclude_current_positions_from_observation: bool = True,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            xml_file,
            frame_skip,
            default_camera_config,
            forward_reward_weight,

            ctrl_cost_weight,
            healthy_reward,
            terminate_when_unhealthy,
            healthy_state_range,
            healthy_z_range,
            healthy_angle_range,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            **kwargs,
        )

        self._forward_reward_weight = forward_reward_weight

        self._ctrl_cost_weight = ctrl_cost_weight

        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy

        self._healthy_state_range = healthy_state_range

        self._healthy_sin_z_range = healthy_sin_z_range
        self._healthy_tau_range = healthy_tau_range

        self._stab_reward_weight_z = stab_reward_weight_z
        

        self._healthy_z_range = healthy_z_range
        self._healthy_angle_range = healthy_angle_range

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=None,
            default_camera_config=default_camera_config,
            **kwargs,
        )

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
                # "rgbd_tuple",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

        obs_size = (
            self.data.qpos[:6].size
            + self.data.qpos[3:7].size
            # + self.data.qvel.size
            # - exclude_current_positions_from_observation
        )
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )

        # th = np.array([0, 0.25,-0.2,0])
        self.action_space = Box(
            # low= np.array([0, 0, 0, -0.44,     -0.17, 0, -0.2, -0.1]), high=np.array([50, 10, 2, 0.44,     0.17, 0.25, 0, 0.1]), shape=(8,), dtype=np.float64
            # low= np.array([5, 0, 0.1, -0.44,     -0.10, 0, -0.2, -0.1]), high=np.array([50, 2, 1, 0.44,     0.10, 0.25, 0.2, 0.1]), shape=(8,), dtype=np.float64
            # low= np.array([5, 0, 0.5, -0.2*0,     0, -0.1, -0.2, -0.1]), high=np.array([20, 0.1*0, 1, 0.2*0,     0, 0.25*0-0.1, 0.2*0-0.2, 0.1*0-0.1]), shape=(8,), dtype=np.float64
            low= np.array([5, 0, 0.5, -0.2*0,     0, np.deg2rad(15), -np.deg2rad(10), -np.deg2rad(10)]), high=np.array([20, 0.1*0, 1, 0.2*0,     0, np.deg2rad(25), np.deg2rad(25), np.deg2rad(10)]), shape=(8,), dtype=np.float64
        )
        # self.memory = InputHolder(MujocoEnv.dt, func)
        self.memory = InputHolder(self.model.opt.timestep, func)
        
        self.leg_virtual = LegRTB()

        self.observation_structure = {
            # "skipped_qpos": 1 * exclude_current_positions_from_observation,
            # "qpos": self.data.qpos.size
            # - 1 * exclude_current_positions_from_observation,
            # "qvel": self.data.qvel.size,
        }

    @property
    def healthy_reward(self):
        return self.is_healthy * self._healthy_reward

    def control_cost(self, action):
        # control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        violations = (action>self.action_space.high)*np.abs(action-self.action_space.high) + (action<self.action_space.low)*np.abs(action-self.action_space.low)
        control_cost = self._ctrl_cost_weight * np.sum(violations)
        # print(control_cost)
        return control_cost

    @property
    def is_healthy(self):
        is_healthy = True
        # z, angle = self.data.qpos[1:3]
        # state = self.state_vector()[2:]
        c_z = self.data.sensordata[2]

        # min_state, max_state = self._healthy_state_range
        # min_z, max_z = self._healthy_z_range
        # min_angle, max_angle = self._healthy_angle_range

        min_sin_z, max_sin_z = self._healthy_sin_z_range
        min_tau, max_tau = self._healthy_tau_range
        # min_sin_y, max_sin_y = self._healthy_sin_y_range

        # healthy_state = np.all(np.logical_and(min_state < state, state < max_state))
        # healthy_z = min_z < z < max_z
        healthy_angle_z = min_sin_z < c_z < max_sin_z
        # print(f'self.data.ctrl = {self.data.ctrl}')
        ##print(self.data.ctrl)

        healthy_tau = (np.abs(self.data.ctrl) < 10**7).all()

        healthy_qvel = (np.abs(self.data.qvel[6:]) < 1e2).all()
        healthy_zvel = (np.abs(self.data.qvel[2]) < 4).all()
        healthy_xvel = (np.abs(self.data.qvel[0]) < 5).all()
        
        # healthy_angle_y  = min_sin_y < c_y < max_sin_y
        # print(healthy_tau)

        is_healthy = all((healthy_angle_z, healthy_tau, healthy_qvel, healthy_zvel, healthy_xvel))
        # is_healthy = healthy_angle_z
        # is_healthy = np.array([healthy_angle_z, healthy_tau])

        return is_healthy

    def _get_obs(self):
        # position = self.data.qpos.flatten()
        # velocity = np.clip(self.data.qvel.flatten(), -10, 10)

        V = self.data.qvel[:3].flatten()
        W = self.data.qvel[3:6].flatten()
        alphas = self.data.qpos[3:7].flatten()
        # touch = self.data.sensordata[]
        
        # print(f'V = {V}; W = {W}; alphas = {alphas}')

        # if self._exclude_current_positions_from_observation:
            # position = position[1:]

        observation = np.concatenate((V, W, alphas)).ravel()
        # print(f'observation_size = {observation.size}')
        return observation

    # def ctrl_f(self, action, holder: InputHolder, leg_virtual: LegRTB):

    #     # action - T_f, T_b, L, alfa, delta_thetas
    #     # print(f'time: {self.data.time}')
    #     t = self.data.time

    #     legdofs= self.model.jnt_dofadr[1:]
    #     legqpos=self.model.jnt_qposadr[1:]

    #     use_traj = 1
    #     use_memory = 1
    #     use_rtb_jacs = 0

    #     nj = 4
    #     nlegs = 6
    #     # qdes = np.array([0, 1.22, 4.01, 5.76])
    #     qdes = np.zeros(nj*1)
    #     dqdes = np.zeros(nj*1)
    #     ddqdes = np.zeros(nj*1)

    #     # # Выходы НС
    #     # T_f = 2
    #     # T_b = 0
    #     # L = 2
    #     # alfa = 0.26
    #     # H = 1
    #     # delta_T = T_f
    #     # delta_thetas = np.array([0, 0.25,-0.2,0])
    #     # C_x, C_y, C_z, a = param_traj(T_f, T_b, L, alfa, delta_thetas)

    #     if use_memory:
    #         holder.update(action, t)
    #         T_f, T_b, C_x, C_y, C_z, a = holder.get_output()
    #     else:
    #         C_x, C_y, C_z, a = param_traj(action)        

    #     if use_traj:
    #         if use_rtb_jacs:
    #             qdes1, dqdes1, ddqdes1 = thetas_traj(t, T_f, T_b, 0, C_x, C_y, C_z, a, leg_virtual.calc_Jinv, leg_virtual.calc_Jdot)
    #             qdes2, dqdes2, ddqdes2 = thetas_traj(t, T_f, T_b, T_f, C_x, C_y, C_z, a, leg_virtual.calc_Jinv, leg_virtual.calc_Jdot)
    #             qdes1[2] = -qdes1[2]
    #             qdes2[2] = -qdes2[2]
    #         else:
    #             qdes1, dqdes1, ddqdes1 = thetas_traj(t, T_f, T_b, 0, C_x, C_y, C_z, a)
    #             qdes2, dqdes2, ddqdes2 = thetas_traj(t, T_f, T_b, T_f, C_x, C_y, C_z, a)
    #             qdes1[2] = -qdes1[2]
    #             qdes2[2] = -qdes2[2]

    #         q0 = [0, 1.22, 4.01-2*np.pi, 5.76-2*np.pi]
    #         qdes1 = qdes1 - np.array(q0)
    #         qdes2 = qdes2 - np.array(q0)

    #     # kp, kd = np.diag([50,40,30,40]*nlegs), np.diag([2,5,2,2]*nlegs)
    #     kp, kd = np.diag([5000,4000,3000,13000]*1), np.diag([90,300,200,200]*1)
    #     u = np.zeros( self.model.nv)
    #     e = np.zeros(nj*nlegs)
    #     de = np.zeros(nj*nlegs)
    #     # print(f'time: { model.time}')
    #     for i in range(nlegs):
    #         if use_traj:
    #             if i in (0, 2, 4):
    #                     qdes = qdes1
    #                     # qdes[2] = qdes1[2] - 2*np.pi
    #                     # qdes[3] = qdes1[3] - 2*np.pi
    #                     dqdes = dqdes1
    #                     ddqdes = ddqdes1
    #                     # print(f'i = {i}, qdes = {qdes}')
    #             else:
    #                     qdes = qdes2
    #                     # qdes[2] = -qdes2[2] - 2*np.pi
    #                     # qdes[3] = -qdes2[3] - 2*np.pi
    #                     dqdes = dqdes2
    #                     ddqdes = ddqdes2
    #                     # print(f'i = {i}, qdes = {qdes}')

    #         e[0+i*4:4+i*4] =  self.data.qpos[legqpos][0+i*4:4+i*4]-qdes
    #         de[0+i*4:4+i*4] =  self.data.qvel[legdofs][0+i*4:4+i*4]-dqdes
            
    #         u[legdofs[0+i*4:4+i*4]] = ddqdes - kp@e[0+i*4:4+i*4] - kd@de[0+i*4:4+i*4]
        
    #     # u[legdofs] = np.array([1,1,1,1])
    #     # print(model.jnt_dofadr)
    #     Mu = np.empty( self.model.nv)
    #     mujoco.mj_mulM( self.model,  self.data, Mu, u)#+c)
    #     tau = Mu +  self.data.qfrc_bias
    #     tau = tau[legdofs]

    #     # print(tau)
    #     # print(data.qpos[:4])
    #     return tau
    
    def qdes2tau_CTC(self, qdes, dqdes, ddqdes):
        legdofs=self.model.jnt_dofadr[1:]
        legqpos=self.model.jnt_qposadr[1:]

        # nj = 4
        nlegs = 6

        # qdes = np.zeros(nj*nlegs)
        # dqdes = np.zeros(nj*nlegs)
        # ddqdes = np.zeros(nj*nlegs)
        e = self.data.qpos[legqpos]-qdes
        de = self.data.qvel[legdofs]-dqdes

        # kp, kd = np.diag([50,40,30,40]*nlegs), np.diag([2,5,2,2]*nlegs)
        # kp, kd = np.diag([5000,4000,3000,13000]*1), np.diag([90,300,200,200]*1)
        kp, kd = np.diag([5000,4000,3000,4000]*nlegs), np.diag([90,300,200,200]*nlegs)
        u = np.zeros(self.model.nv)
        u[legdofs] = ddqdes - kp@e - kd@de
        
        # u[legdofs] = np.array([1,1,1,1])
        # print(model.jnt_dofadr)
        Mu = np.empty(self.model.nv)
        mujoco.mj_mulM(self.model, self.data, Mu, u)#+c)
        tau = Mu + self.data.qfrc_bias
        tau = tau[legdofs]
        # print(tau)
        # print(data.qpos[:4])
        return tau

    def action2qdes_3pod(self, action, holder: InputHolder, leg_virtual: LegRTB):
        t = self.data.time

        use_traj = 1
        use_memory = 1
        use_rtb_jacs = 0

        nj = 4
        nlegs = 6
        # qdes = np.array([0, 1.22, 4.01, 5.76])
        qdes = np.zeros(nj*nlegs)
        dqdes = np.zeros(nj*nlegs)
        ddqdes = np.zeros(nj*nlegs)

        # # Выходы НС
        # T_f = 7
        # T_b = 0
        # L = 1
        # alfa = 0
        # # H = 1
        # delta_T = T_f
        # delta_thetas = np.array([0, 0.25,-0.2,0])

        # C_x_left, C_y_left, C_z_left, a_left = param_traj(True, T_f, T_b, L, alfa, delta_thetas)
        # C_x_right, C_y_right, C_z_right, a_right = param_traj(False, T_f, T_b, L, alfa, delta_thetas)
        # print(f'alfa = {alfa}')

        if use_traj:
            if use_memory:
                holder.update(action, t)
                T_f, T_b, C_x_left, C_y_left, C_z_left, a_left, C_x_right, C_y_right, C_z_right, a_right = holder.get_output()
            else:
                # C_x, C_y, C_z, a = param_traj(T_f, T_b, L, alfa, delta_thetas)   
                C_x_left, C_y_left, C_z_left, a_left = param_traj(True, action)
                C_x_right, C_y_right, C_z_right, a_right = param_traj(False, action)

            if use_rtb_jacs:
                qdes1_left, dqdes1_left, ddqdes1_left = thetas_traj(t, T_f, T_b, 0, C_x_left, C_y_left, C_z_left, a_left, leg_virtual.calc_Jinv, leg_virtual.calc_Jdot)
                qdes1_right, dqdes1_right, ddqdes1_right = thetas_traj(t, T_f, T_b, 0, C_x_right, C_y_right, C_z_right, a_right, leg_virtual.calc_Jinv, leg_virtual.calc_Jdot)
                qdes2_left, dqdes2_left, ddqdes2_left = thetas_traj(t, T_f, T_b, T_f, C_x_left, C_y_left, C_z_left, a_left, leg_virtual.calc_Jinv, leg_virtual.calc_Jdot)
                qdes2_right, dqdes2_right, ddqdes2_right = thetas_traj(t, T_f, T_b, T_f, C_x_right, C_y_right, C_z_right, a_right, leg_virtual.calc_Jinv, leg_virtual.calc_Jdot)
                
                qdes1_left[2] = -qdes1_left[2]
                qdes1_right[2] = -qdes1_right[2]
                qdes2_left[2] = -qdes2_left[2]
                qdes2_right[2] = -qdes2_right[2]
            else:
                # qdes1, dqdes1, ddqdes1 = thetas_traj(t, T_f, T_b, 0, C_x, C_y, C_z, a)
                # qdes2, dqdes2, ddqdes2 = thetas_traj(t, T_f, T_b, delta_T, C_x, C_y, C_z, a)
                qdes1_left, dqdes1_left, ddqdes1_left = thetas_traj(t, T_f, T_b, 0, C_x_left, C_y_left, C_z_left, a_left) # deltaT = 0
                qdes1_right, dqdes1_right, ddqdes1_right = thetas_traj(t, T_f, T_b, 0, C_x_right, C_y_right, C_z_right, a_right) # deltaT = 0
                qdes2_left, dqdes2_left, ddqdes2_left = thetas_traj(t, T_f, T_b, T_f, C_x_left, C_y_left, C_z_left, a_left) # deltaT = Tf
                qdes2_right, dqdes2_right, ddqdes2_right = thetas_traj(t, T_f, T_b, T_f, C_x_right, C_y_right, C_z_right, a_right) # deltaT = Tf
                qdes1_left[2] = -qdes1_left[2]
                qdes1_right[2] = -qdes1_right[2]
                qdes2_left[2] = -qdes2_left[2]
                qdes2_right[2] = -qdes2_right[2]

            q0 = [0, 1.22, 4.01-2*np.pi, 5.76-2*np.pi]
            qdes1_left = qdes1_left - np.array(q0)
            qdes1_right = qdes1_right - np.array(q0)
            qdes2_left = qdes2_left - np.array(q0)
            qdes2_right = qdes2_right - np.array(q0)

        for i in range(nlegs):
            if use_traj:
                if i in (0, 4):
                    qdes[0+i*4:4+i*4] = qdes1_right
                    dqdes[0+i*4:4+i*4] = dqdes1_right
                    ddqdes[0+i*4:4+i*4] = ddqdes1_right
                elif i == 2:
                    qdes[0+i*4:4+i*4] = qdes1_left
                    dqdes[0+i*4:4+i*4] = dqdes1_left
                    ddqdes[0+i*4:4+i*4] = ddqdes1_left
                elif i in (1, 3):
                    qdes[0+i*4:4+i*4] = qdes2_left
                    dqdes[0+i*4:4+i*4] = dqdes2_left
                    ddqdes[0+i*4:4+i*4] = ddqdes2_left
                else:
                    qdes[0+i*4:4+i*4] = qdes2_right
                    dqdes[0+i*4:4+i*4] = dqdes2_right
                    ddqdes[0+i*4:4+i*4] = ddqdes2_right
        return qdes, dqdes, ddqdes
                        
    def step(self, action):
        # x_position_before = self.data.qpos[0]
        # print(f'time: {self.data.time}')
        y_position_before = self.data.qpos[1]

        action_clipped = np.clip(action, self.action_space.low, self.action_space.high)
        qdes, dqdes, ddqdes = self.action2qdes_3pod(action_clipped, self.memory, self.leg_virtual)
        torques = self.qdes2tau_CTC(qdes, dqdes, ddqdes)

        self.do_simulation(torques, self.frame_skip)
        # x_position_after = self.data.qpos[0]

        y_position_after = self.data.qpos[1]
        # x_velocity = (x_position_after - x_position_before) / self.dt

        moving_along_axes_y = (y_position_after - y_position_before)
        # alpha = self.data.sensordata
        

        observation = self._get_obs()
        reward, reward_info = self._get_rew(moving_along_axes_y, action)
        terminated = (not self.is_healthy) and self._terminate_when_unhealthy
        info = {
            # "x_position": x_position_after,
            "moving_along_axes_y": moving_along_axes_y,
            # "sin_x":,
            # "sin_y":,
            # "z_distance_from_origin": self.data.qpos[1] - self.init_qpos[1],
            # "x_velocity": x_velocity,
            **reward_info,
        }

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return observation, reward, terminated, False, info

    def _get_rew(self, moving_along_axes_y, action):

        forward_reward = self._forward_reward_weight * moving_along_axes_y
        # print(f'forward_reward = {forward_reward}')
        s_z = self.data.sensordata[2]

        stab_reward = - self._stab_reward_weight_z * abs(s_z) 
        # print(f'stab_reward = {stab_reward}')

        healthy_reward = self.healthy_reward
        # print(f'healthy_reward = {healthy_reward}')
        rewards = forward_reward + healthy_reward + stab_reward*0
        # reward = forward_reward - stab_reward + healthy_reward

        ctrl_cost = self.control_cost(action)
        costs = ctrl_cost

        reward = rewards - costs

        reward_info = {
            "reward_forward": forward_reward,
            # "reward_ctrl": -ctrl_cost,
            "reward_survive": healthy_reward,
        }

        return reward, reward_info

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def _get_reset_info(self):
        return {
            "x_position": self.data.qpos[0],
            "z_distance_from_origin": self.data.qpos[1] - self.init_qpos[1],
        }