import os

import numpy as np
from gymnasium.utils.ezpickle import EzPickle
from gymnasium_robotics.envs.robot_env import MujocoRobotEnv
from gymnasium_robotics.utils import rotations

# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('fetch', 'pick_and_place.xml')

DEFAULT_CAMERA_CONFIG = {
    "distance": 2.5,
    "azimuth": 132.0,
    "elevation": -14.0,
    "lookat": np.array([1.3, 0.75, 0.55]),
}


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class FetchEnv(MujocoRobotEnv):
    """Superclass for all Fetch environments."""

    def __init__(
            self,
            model_path,
            n_substeps,
            gripper_extra_height,
            block_gripper,
            has_object,
            target_in_the_air,
            target_offset,
            obj_range,
            target_range,
            distance_threshold,
            initial_qpos,
            reward_type,
            fix_object,
            fix_goal
    ):
        """Initializes a new Fetch environment.

        Args:
            model_path (string): path to the environments XML file
            n_substeps (int): number of substeps the simulation runs on every call to step
            gripper_extra_height (float): additional height above the table when positioning the gripper
            block_gripper (boolean): whether the gripper is blocked (i.e. not movable) or not
            has_object (boolean): whether the environment has an object
            target_in_the_air (boolean): whether the target should be in the air above the table or on the table surface
            target_offset (float or array with 3 elements): offset of the target
            obj_range (float): range of a uniform distribution for sampling initial object positions
            target_range (float): range of a uniform distribution for sampling a target
            distance_threshold (float): the threshold after which a goal is considered achieved
            initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
            reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
            fix_object (boolean): whether the object is fixed or not
            fix_goal (boolean): whether the goal is fixed or not
        """
        self.gripper_extra_height = gripper_extra_height
        self.block_gripper = block_gripper
        self.has_object = has_object
        self.target_in_the_air = target_in_the_air
        self.target_offset = target_offset
        self.obj_range = obj_range
        self.target_range = target_range
        self.distance_threshold = distance_threshold
        self.reward_type = reward_type

        self.fix_object = fix_object
        if fix_object:
            self.fixed_object = np.array([1.4, 0.6])

        self.fix_goal = fix_goal
        if fix_goal:
            self.fixed_goal = np.array([1.48673746, 0.69548325, 0.6])

        super(FetchEnv, self).__init__(
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            model_path=model_path,
            n_substeps=n_substeps,
            n_actions=4,
            initial_qpos=initial_qpos,
            # render_mode="human",  # Uncommenting it will use the in-built rendering (unintended)
        )

    # GoalEnv methods
    # ----------------------------

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, goal)
        if self.reward_type == "sparse":
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d

    # RobotEnv methods
    # ----------------------------

    def _step_callback(self):
        # if self.block_gripper:
        #     self.sim.data.set_joint_qpos("robot0:l_gripper_finger_joint", 0.0)
        #     self.sim.data.set_joint_qpos("robot0:r_gripper_finger_joint", 0.0)
        #     self.sim.forward()

        if self.block_gripper:
            self._utils.set_joint_qpos(
                self.model, self.data, "robot0:l_gripper_finger_joint", 0.0
            )
            self._utils.set_joint_qpos(
                self.model, self.data, "robot0:r_gripper_finger_joint", 0.0
            )
            self._mujoco.mj_forward(self.model, self.data)

    def _set_action(self, action):
        # assert action.shape == (4,)
        # action = (
        #     action.copy()
        # )  # ensure that we don't change the action outside of this scope
        # pos_ctrl, gripper_ctrl = action[:3], action[3]
        #
        # pos_ctrl *= 0.05  # limit maximum change in position
        # rot_ctrl = [
        #     1.0,
        #     0.0,
        #     1.0,
        #     0.0,
        # ]  # fixed rotation of the end effector, expressed as a quaternion
        # gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        # assert gripper_ctrl.shape == (2,)
        # if self.block_gripper:
        #     gripper_ctrl = np.zeros_like(gripper_ctrl)
        # action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])
        #
        # # Apply action to simulation.
        # utils.ctrl_set_action(self.sim, action)
        # utils.mocap_set_action(self.sim, action)

        assert action.shape == (4,)
        action = (
            action.copy()
        )  # ensure that we don't change the action outside of this scope
        pos_ctrl, gripper_ctrl = action[:3], action[3]

        pos_ctrl *= 0.05  # limit maximum change in position
        rot_ctrl = [
            1.0,
            0.0,
            1.0,
            0.0,
        ]  # fixed rotation of the end effector, expressed as a quaternion
        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        assert gripper_ctrl.shape == (2,)
        if self.block_gripper:
            gripper_ctrl = np.zeros_like(gripper_ctrl)
        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

        # Apply action to simulation.
        self._utils.ctrl_set_action(self.model, self.data, action)
        self._utils.mocap_set_action(self.model, self.data, action)

    def _get_obs(self):
        # # positions
        # grip_pos = self.sim.data.get_site_xpos("robot0:grip")
        # dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        # grip_velp = self.sim.data.get_site_xvelp("robot0:grip") * dt
        # robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)
        # if self.has_object:
        #     object_pos = self.sim.data.get_site_xpos("object0")
        #     # rotations
        #     object_rot = rotations.mat2euler(self.sim.data.get_site_xmat("object0"))
        #     # velocities
        #     object_velp = self.sim.data.get_site_xvelp("object0") * dt
        #     object_velr = self.sim.data.get_site_xvelr("object0") * dt
        #     # gripper state
        #     object_rel_pos = object_pos - grip_pos
        #     object_velp -= grip_velp
        # else:
        #     object_pos = (
        #         object_rot
        #     ) = object_velp = object_velr = object_rel_pos = np.zeros(0)
        # gripper_state = robot_qpos[-2:]
        # gripper_vel = (
        #     robot_qvel[-2:] * dt
        # )  # change to a scalar if the gripper is made symmetric
        #
        # if not self.has_object:
        #     achieved_goal = grip_pos.copy()
        # else:
        #     achieved_goal = np.squeeze(object_pos.copy())
        # obs = np.concatenate(
        #     [
        #         grip_pos,
        #         object_pos.ravel(),
        #         object_rel_pos.ravel(),
        #         gripper_state,
        #         object_rot.ravel(),
        #         object_velp.ravel(),
        #         object_velr.ravel(),
        #         grip_velp,
        #         gripper_vel,
        #     ]
        # )
        #
        # return {
        #     "observation": obs.copy(),
        #     "achieved_goal": achieved_goal.copy(),
        #     "desired_goal": self.goal.copy(),
        # }

        # positions
        grip_pos = self._utils.get_site_xpos(self.model, self.data, "robot0:grip")

        dt = self.n_substeps * self.model.opt.timestep
        grip_velp = (
                self._utils.get_site_xvelp(self.model, self.data, "robot0:grip") * dt
        )

        robot_qpos, robot_qvel = self._utils.robot_get_obs(
            self.model, self.data, self._model_names.joint_names
        )
        if self.has_object:
            object_pos = self._utils.get_site_xpos(self.model, self.data, "object0")
            # rotations
            object_rot = rotations.mat2euler(
                self._utils.get_site_xmat(self.model, self.data, "object0")
            )
            # velocities
            object_velp = (
                    self._utils.get_site_xvelp(self.model, self.data, "object0") * dt
            )
            object_velr = (
                    self._utils.get_site_xvelr(self.model, self.data, "object0") * dt
            )
            # gripper state
            object_rel_pos = object_pos - grip_pos
            object_velp -= grip_velp
        else:
            object_pos = (
                object_rot
            ) = object_velp = object_velr = object_rel_pos = np.zeros(0)
        gripper_state = robot_qpos[-2:]

        gripper_vel = (
                robot_qvel[-2:] * dt
        )  # change to a scalar if the gripper is made symmetric

        # return (
        #     grip_pos,
        #     object_pos,
        #     object_rel_pos,
        #     gripper_state,
        #     object_rot,
        #     object_velp,
        #     object_velr,
        #     grip_velp,
        #     gripper_vel,
        # )

        if not self.has_object:
            achieved_goal = grip_pos.copy()
        else:
            achieved_goal = np.squeeze(object_pos.copy())
        obs = np.concatenate(
            [
                grip_pos,
                object_pos.ravel(),
                object_rel_pos.ravel(),
                gripper_state,
                object_rot.ravel(),
                object_velp.ravel(),
                object_velr.ravel(),
                grip_velp,
                gripper_vel,
            ]
        )

        return {
            "observation": obs.copy(),
            "achieved_goal": achieved_goal.copy(),
            "desired_goal": self.goal.copy(),
        }

    def _get_gripper_xpos(self):
        body_id = self._model_names.body_name2id["robot0:gripper_link"]
        return self.data.xpos[body_id]

    def _render_callback(self):
        # # Visualize target.
        # sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        # site_id = self.sim.model.site_name2id("target0")
        # self.sim.model.site_pos[site_id] = self.goal - sites_offset[0]
        # self.sim.forward()

        # Visualize target.
        sites_offset = (self.data.site_xpos - self.model.site_pos).copy()
        site_id = self._mujoco.mj_name2id(
            self.model, self._mujoco.mjtObj.mjOBJ_SITE, "target0"
        )
        self.model.site_pos[site_id] = self.goal - sites_offset[0]
        self._mujoco.mj_forward(self.model, self.data)

    def _reset_sim(self):
        # self.sim.set_state(self.initial_state)
        #
        # # Randomize start position of object (by adding offsets to the gripper pos).
        # if self.has_object:
        #     if not self.fix_object:
        #         object_xpos = self.initial_gripper_xpos[:2]
        #         while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1:
        #             object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
        #     else:
        #         object_xpos = self.fixed_object
        #     object_qpos = self.sim.data.get_joint_qpos("object0:joint")
        #     assert object_qpos.shape == (7,)
        #     object_qpos[:2] = object_xpos
        #     self.sim.data.set_joint_qpos("object0:joint", object_qpos)
        #
        # self.sim.forward()
        # return True

        self.data.time = self.initial_time
        self.data.qpos[:] = np.copy(self.initial_qpos)
        self.data.qvel[:] = np.copy(self.initial_qvel)
        if self.model.na != 0:
            self.data.act[:] = None

        # Randomize start position of object.
        if self.has_object:
            if not self.fix_object:
                object_xpos = self.initial_gripper_xpos[:2]
                while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1:
                    object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(
                        -self.obj_range, self.obj_range, size=2
                    )
            else:
                object_xpos = self.fixed_object
            object_qpos = self._utils.get_joint_qpos(
                self.model, self.data, "object0:joint"
            )
            assert object_qpos.shape == (7,)
            object_qpos[:2] = object_xpos
            self._utils.set_joint_qpos(
                self.model, self.data, "object0:joint", object_qpos
            )

        self._mujoco.mj_forward(self.model, self.data)
        return True

    def forced_reset(self, state_dict):

        # # Set State
        # initial_state = state_dict['init_state']
        # self.sim.set_state(initial_state)
        # self._mujoco.mj_forward(self.model, self.data)
        #
        # # Set Goal
        # self.goal = state_dict['goal']
        # obs = self._get_obs()
        # return obs
        raise NotImplementedError

    def get_state_dict(self):
        # state = self.sim.get_state()
        # goal = self.goal
        # return {
        #     'init_state': state,
        #     'goal': goal
        # }
        raise NotImplementedError

    def _sample_goal(self):
        if not self.fix_goal:
            if self.has_object:
                goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(
                    -self.target_range, self.target_range, size=3
                )
                goal += self.target_offset
                goal[2] = self.height_offset
                if self.target_in_the_air and self.np_random.uniform() < 0.5:
                    goal[2] += self.np_random.uniform(0, 0.45)
            else:
                goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(
                    -self.target_range, self.target_range, size=3
                )
        else:
            goal = self.fixed_goal
        return goal.copy()

    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self._utils.set_joint_qpos(self.model, self.data, name, value)
        self._utils.reset_mocap_welds(self.model, self.data)
        self._mujoco.mj_forward(self.model, self.data)

        # Move end effector into position.
        gripper_target = np.array(
            [-0.498, 0.005, -0.431 + self.gripper_extra_height]
        ) + self._utils.get_site_xpos(self.model, self.data, "robot0:grip")
        gripper_rotation = np.array([1.0, 0.0, 1.0, 0.0])
        self._utils.set_mocap_pos(self.model, self.data, "robot0:mocap", gripper_target)
        self._utils.set_mocap_quat(self.model, self.data, "robot0:mocap", gripper_rotation)
        for _ in range(10):
            self._mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)

        # Extract information for sampling goals.
        self.initial_gripper_xpos = self._utils.get_site_xpos(self.model, self.data, "robot0:grip").copy()
        if self.has_object:
            self.height_offset = self._utils.get_site_xpos(self.model, self.data, "object0")[2]

    def render(self, mode="human", width=500, height=500):
        """Overrides the render method to support rendering with control."""
        self._render_callback()
        return self.mujoco_renderer.render(mode)


class FetchPickAndPlaceEnv(FetchEnv, EzPickle):
    """
        ## Description

        This environment was introduced in ["Multi-Goal Reinforcement Learning: Challenging Robotics Environments and Request for Research"](https://arxiv.org/abs/1802.09464).

        The task in the environment is for a manipulator to move a block to a target position on top of a table or in mid-air. The robot is a 7-DoF [Fetch Mobile Manipulator](https://fetchrobotics.com/) with a two-fingered parallel gripper.
        The robot is controlled by small displacements of the gripper in Cartesian coordinates and the inverse kinematics are computed internally by the MuJoCo framework. The gripper can be opened or closed in order to perform the graspping operation of pick and place.
        The task is also continuing which means that the robot has to maintain the block in the target position for an indefinite period of time.

        The control frequency of the robot is of `f = 25 Hz`. This is achieved by applying the same action in 20 subsequent simulator step (with a time step of `dt = 0.002 s`) before returning the control to the robot.

        ## Action Space

        The action space is a `Box(-1.0, 1.0, (4,), float32)`. An action represents the Cartesian displacement dx, dy, and dz of the end effector. In addition to a last action that controls closing and opening of the gripper.

        | Num | Action                                                 | Control Min | Control Max | Name (in corresponding XML file)                                | Joint | Unit         |
        | --- | ------------------------------------------------------ | ----------- | ----------- | --------------------------------------------------------------- | ----- | ------------ |
        | 0   | Displacement of the end effector in the x direction dx | -1          | 1           | robot0:mocap                                                    | hinge | position (m) |
        | 1   | Displacement of the end effector in the y direction dy | -1          | 1           | robot0:mocap                                                    | hinge | position (m) |
        | 2   | Displacement of the end effector in the z direction dz | -1          | 1           | robot0:mocap                                                    | hinge | position (m) |
        | 3   | Positional displacement per timestep of each finger of the gripper  | -1          | 1           | robot0:r_gripper_finger_joint and robot0:l_gripper_finger_joint | hinge | position (m) |

        ## Observation Space

        The observation is a `goal-aware observation space`. It consists of a dictionary with information about the robot's end effector state and goal. The kinematics observations are derived from Mujoco bodies known as [sites](https://mujoco.readthedocs.io/en/latest/XMLreference.html?highlight=site#body-site) attached to the body of interest such as the block or the end effector.
        Only the observations from the gripper fingers are derived from joints. Also to take into account the temporal influence of the step time, velocity values are multiplied by the step time dt=number_of_sub_steps*sub_step_time. The dictionary consists of the following 3 keys:

        * `observation`: its value is an `ndarray` of shape `(25,)`. It consists of kinematic information of the block object and gripper. The elements of the array correspond to the following:

        | Num | Observation                                                                                                                           | Min    | Max    | Site Name (in corresponding XML file) | Joint Name (in corresponding XML file) |Joint Type| Unit                     |
        |-----|---------------------------------------------------------------------------------------------------------------------------------------|--------|--------|---------------------------------------|----------------------------------------|----------|--------------------------|
        | 0   | End effector x position in global coordinates                                                                                         | -Inf   | Inf    | robot0:grip                           |-                                       |-         | position (m)             |
        | 1   | End effector y position in global coordinates                                                                                         | -Inf   | Inf    | robot0:grip                           |-                                       |-         | position (m)             |
        | 2   | End effector z position in global coordinates                                                                                         | -Inf   | Inf    | robot0:grip                           |-                                       |-         | position (m)             |
        | 3   | Block x position in global coordinates                                                                                                | -Inf   | Inf    | object0                               |-                                       |-         | position (m)             |
        | 4   | Block y position in global coordinates                                                                                                | -Inf   | Inf    | object0                               |-                                       |-         | position (m)             |
        | 5   | Block z position in global coordinates                                                                                                | -Inf   | Inf    | object0                               |-                                       |-         | position (m)             |
        | 6   | Relative block x position with respect to gripper x position in globla coordinates. Equals to x<sub>gripper</sub> - x<sub>block</sub> | -Inf   | Inf    | object0                               |-                                       |-         | position (m)             |
        | 7   | Relative block y position with respect to gripper y position in globla coordinates. Equals to y<sub>gripper</sub> - y<sub>block</sub> | -Inf   | Inf    | object0                               |-                                       |-         | position (m)             |
        | 8   | Relative block z position with respect to gripper z position in globla coordinates. Equals to z<sub>gripper</sub> - z<sub>block</sub> | -Inf   | Inf    | object0                               |-                                       |-         | position (m)             |
        | 9   | Joint displacement of the right gripper finger                                                                                        | -Inf   | Inf    |-                                      | robot0:r_gripper_finger_joint          | hinge    | position (m)             |
        | 10  | Joint displacement of the left gripper finger                                                                                         | -Inf   | Inf    |-                                      | robot0:l_gripper_finger_joint          | hinge    | position (m)             |
        | 11  | Global x rotation of the block in a XYZ Euler frame rotation                                                                          | -Inf   | Inf    | object0                               |-                                       |-         | angle (rad)              |
        | 12  | Global y rotation of the block in a XYZ Euler frame rotation                                                                          | -Inf   | Inf    | object0                               |-                                       |-         | angle (rad)              |
        | 13  | Global z rotation of the block in a XYZ Euler frame rotation                                                                          | -Inf   | Inf    | object0                               |-                                       |-         | angle (rad)              |
        | 14  | Relative block linear velocity in x direction with respect to the gripper                                                              | -Inf   | Inf    | object0                               |-                                       |-         | velocity (m/s)           |
        | 15  | Relative block linear velocity in y direction with respect to the gripper                                                              | -Inf   | Inf    | object0                               |-                                       |-         | velocity (m/s)           |
        | 16  | Relative block linear velocity in z direction                                                                                         | -Inf   | Inf    | object0                               |-                                       |-         | velocity (m/s)           |
        | 17  | Block angular velocity along the x axis                                                                                               | -Inf   | Inf    | object0                               |-                                       |-         | angular velocity (rad/s) |
        | 18  | Block angular velocity along the y axis                                                                                               | -Inf   | Inf    | object0                               |-                                       |-         | angular velocity (rad/s) |
        | 19  | Block angular velocity along the z axis                                                                                               | -Inf   | Inf    | object0                               |-                                       |-         | angular velocity (rad/s) |
        | 20  | End effector linear velocity x direction                                                                                              | -Inf   | Inf    | robot0:grip                           |-                                       |-         | velocity (m/s)           |
        | 21  | End effector linear velocity y direction                                                                                              | -Inf   | Inf    | robot0:grip                           |-                                       |-         | velocity (m/s)           |
        | 22  | End effector linear velocity z direction                                                                                              | -Inf   | Inf    | robot0:grip                           |-                                       |-         | velocity (m/s)           |
        | 23  | Right gripper finger linear velocity                                                                                                  | -Inf   | Inf    |-                                      | robot0:r_gripper_finger_joint          | hinge    | velocity (m/s)           |
        | 24  | Left gripper finger linear velocity                                                                                                   | -Inf   | Inf    |-                                      | robot0:l_gripper_finger_joint          | hinge    | velocity (m/s)           |

        * `desired_goal`: this key represents the final goal to be achieved. In this environment it is a 3-dimensional `ndarray`, `(3,)`, that consists of the three cartesian coordinates of the desired final block position `[x,y,z]`. In order for the robot to perform a pick and place trajectory, the goal position can be elevated over the table or on top of the table. The elements of the array are the following:

        | Num | Observation                                                                                                                           | Min    | Max    | Site Name (in corresponding XML file) |Unit          |
        |-----|---------------------------------------------------------------------------------------------------------------------------------------|--------|--------|---------------------------------------|--------------|
        | 0   | Final goal block position in the x coordinate                                                                                         | -Inf   | Inf    | target0                               | position (m) |
        | 1   | Final goal block position in the y coordinate                                                                                         | -Inf   | Inf    | target0                               | position (m) |
        | 2   | Final goal block position in the z coordinate                                                                                         | -Inf   | Inf    | target0                               | position (m) |

        * `achieved_goal`: this key represents the current state of the block, as if it would have achieved a goal. This is useful for goal orientated learning algorithms such as those that use [Hindsight Experience Replay](https://arxiv.org/abs/1707.01495) (HER). The value is an `ndarray` with shape `(3,)`. The elements of the array are the following:

        | Num | Observation                                                                                                                           | Min    | Max    | Site Name (in corresponding XML file) |Unit          |
        |-----|---------------------------------------------------------------------------------------------------------------------------------------|--------|--------|---------------------------------------|--------------|
        | 0   | Current block position in the x coordinate                                                                                            | -Inf   | Inf    | object0                               | position (m) |
        | 1   | Current block position in the y coordinate                                                                                            | -Inf   | Inf    | object0                               | position (m) |
        | 2   | Current block position in the z coordinate                                                                                            | -Inf   | Inf    | object0                               | position (m) |


        ## Rewards

        The reward can be initialized as `sparse` or `dense`:
        - *sparse*: the returned reward can have two values: `-1` if the block hasn't reached its final target position, and `0` if the block is in the final target position (the block is considered to have reached the goal if the Euclidean distance between both is lower than 0.05 m).
        - *dense*: the returned reward is the negative Euclidean distance between the achieved goal position and the desired goal.

        To initialize this environment with one of the mentioned reward functions the type of reward must be specified in the id string when the environment is initialized. For `sparse` reward the id is the default of the environment, `FetchPickAndPlace-v2`. However, for `dense` reward the id must be modified to `FetchPickAndPlaceDense-v2` and initialized as follows:

        ```python
        import gymnasium as gym

        env = gym.make('FetchPickAndPlaceDense-v2')
        ```

        ## Starting State

        When the environment is reset the gripper is placed in the following global cartesian coordinates `(x,y,z) = [1.3419 0.7491 0.555] m`, and its orientation in quaternions is `(w,x,y,z) = [1.0, 0.0, 1.0, 0.0]`. The joint positions are computed by inverse kinematics internally by MuJoCo. The base of the robot will always be fixed at `(x,y,z) = [0.405, 0.48, 0]` in global coordinates.

        The block's position has a fixed height of `(z) = [0.42] m ` (on top of the table). The initial `(x,y)` position of the block is the gripper's x and y coordinates plus an offset sampled from a uniform distribution with a range of `[-0.15, 0.15] m`. Offset samples are generated until the 2-dimensional Euclidean distance from the gripper to the block is greater than `0.1 m`.
        The initial orientation of the block is the same as for the gripper, `(w,x,y,z) = [1.0, 0.0, 1.0, 0.0]`.

        Finally the target position where the robot has to move the block is generated. The target can be in mid-air or over the table. The random target is also generated by adding an offset to the initial grippers position `(x,y)` sampled from a uniform distribution with a range of `[-0.15, 0.15] m`.
        The height of the target is initialized at `(z) = [0.42] m ` and an offset is added to it sampled from another uniform distribution with a range of `[0, 0.45] m`.


        ## Episode End

        The episode will be `truncated` when the duration reaches a total of `max_episode_steps` which by default is set to 50 timesteps.
        The episode is never `terminated` since the task is continuing with infinite horizon.

        ## Arguments

        To increase/decrease the maximum number of timesteps before the episode is `truncated` the `max_episode_steps` argument can be set at initialization. The default value is 50. For example, to increase the total number of timesteps to 100 make the environment as follows:

        ```python
        import gymnasium as gym

        env = gym.make('FetchPickAndPlace-v2', max_episode_steps=100)
        ```

        ## Version History

        * v2: the environment depends on the newest [mujoco python bindings](https://mujoco.readthedocs.io/en/latest/python.html) maintained by the MuJoCo team in Deepmind.
        * v1: the environment depends on `mujoco_py` which is no longer maintained.
        """

    def __init__(self, reward_type='sparse', distance_threshold=0.05, fix_object=False, fix_goal=False):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
        }
        FetchEnv.__init__(
            self, MODEL_XML_PATH, has_object=True, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=distance_threshold,
            initial_qpos=initial_qpos, reward_type=reward_type,
            fix_object=fix_object, fix_goal=fix_goal
        )
        EzPickle.__init__(self)
