import time
import numpy as np
from typing import Tuple
import collections
import pickle
import tensorflow as tf

ACTION_TO_LATENT_MAPPING = {
    'pick:obj0': 1,
    'drop:obj0': 2,
    'pick:obj1': 3,
    'drop:obj1': 4,
    'stay': 0
}


def Step(observation, reward, done, **kwargs):
    """
    Convenience method creating a namedtuple with the results of the
    environment's step method.
    Put extra diagnostic info in the kwargs
    """
    _Step = collections.namedtuple("Step", ["observation", "reward", "done", "info"])
    return _Step(observation, reward, done, kwargs)


def clip_action(ac):
    return np.clip(ac, -1, 1)


class PnPEnv(object):
    def __init__(self, full_space_as_goal=False, goal_weight=1., terminal_eps=0.01, feasible_hand=True, two_obj=False,
                 first_in_place=False, stacking=False, target_in_the_air=True, fix_goal=False):
        """
        Pick and Place Environment: can be single or multi-object
        Args:
            goal_weight:
            terminal_eps:
            full_space_as_goal: Whether to add gripper position [:3] to the goal space
            feasible_hand:
            two_obj:
            first_in_place:
            stacking:
            fix_goal:
        """

        if not two_obj:
            from .Fetch_Base.fetch_env_oneobj import FetchPickAndPlaceEnv
            env = FetchPickAndPlaceEnv()
        else:
            from .Fetch_Base.fetch_env_twoobj import FetchPickAndPlaceEnv
            env = FetchPickAndPlaceEnv(stacking=stacking, first_in_place=first_in_place,
                                       target_in_the_air=target_in_the_air)

        env.unwrapped.spec = self
        self._env = env

        self._observation_space = env.observation_space.spaces['observation']
        self._action_space = env.action_space

        self._current_goal = None

        self.goal_weight = goal_weight
        self.terminal_eps = terminal_eps
        self.full_space_as_goal = full_space_as_goal
        self.two_obj = two_obj

        self.feasible_hand = feasible_hand  # if the position of the hand is always feasible to achieve

        self.fix_goal = fix_goal
        if fix_goal:
            self.fixed_goal = np.array([1.48673746, 0.69548325, 0.6])
            
        if two_obj:
            self.latent_dim = 5
        else:
            self.latent_dim = 3

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def current_goal(self):
        return self._current_goal

    def reset(self):
        d = self._env.reset()
        # Update the goal based on some checks
        self.update_goal(d=d)
        return self._transform_obs(d['observation'])
    
    def forced_reset(self, state_dict):
        d = self._env.forced_reset(state_dict)
        # Update the goal based on some checks
        self.update_goal(d=d)
        return self._transform_obs(d['observation'])
    
    def get_state_dict(self):
        state_dict = self._env.get_state_dict()
        # tf.print("PnPEnv: {}".format(state_dict['goal']))
        return state_dict

    def step(self, action):
        next_obs, reward, _, info = self._env.step(
            action)  # FetchPickAndPlaceEnv freezes done to False and stores termination response in info
        next_obs = self._transform_obs(next_obs['observation'])  # Remove unwanted portions of the observed state
        info['obs2goal'] = self.transform_to_goal_space(next_obs)
        info['distance'] = np.linalg.norm(self.current_goal - info['obs2goal'])
        if self.full_space_as_goal:
            info['block_distance'] = np.linalg.norm((self.current_goal - info['obs2goal'])[3:6])
            info['hand_distance'] = np.linalg.norm((self.current_goal - info['obs2goal'])[0:3])
        info['goal_reached'] = info['distance'] < self.terminal_eps
        done = info['goal_reached']
        return Step(next_obs, reward, done, **info)

    def get_current_obs(self):
        """
        :return: current observation (state of the robot)
        """
        return self._transform_obs(self._env._get_obs()['observation'])

    def transform_to_goal_space(self, obs):
        """
            Transform the observation to the goal space by extracting the achieved goal from the observation
            For the PnP, it corresponds to obj. positions
        """
        if not self.full_space_as_goal:
            ret = np.array(obs[3:6])
        else:
            ret = np.array(obs[:6])
        if self.two_obj:
            ret = np.concatenate([ret, obs[6:9]])
        return ret

    def render(self):
        self._env.render()

    def _transform_obs(self, obs):
        """
            Extract the relevant information from the observation
        """
        if self.two_obj:
            return obs[:16]
        else:
            return obs[:10]

    def sample_hand_pos(self, block_pos):
        if block_pos[2] == self._env.height_offset or not self.feasible_hand:
            xy = self._env.initial_gripper_xpos[:2] + np.random.uniform(-0.15, 0.15, size=2)
            z = np.random.uniform(self._env.height_offset, self._env.height_offset + 0.3)
            return np.concatenate([xy, [z]])
        else:
            return block_pos

    def update_goal(self, d=None):
        """
            Set the goal for the env. in the current episode
        """
        # Fix objects
        if self.get_current_obs()[5] < self._env.height_offset or \
                np.any(self.get_current_obs()[3:5] > self._env.initial_gripper_xpos[:2] + 0.15) or \
                np.any(self.get_current_obs()[3:5] < self._env.initial_gripper_xpos[:2] - 0.15):
            self._env._reset_sim()

        # Fix Goals
        if self.fix_goal:
            self._current_goal = self.fixed_goal
        else:
            
            if d is not None:
                self._current_goal = d['desired_goal']
            else:
                self._current_goal = self._env.goal = np.copy(self._env._sample_goal())
            
            if self.full_space_as_goal:
                self._current_goal = np.concatenate([self.sample_hand_pos(self._current_goal), self._current_goal])

    def set_feasible_hand(self, bool):
        self.feasible_hand = bool
        
    def get_init_latent_mode(self):
        # TODO: Change this for two-object case
        init_latent_mode = ACTION_TO_LATENT_MAPPING['pick:obj0']
        init_latent_mode = tf.one_hot(init_latent_mode, depth=self.latent_dim, dtype=tf.float32)
        init_latent_mode = tf.squeeze(init_latent_mode)
        return init_latent_mode


class MyPnPEnvWrapperForGoalGAIL(PnPEnv):
    def __init__(self, full_space_as_goal=False, **kwargs):
        """
        GoalGAIL compatible Wrapper for PnP Env
        Args:
            full_space_as_goal:
            **kwargs:
        """
        super(MyPnPEnvWrapperForGoalGAIL, self).__init__(full_space_as_goal, **kwargs)

    def reset(self, render=False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        obs = super(MyPnPEnvWrapperForGoalGAIL, self).reset()
        if render:
            super(MyPnPEnvWrapperForGoalGAIL, self).render()
        achieved_goal = super(MyPnPEnvWrapperForGoalGAIL, self).transform_to_goal_space(obs)
        desired_goal = super(MyPnPEnvWrapperForGoalGAIL, self).current_goal
        return obs.astype(np.float32), achieved_goal.astype(np.float32), desired_goal.astype(np.float32)

    def step(self, action, render=False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if len(action.shape) > 1:
            action = action[0]
        obs, _, done, info = super(MyPnPEnvWrapperForGoalGAIL, self).step(action)  # ignore reward (re-computed in HER)

        if render:
            super(MyPnPEnvWrapperForGoalGAIL, self).render()
        
        achieved_goal = super(MyPnPEnvWrapperForGoalGAIL, self).transform_to_goal_space(obs)
        desired_goal = super(MyPnPEnvWrapperForGoalGAIL, self).current_goal
        success = int(done)
        
        return obs.astype(np.float32), achieved_goal.astype(np.float32), desired_goal.astype(np.float32), np.array(success, np.int32), info['distance'].astype(np.float32)

    def forced_reset(self, state_dict, render=False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        obs = super(MyPnPEnvWrapperForGoalGAIL, self).forced_reset(state_dict)
        if render:
            super(MyPnPEnvWrapperForGoalGAIL, self).render()
        achieved_goal = super(MyPnPEnvWrapperForGoalGAIL, self).transform_to_goal_space(obs)
        desired_goal = super(MyPnPEnvWrapperForGoalGAIL, self).current_goal
        return obs.astype(np.float32), achieved_goal.astype(np.float32), desired_goal.astype(np.float32)
    
    def get_state_dict(self):
        state_dict = super(MyPnPEnvWrapperForGoalGAIL, self).get_state_dict()
        # tf.print("MyPnPEnvWrapperForGoalGAIL: {}".format(state_dict['goal']))
        return state_dict
        
    def reward_fn(self, ag_2, g, o, relative_goal=False, distance_metric='L1', only_feasible=False,
                  extend_dist_rew_weight=0.):
        """
            Custom reward function with two components:
                (a) 0/1 for whether goal is reached (weighed by self.goal_weight),
                (b) relative distance between achieved goal and desired goal (weighed by extend_dist_rew_weight)
        """
        
        if relative_goal:
            dif = o[:, -2:]
        else:
            dif = ag_2 - g
        
        if distance_metric == 'L1':
            goal_distance = np.linalg.norm(dif, ord=1, axis=-1)
        elif distance_metric == 'L2':
            goal_distance = np.linalg.norm(dif, ord=2, axis=-1)
        elif callable(distance_metric):
            goal_distance = distance_metric(ag_2, g)
        else:
            raise NotImplementedError('Unsupported distance metric type.')

        # if only_feasible:
        #     ret = np.logical_and(goal_distance < self.terminal_eps, [self.is_feasible(g_ind) for g_ind in
        #                                                              g]) * self.goal_weight - \
        #           extend_dist_rew_weight * goal_distance
        # else:
        ret = (goal_distance < self.terminal_eps) * self.goal_weight - extend_dist_rew_weight * goal_distance

        return ret


class PnPExpert:
    def __init__(self, env, full_space_as_goal=False, noise_eps=0.0, random_eps=0.0):
        self.env = env
        self.step_size = 6
        self.latent_dim =  env.latent_dim
        self.full_space_as_goal = full_space_as_goal

        self.reset()

    def act(self, state, achieved_goal, goal, noise_eps=0., random_eps=0., compute_c=True, **kwargs):
        # a, latent_mode = self.get_action(state[0])
        # a = self.add_noise_to_action(a, noise_eps, random_eps)

        a, latent_mode = tf.numpy_function(func=self.get_action, inp=[state[0]], Tout=[tf.float32, tf.int32])
        a = tf.numpy_function(func=self.add_noise_to_action, inp=[a, noise_eps, random_eps], Tout=tf.float32)
        a = tf.squeeze(a)
        
        latent_mode = tf.one_hot(latent_mode, depth=self.latent_dim, dtype=tf.float32)
        latent_mode = tf.squeeze(latent_mode)
        if compute_c:
            return a, latent_mode
        else:
            return a

    def get_action(self, o):
        gripper_pos = o[:3]
        block_pos = o[3:6]
        goal_pos = self.env.current_goal[-3:]
        block_rel_pos = o[6:9]  # block_pos - gripper_pos
        block_target = block_rel_pos + np.array([0, 0, 0.03])
        # print("---- #  |{}| -> |{}| # ----".format(np.linalg.norm(block_rel_pos), np.linalg.norm(goal_pos-block_pos)))

        if np.linalg.norm(block_rel_pos) > 0.05 and np.linalg.norm(block_rel_pos - np.array([0, 0, 0.03])) > 0.001:
            if self.block0_picked:
                self.block0_picked = False
            ac, ac_type = self.goto_block(gripper_pos, block_pos)

        elif np.linalg.norm(block_rel_pos) > 0.01 and not self.block0_picked:
            ac, ac_type = self.pickup_block(gripper_pos, block_pos)

        else:
            self.block0_picked = True
            ac, ac_type = self.goto_goal(block_pos, goal_pos)

        return ac, np.array([ACTION_TO_LATENT_MAPPING[ac_type]], dtype=np.int32)

    def reset(self):
        #  # Hack: This flag helps bypass the irregular behaviour i.e. when block is picked and
        # gripper starts to move towards the goal, object's pos relative to gripper increases which then violates the
        # move-to-goal condition 'np.linalg.norm(block_rel_pos) < 0.01'(see orig. implementation of GoalGAIL).
        # This behaviour is periodic making the gripper oscillate
        self.block0_picked = False

    def goto_block(self, cur_pos, block_pos):
        target_pos = block_pos + np.array([0, 0, 0.03])
        ac = clip_action((target_pos - cur_pos) * self.step_size)
        ac = np.concatenate([ac, np.array([1])], dtype=np.float32)
        ac_type = 'pick:obj0'
        # print(" [[Goto Block]]: ", ac)
        return ac, ac_type

    def pickup_block(self, cur_pos, block_pos, ):
        block_rel_pos = block_pos - cur_pos

        if np.linalg.norm(block_rel_pos) < 0.01:
            ac = np.array([0, 0, 0, -1.], dtype=np.float32)
            ac_type = 'stay'
            # print(" [[Pick (keep holding) Block]]: ", ac)
        else:
            ac = clip_action(block_rel_pos * self.step_size)
            ac = np.concatenate([ac, np.array([-0.005])], dtype=np.float32)
            # print(" [[Pick (sub goto) Block]]: ", ac)
            ac_type = 'pick:obj0'
        return ac, ac_type

    def goto_goal(self, cur_pos, goal_pos, grip=-1.):
        if np.linalg.norm((goal_pos - cur_pos)) > 0.01:
            ac = clip_action((goal_pos - cur_pos) * self.step_size)
            ac = np.concatenate([ac, np.array([grip])], dtype=np.float32)
            # print(" [[Goto Goal]]: ", ac)
            ac_type = 'drop:obj0'
        else:
            ac = np.array([0, 0, 0, grip], dtype=np.float32)
            # print(" [[Stay]]: ", ac)
            ac_type = 'stay'
        return ac, ac_type

    def add_noise_to_action(self, a, noise_eps=0., random_eps=0.,):
        a = np.array([a])
        noise = noise_eps * np.random.randn(*a.shape)  # gaussian noise
        a += noise
        a = np.clip(a, -1, 1)
        a += np.random.binomial(1, random_eps, a.shape[0]).reshape(-1, 1) * (self._random_action(a.shape[0]) - a)
        return a

    def _random_action(self, n):
        return np.random.uniform(low=-1, high=1, size=(n, 4))


class PnPExpertTwoObj:
    def __init__(self, env, step_size=0.03, full_space_as_goal=False, expert_behaviour: str = '0'):
        self.env = env
        self.step_size = 6
        self.full_space_as_goal = full_space_as_goal
        self.latent_dim = env.latent_dim

        # Thresholds
        self.sub_goal_height = 0.55  # Height to which block will be first taken before moving towards goal.
        # This should be greater than block's height
        self.dataset2_thresh = 0.7  # The probability of picking the obj0 first
        self.switch_prob = 0.05  # The probability of switching mid-demo
        self.num_do_switch: int = 0  # Number of times to switch before demo ends [DO NOT CHANGE HERE]

        self.expert_behaviour: str = expert_behaviour  # One of ['0', '1', '2']
        self.reset()

    def act(self, state, achieved_goal, goal, noise_eps=0., random_eps=0., compute_c=True, **kwargs):

        a, latent_mode = tf.numpy_function(func=self.get_action, inp=[state[0]], Tout=[tf.float32, tf.int32])
        a = tf.numpy_function(func=self.add_noise_to_action, inp=[a, noise_eps, random_eps], Tout=tf.float32)
        a = tf.squeeze(a)
        
        latent_mode = tf.one_hot(latent_mode, depth=self.latent_dim, dtype=tf.float32)
        latent_mode = tf.squeeze(latent_mode)
        if compute_c:
            return a, latent_mode
        else:
            return a

    def get_action(self, o) -> Tuple[np.ndarray, np.ndarray]:
        time.sleep(0.01)

        gripper_pos = o[:3]
        block_pos0 = o[3:6]
        block_pos1 = o[6:9]
        relative_block_pos0 = o[9:12]
        relative_block_pos1 = o[12:15]
        current_goal_block0 = self.env.current_goal[-6:-3]
        current_goal_block1 = self.env.current_goal[-3:]

        # Do Switch ????
        switch = False
        # To change expert's behaviour mid-trajectory
        if self.num_do_switch:
            # Do switch with 5% prob.
            # For values like 10, 20, 50%, we are always switching before any obj is being picked, thus dec. the value
            switch = True if np.random.uniform(0, 1) < self.switch_prob else False

        if switch:
            # print("<---------------> Switching")

            self.num_do_switch -= 1

            # Case 1: When none of the blocks have been sent towards their goal
            if not self.block1_picked and not self.block0_picked:

                # print("Case1: None Pick")

                if self.picking_order == 'zero_first':
                    self.picking_order = 'one_first'
                else:
                    self.picking_order = 'zero_first'

                # Check if gripper is holding any block (we set block_picked after a block is gripped)
                if np.linalg.norm(relative_block_pos0) < 0.01:
                    # print("Release Block 0")
                    # Action = RELEASE
                    ac = np.array([gripper_pos[0], gripper_pos[1], 1., 1.], dtype=np.float32)
                    ac_type = 'pick:obj1'
                    return ac, np.array([ACTION_TO_LATENT_MAPPING[ac_type]], dtype=np.int32)

                elif np.linalg.norm(relative_block_pos1) < 0.01:
                    # print("Release Block 1")
                    # Action = RELEASE
                    ac = np.array([gripper_pos[0], gripper_pos[1], 1., 1.], dtype=np.float32)
                    ac_type = 'pick:obj0'
                    return ac, np.array([ACTION_TO_LATENT_MAPPING[ac_type]], dtype=np.int32)

        def deal_with_block(block_pos, relative_block_pos, current_goal_block, block_picked, tempGoal_reached):
            if np.linalg.norm(relative_block_pos) > 0.1 \
                    and np.linalg.norm(relative_block_pos - np.array([0, 0, 0.08])) > 0.001:
                _ac, ac_type = self.goto_block(gripper_pos, block_pos)
            elif np.linalg.norm(relative_block_pos) > 0.01 and not block_picked:
                _ac, ac_type = self.pickup_block(gripper_pos, block_pos)
            else:
                block_picked = True
                # Collision avoid: Move the block vertically up first
                if not tempGoal_reached:
                    sub_goal = np.concatenate([block_pos[:2], np.array([self.sub_goal_height])])
                    _ac = clip_action((sub_goal - block_pos) * self.step_size)
                    _ac = np.concatenate([_ac, np.array([-1])], dtype=np.float32)

                    if np.linalg.norm((sub_goal - block_pos)) < 0.05:  # Use distance thresh of FetchEnv
                        tempGoal_reached = True

                    ac_type = 'drop:obj'
                else:
                    _ac, ac_type = self.goto_goal(block_pos, current_goal_block)
            return _ac, ac_type, block_picked, tempGoal_reached

        # if the first and second block are placed in the right place
        # Earlier logic of this `if' branch was only checking if block 1 was at its goal, this sometimes led to
        # misbehaviour in cases when block 1 was at its goal but not block 0.
        # Thus, we need to make sure that all the blocks are at their goals

        if self.picking_order == 'zero_first':

            if np.linalg.norm(block_pos1 - current_goal_block1) < 0.01 and \
                    np.linalg.norm(block_pos0 - current_goal_block0) < 0.01:

                # print(" [[Stay]] ")
                # if self.full_space_as_goal:
                #     if np.linalg.norm(relative_block_pos1) < 0.04:
                #         ac = np.array([0., 0., 1., 1.])
                #     else:
                #         ac = self.goto_goal(gripper_pos, self.env.current_goal[:3], grip=1)
                # else:
                ac = np.array([0., 0., 0., -1.], dtype=np.float32)
                ac_type = 'stay'

            # Deal with block 1
            elif np.linalg.norm(block_pos0 - current_goal_block0) < 0.01:

                if np.linalg.norm(relative_block_pos0) < 0.1 and np.linalg.norm(relative_block_pos0[:2]) < 0.02:
                    # print(" [[Move up]] ")  # Moving up before going towards next block
                    ac = np.array([0, 0, 1., 1.], dtype=np.float32)
                    ac_type = 'pick:obj1'
                else:
                    # print(" [[PnP Block1]] ")
                    ac, ac_type, self.block1_picked, self.tempGoal1_reached = deal_with_block(block_pos1,
                                                                                              relative_block_pos1,
                                                                                              current_goal_block1,
                                                                                              self.block1_picked,
                                                                                              self.tempGoal1_reached)
                    ac_type = ac_type + '1' if ac_type != 'stay' else ac_type

            # Deal with block 0
            else:

                # print(" [[PnP Block0]] ")
                ac, ac_type, self.block0_picked, self.tempGoal0_reached = deal_with_block(block_pos0,
                                                                                          relative_block_pos0,
                                                                                          current_goal_block0,
                                                                                          self.block0_picked,
                                                                                          self.tempGoal0_reached)
                ac_type = ac_type + '0' if ac_type != 'stay' else ac_type

        else:

            if np.linalg.norm(block_pos1 - current_goal_block1) < 0.01 and \
                    np.linalg.norm(block_pos0 - current_goal_block0) < 0.01:
                # print(" [[Stay]] ")
                # if self.full_space_as_goal:
                #     if np.linalg.norm(relative_block_pos1) < 0.04:
                #         ac = np.array([0., 0., 1., 1.])
                #     else:
                #         ac = self.goto_goal(gripper_pos, self.env.current_goal[:3], grip=1)
                # else:
                ac = np.array([0., 0., 0., -1.], dtype=np.float32)
                ac_type = 'stay'

            # Deal with block 0
            elif np.linalg.norm(block_pos1 - current_goal_block1) < 0.01:
                if np.linalg.norm(relative_block_pos1) < 0.1 and np.linalg.norm(relative_block_pos1[:2]) < 0.02:
                    # print(" [[Move up]] ")  # Moving up before going towards next block
                    ac = np.array([0, 0, 1., 1.], dtype=np.float32)
                    ac_type = 'pick:obj0'
                else:
                    # print(" [[PnP Block0]] ")
                    ac, ac_type, self.block0_picked, self.tempGoal0_reached = deal_with_block(block_pos0,
                                                                                              relative_block_pos0,
                                                                                              current_goal_block0,
                                                                                              self.block0_picked,
                                                                                              self.tempGoal0_reached)
                    ac_type = ac_type + '0' if ac_type != 'stay' else ac_type

            # Deal with block 1
            else:
                # print(" [[PnP Block1]] ")
                ac, ac_type, self.block1_picked, self.tempGoal1_reached = deal_with_block(block_pos1,
                                                                                          relative_block_pos1,
                                                                                          current_goal_block1,
                                                                                          self.block1_picked,
                                                                                          self.tempGoal1_reached)
                ac_type = ac_type + '1' if ac_type != 'stay' else ac_type

        return ac, np.array([ACTION_TO_LATENT_MAPPING[ac_type]], dtype=np.int32)

    def reset(self):
        self.block0_picked, self.block1_picked = False, False

        # Setting the sub_goal flags to False will encourage expert to first move vertically up
        # before moving towards goal
        self.tempGoal0_reached, self.tempGoal1_reached = False, False

        # DATASET 1: For fixed picking order (in stacking the env is rendered with goal0 on table, thus use zero_first)
        if self.expert_behaviour == '0':
            self.picking_order: str = 'zero_first'

        else:

            # DATASET 2: For random picking order (70-30 Split)
            if np.random.uniform(0, 1) < self.dataset2_thresh:
                self.picking_order: str = 'zero_first'
            else:
                self.picking_order: str = 'one_first'

            # DATASET 3: When the picking order changes mid-demo
            # To change the behaviour mid-demo
            if self.expert_behaviour == '1':
                self.num_do_switch: int = 0
            elif self.expert_behaviour == '2':
                self.num_do_switch: int = 1
            else:
                print("PICK Expert Behaviour from '0', '1', '2'!!")
                raise NotImplementedError

        print("[INITIAL] Picking: ", self.picking_order)

    def goto_block(self, cur_pos, block_pos):
        target_pos = block_pos + np.array([0, 0, 0.08])
        ac = clip_action((target_pos - cur_pos) * self.step_size)
        ac_type = 'pick:obj'
        return np.concatenate([ac, np.array([1])], dtype=np.float32), ac_type

    def pickup_block(self, cur_pos, block_pos):
        if np.linalg.norm(cur_pos - block_pos) < 0.01:  # and gripper_state > 0.025: # TODO: need to adjust
            ac = np.array([0, 0, 0, -1.], dtype=np.float32)
            ac_type = 'stay'
        else:
            ac = clip_action((block_pos - cur_pos) * self.step_size)
            ac = np.concatenate([ac, np.array([-0.005])], dtype=np.float32)
            ac_type = 'pick:obj'
        return ac, ac_type

    def goto_goal(self, cur_pos, goal_pos, grip=-1):

        # Collision avoid: We first move the gripper towards (x, y) coordinates of goal maintaining the gripper height
        if np.linalg.norm((goal_pos[:2] - cur_pos[:2])) > 0.01:
            ac = clip_action((goal_pos[:2] - cur_pos[:2]) * self.step_size)
            ac = np.concatenate([ac, np.array([0]), np.array([grip])], dtype=np.float32)
            ac_type = 'drop:obj'

        # We then move the gripper towards the goal -> Doing this after above action brings gripper vertically down
        elif np.linalg.norm((goal_pos - cur_pos)) > 0.01:
            ac = clip_action((goal_pos - cur_pos) * self.step_size)
            ac = np.concatenate([ac, np.array([grip])], dtype=np.float32)
            ac_type = 'drop:obj'

        else:
            ac = np.array([0, 0, 0, grip], dtype=np.float32)
            ac_type = 'stay'
        return ac, ac_type

    def add_noise_to_action(self, a, noise_eps=0., random_eps=0.,):
        a = np.array([a])
        noise = noise_eps * np.random.randn(*a.shape)  # gaussian noise
        a += noise
        a = np.clip(a, -1, 1)
        a += np.random.binomial(1, random_eps, a.shape[0]).reshape(-1, 1) * (self._random_action(a.shape[0]) - a)
        return a

    def _random_action(self, n):
        return np.random.uniform(low=-1, high=1, size=(n, 4))
