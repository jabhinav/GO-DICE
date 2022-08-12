import os
import pickle as pkl
import gym
import numpy as np
from utils.misc import one_hot_encode
from utils.mpi import Normalizer


def get_env_params(env):
    obs = env.reset()
    params = {'obs': obs['observation'].shape[0], 'goal': obs['desired_goal'].shape[0],
              'action': env.action_space.shape[0], 'action_max': env.action_space.high[0],  # Highest (lowest) values
              'max_timesteps': env._max_episode_steps}
    return params


class FetchPickAndPlace:
    def __init__(self, data_dir=''):
        self.data_dir = data_dir
        self.env = gym.make('FetchPickAndPlace-v1')
        self.env.reset()
        self.params = get_env_params(self.env)
        self.clip_range = 5.

    def save_demos(self, actions, observations, states, infos, latent_states, num_demos=100, initStateSpace="random"):
        # Save Data as a dict
        data = {
            's': np.array(states),
            'os': observations,
            'a': np.array(actions),
            'c': np.array(latent_states),
            'top_c': np.array(latent_states),
            'infos': infos
        }

        with open(os.path.join(self.data_dir, "stateaction_latent_dynamic_test.pkl"), 'wb') as o:
            pkl.dump(data, o, protocol=2)
        # fileName = "data_fetch_{}_{}.npz".format(initStateSpace, num_demos)
        # np.savez_compressed(fileName, acs=actions, obs=observations, c=latent_states, info=infos)  # save the file

    def generate_demos(self, num_demos, save=True, verbose=0):
        actions = []
        observations = []  # Dictionary containing agent's state, desired_goal and achieved_goal
        states = []  # Concatenated array of agent's state and desired goal position
        infos = []  # Contains dictionary with information keys like 'is_success', 'TimeLimit.truncated'
        latent_states = []

        per_traj_pick_std = []
        per_traj_drop_std = []

        # Generate Demos
        while len(actions) < num_demos:
            obs = self.env.reset()
            if verbose:
                self.env.render()
            print("ITERATION NUMBER ", len(actions))
            # Collect Demo
            episodeAcs, episodeObs, episodeStates, episodeInfo, episodeLatentStates = self.pick_and_go_to_goal(self.env,
                                                                                                               obs,
                                                                                                               verbose)
            actions.append(np.array(episodeAcs))
            observations.append(episodeObs)
            states.append(np.array(episodeStates))
            infos.append(episodeInfo)

            # Compute Stats
            transn_idx = 0
            for i in range(len(episodeLatentStates)-1):
                if episodeLatentStates[i] != episodeLatentStates[i+1]:
                    transn_idx = i+1
            per_traj_pick_std.append(np.std(episodeAcs[:transn_idx], axis=0))
            per_traj_drop_std.append(np.std(episodeAcs[transn_idx:], axis=0))

            episodeLatentStates = [one_hot_encode(np.array(c, dtype=np.int), dim=2)[0] for c in episodeLatentStates]
            latent_states.append(np.array(episodeLatentStates))

        # Display stats
        # print("Std Deviation for actions across pick-traj: ", per_traj_pick_std)
        print("Mean std dev (pick-traj): ", np.mean(per_traj_pick_std, axis=0))

        # print("Std Deviation for actions across drop-traj: ", per_traj_drop_std)
        print("Mean std dev (drop-traj): ", np.mean(per_traj_drop_std, axis=0))

        if save:
            self.save_demos(actions, observations, states, infos, latent_states, num_demos)

    def pick_and_go_to_goal(self, env, lastObs, verbose=0):
        goal = lastObs['desired_goal']
        objectPos = lastObs['observation'][3:6]
        object_rel_pos = lastObs['observation'][6:9]

        episodeAcs = []
        episodeObs = []
        episodeStates = []
        episodeInfo = []
        episodeLatentStates = []

        object_oriented_goal = object_rel_pos.copy()
        object_oriented_goal[2] += 0.03  # (+ in z-axis) first make the gripper go slightly above the object

        timeStep = 0  # count the total number of time-steps
        episodeObs.append(lastObs)
        episodeStates.append(np.concatenate([lastObs['observation'], lastObs['desired_goal']]))

        # Stage 1: Reach the object with an open gripper (reach within |delta(x), delta(y), fixed_delta(z)|)
        while np.linalg.norm(object_oriented_goal) >= 0.005 and timeStep <= env._max_episode_steps:
            if verbose:
                env.render()
            action = [0, 0, 0, 0]
            object_oriented_goal = object_rel_pos.copy()
            object_oriented_goal[2] += 0.03  # for every new observation, assume that object is higher along z-axis

            for i in range(len(object_oriented_goal)):
                action[i] = object_oriented_goal[i] * 6

            action[len(action) - 1] = 0.05  # for the last action which is gripper opening/closing

            obsDataNew, reward, done, info = env.train(action)
            timeStep += 1

            episodeLatentStates.append(0)
            episodeAcs.append(action)
            episodeInfo.append(info)
            episodeObs.append(obsDataNew)
            episodeStates.append(np.concatenate([obsDataNew['observation'], obsDataNew['desired_goal']]))

            objectPos = obsDataNew['observation'][3:6]
            object_rel_pos = obsDataNew['observation'][6:9]

        # Stage 2: Close the gripper around the object (delta(x), delta(y), delta(z)) all decreases
        # (the gripper pos can be slightly different from object's at the end of stage 2)
        while np.linalg.norm(object_rel_pos) >= 0.005 and timeStep <= env._max_episode_steps:
            if verbose:
                env.render()
            action = [0, 0, 0, 0]
            for i in range(len(object_rel_pos)):
                action[i] = object_rel_pos[i] * 6

            action[len(action) - 1] = -0.005  # Close

            obsDataNew, reward, done, info = env.train(action)
            timeStep += 1

            episodeLatentStates.append(0)
            episodeAcs.append(action)
            episodeInfo.append(info)
            episodeObs.append(obsDataNew)
            episodeStates.append(np.concatenate([obsDataNew['observation'], obsDataNew['desired_goal']]))

            objectPos = obsDataNew['observation'][3:6]
            object_rel_pos = obsDataNew['observation'][6:9]

        # Stage 3: Bring the object to the goal
        # Note: your achieved goal is the object's pos which should be within epsilon distance from the desired goal
        while np.linalg.norm(goal - objectPos) >= 0.01 and timeStep <= env._max_episode_steps:
            if verbose:
                env.render()
            action = [0, 0, 0, 0]
            for i in range(len(goal - objectPos)):
                action[i] = (goal - objectPos)[i] * 6

            action[len(action) - 1] = -0.005  # Keep the gripper closed

            obsDataNew, reward, done, info = env.train(action)
            timeStep += 1

            episodeLatentStates.append(1)
            episodeAcs.append(action)
            episodeInfo.append(info)
            episodeObs.append(obsDataNew)
            episodeStates.append(np.concatenate([obsDataNew['observation'], obsDataNew['desired_goal']]))

            objectPos = obsDataNew['observation'][3:6]
            object_rel_pos = obsDataNew['observation'][6:9]

        # Stage 4: Populate the remaining time-steps with no action taken states
        while True:  # limit the number of time-steps in the episode to a fixed duration
            if verbose:
                env.render()
            action = [0, 0, 0, 0]

            action[len(action) - 1] = -0.005  # keep the gripper closed

            obsDataNew, reward, done, info = env.train(action)
            timeStep += 1

            episodeLatentStates.append(1)
            episodeAcs.append(action)
            episodeInfo.append(info)
            episodeObs.append(obsDataNew)
            episodeStates.append(np.concatenate([obsDataNew['observation'], obsDataNew['desired_goal']]))

            # objectPos = obsDataNew['observation'][3:6]
            # object_rel_pos = obsDataNew['observation'][6:9]

            if timeStep >= env._max_episode_steps: break

        return episodeAcs, episodeObs, episodeStates, episodeInfo, episodeLatentStates


def main():
    """
    Fetch Tasks:
    - Goal 3D
    - Action 4D (x, y, z, I) where x,y,z specify gripper moment in cartesian co-ords and I=0/1 specify gripper is open/close
      (The taken gripper action is applied for 20 subsequent simulator steps before control is returned back to agent
    - Observation Elements := (grip_pos, object_pos, object_rel_pos, gripper_state(size=1), object_rot, object_velp, object_velr, grip_velp, gripper_vel):
        - (x,y,z,v) of gripper where v is the velocity
        - (x,y,z,v) of robot's gripper
        - If object is present:
            - s_o = (x,y,z) of object,
            - theta: rotation of object using Euler angle,
            - v and omega, its linear and angular velocities
            - rel(s_o) and rel(v) relative position and velocity of object w.r.t gripper
    """
    data_dir = '/Users/apple/Desktop/PhDCoursework/COMP641/InfoGAIL/training_data/OpenAIPickandPlace'
    env = FetchPickAndPlace(data_dir)
    env.generate_demos(num_demos=1000, save=False, verbose=0)


def rough():
    # This env has smaller state dimension from the rest
    # print("\n------- FetchReach -------")
    # env = gym.make('FetchReach-v1')
    # obs = env.reset()
    # print(obs['observation'].shape)
    # env.render()

    # All three following envs have the same state dim. and involves working with an object(or puck)
    # print("\n------- FetchPush -------")
    # env = gym.make('FetchPush-v1')
    # obs = env.reset()
    # print(obs['observation'].shape)
    # env.render()

    print("\n------- FetchPickAndPlace -------")
    env = gym.make('FetchPickAndPlace-v1')
    obs = env.reset()
    print(obs['observation'].shape)
    env.render()

    # This env is more difficult than the rest
    # print("\n------- FetchSlide -------")
    # env = gym.make('FetchSlide-v1')
    # obs = env.reset()
    # print(obs['observation'].shape)
    # env.render()


# import gym
#
# env = gym.make('FetchReach-v1')
# obs = env.reset()
# print(obs)
# print(env.action_space)
# print(env.observation_space)
# env.render()
if __name__ == "__main__":
    main()
