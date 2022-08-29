import copy
import os
import sys
import pickle as pkl
import numpy as np
import random
from itertools import permutations
from typing import List, Tuple
from utils.misc import one_hot_encode

step_to_move = {
    'u': [-1, 0],  # Up
    'r': [0, 1],  # Right
    'l': [0, -1],  # Left
    'd': [1, 0],  # Down
}

step_to_id = {
    'u': 0,  # Up
    'r': 1,  # Right
    'l': 2,  # Left
    'd': 3,  # Down
}

grid_regions = {
    0: [[0, 0], [0, 1], [1, 0], [1, 1]],
    1: [[0, 3], [0, 4], [1, 3], [1, 4]],
    2: [[3, 0], [3, 1], [4, 0], [4, 1]],
    3: [[3, 3], [3, 4], [4, 3], [4, 4]]
}

num_objects = 3
num_actions = 4
grid_length_x = 5
grid_length_y = 5
obstacles = [[0, 2], [1, 2], [3, 2], [4, 2]]


def sample_xy(valid_posns: List):
    picked_pos = random.choice(valid_posns)
    return [picked_pos[0], picked_pos[1]]


def _find(state, _id):
    x, y = np.where(state[:, :, _id])
    assert len(x) == len(y) == 1
    return [x[0], y[0]]


def return_valid_actions(agent, obs):
    step_actions = {}
    for a in step_to_move.keys():
        new_agent_pos = list(np.array(agent) + np.array(step_to_move[a]))
        if new_agent_pos[0] >= 5 or new_agent_pos[0] < 0 or new_agent_pos[1] >= 5 or new_agent_pos[1] < 0.:
            continue
        elif new_agent_pos in obs:
            continue
        else:
            step_actions[a] = new_agent_pos
    return step_actions


def apply_action(s, a):
    """
    Returns: s_{t+1} from applying a_t to s_t
    """
    # neighs = {}
    stepid_to_move = {
        0: [-1, 0],  # Up
        1: [0, 1],  # Right
        2: [0, -1],  # Left
        3: [1, 0],  # Down
    }

    # Apply the action on current state if provided else get from env: Useful for HER

    # Locate agent's position
    curr_agent_pos = _find(s, 0)

    new_agent_pos = list(np.array(curr_agent_pos) + np.array(stepid_to_move[a]))

    # 1] Out-of-space actions - stay
    if new_agent_pos[0] >= grid_length_x or new_agent_pos[0] < 0 or new_agent_pos[1] >= grid_length_y or \
            new_agent_pos[1] < 0.:
        # neighs[a] = s
        return s
    else:
        # Determine the statuses of agent and objects in next state
        z = s[curr_agent_pos[0], curr_agent_pos[1], :].copy()
        z_prime = s[new_agent_pos[0], new_agent_pos[1], :].copy()

        # 2] Obstacle-hitting actions - stay
        if z_prime[5] == 1:
            # neighs[a] = s
            return s
        else:

            # 3] Object aggregating actions (not at goal) - stay
            if np.count_nonzero(z_prime[1:4] + z[1:4]) > 1 and z_prime[4] != 1 and z[4] != 1:
                # neighs[a] = s
                return s

            # 4] Update status of the positions moved from and moved to by the agent
            else:
                # Move the agent and the objects from z to z_prime
                if z[4] != 1:
                    z_prime[:4] = z_prime[:4] + z[:4]
                    z[:4] = z[:4] - z[:4]
                # Move the agent from z to z_prime but leave the objects at z
                else:
                    z_prime[0] = 1
                    z[0] = 0

                s[curr_agent_pos[0], curr_agent_pos[1], :] = z
                s[new_agent_pos[0], new_agent_pos[1], :] = z_prime
                # neighs[a] = s_prime
                return s


def reach_ooi(agent, ooi, obs) -> Tuple[List, List]:
    """
    The following logic compares any actions based on their impact on distance from the goal.
    Shortest Path not guaranteed!
    """

    actions = []

    delta_x = agent[0] - ooi[0]
    delta_y = agent[1] - ooi[1]
    # print("Delta X: ", delta_x, " Delta Y: ", delta_y)
    while abs(delta_x) + abs(delta_y) != 0:

        valid_actions = return_valid_actions(agent, obs)

        # Take actions to reduce delta_x and delta_y

        # Case 1
        if delta_x > 0:  # Go up
            if 'u' in valid_actions.keys():
                actions.append('u')
                agent = list(np.array(agent) + np.array(step_to_move['u']))
                # print("#2: ", actions, agent)

            else:  # Up blocked by obstacle -> Go right or left
                if 'r' in valid_actions.keys() and 'l' in valid_actions.keys():
                    r_new_delta_y = list(np.array(agent) + np.array(step_to_move['r']))[1] - ooi[1]
                    l_new_delta_y = list(np.array(agent) + np.array(step_to_move['l']))[1] - ooi[1]

                    # When both are equal - take left
                    if abs(r_new_delta_y) < abs(l_new_delta_y):
                        actions.append('r')
                        agent = list(np.array(agent) + np.array(step_to_move['r']))
                    else:
                        actions.append('l')
                        agent = list(np.array(agent) + np.array(step_to_move['l']))
                elif 'r' in valid_actions.keys():
                    actions.append('r')
                    agent = list(np.array(agent) + np.array(step_to_move['r']))
                elif 'l' in valid_actions.keys():
                    actions.append('l')
                    agent = list(np.array(agent) + np.array(step_to_move['l']))
                else:
                    print("Error")
                    break

        # Case 2
        elif delta_x < 0: # Go Down
            if 'd' in valid_actions.keys():
                actions.append('d')
                agent = list(np.array(agent) + np.array(step_to_move['d']))
            else:  # Down blocked by obstacle -> Go right or left
                if 'r' in valid_actions.keys() and 'l' in valid_actions.keys():
                    r_new_delta_y = list(np.array(agent) + np.array(step_to_move['r']))[1] - ooi[1]
                    l_new_delta_y = list(np.array(agent) + np.array(step_to_move['l']))[1] - ooi[1]

                    # When both are equal - take left
                    if abs(r_new_delta_y) < abs(l_new_delta_y):
                        actions.append('r')
                        agent = list(np.array(agent) + np.array(step_to_move['r']))
                    else:
                        actions.append('l')
                        agent = list(np.array(agent) + np.array(step_to_move['l']))
                elif 'r' in valid_actions.keys():
                    actions.append('r')
                    agent = list(np.array(agent) + np.array(step_to_move['r']))
                elif 'l' in valid_actions.keys():
                    actions.append('l')
                    agent = list(np.array(agent) + np.array(step_to_move['l']))
                else:
                    print("Error")
                    break

        # Case 3
        elif delta_y > 0:  # Go Left
            if 'l' in valid_actions.keys():
                actions.append('l')
                agent = list(np.array(agent) + np.array(step_to_move['l']))
                # print("#3: ", actions, agent)
            else:  # Left blocked by obstacle -> Go up or down
                if 'u' in valid_actions.keys() and 'd' in valid_actions.keys():
                    u_new_delta_y = list(np.array(agent) + np.array(step_to_move['u']))[1] - ooi[1]
                    d_new_delta_y = list(np.array(agent) + np.array(step_to_move['d']))[1] - ooi[1]

                    # When both are equal - go down
                    if abs(u_new_delta_y) < abs(d_new_delta_y):
                        actions.append('u')
                        agent = list(np.array(agent) + np.array(step_to_move['u']))
                    else:
                        actions.append('d')
                        agent = list(np.array(agent) + np.array(step_to_move['d']))
                elif 'u' in valid_actions.keys():
                    actions.append('u')
                    agent = list(np.array(agent) + np.array(step_to_move['u']))
                elif 'd' in valid_actions.keys():
                    actions.append('d')
                    agent = list(np.array(agent) + np.array(step_to_move['d']))
                else:
                    print("Error")
                    break

        # Case 4
        else:
            if 'r' in valid_actions.keys():
                actions.append('r')
                agent = list(np.array(agent) + np.array(step_to_move['r']))
            else:  # Right blocked by obstacle -> Go up or down
                if 'u' in valid_actions.keys() and 'd' in valid_actions.keys():
                    u_new_delta_y = list(np.array(agent) + np.array(step_to_move['u']))[1] - ooi[1]
                    d_new_delta_y = list(np.array(agent) + np.array(step_to_move['d']))[1] - ooi[1]

                    # When both are equal - go down
                    if abs(u_new_delta_y) < abs(d_new_delta_y):
                        actions.append('u')
                        agent = list(np.array(agent) + np.array(step_to_move['u']))
                    else:
                        actions.append('d')
                        agent = list(np.array(agent) + np.array(step_to_move['d']))
                elif 'u' in valid_actions.keys():
                    actions.append('u')
                    agent = list(np.array(agent) + np.array(step_to_move['u']))
                elif 'd' in valid_actions.keys():
                    actions.append('d')
                    agent = list(np.array(agent) + np.array(step_to_move['d']))
                else:
                    print("Error")
                    break

        # Update
        delta_x = agent[0] - ooi[0]
        delta_y = agent[1] - ooi[1]
        # print("Delta X: ", delta_x, " Delta Y: ", delta_y)

    return agent, actions


def reach_roi(agent, ooi, obs):
    ooi_reg = 'l' if ooi[1] < 2 else 'r'
    action_seq = []

    # If agent is at the bridge
    if agent == [2, 2]:
        if ooi_reg == 'r':
            action_seq.append('r')
            agent = list(np.array(agent) + np.array(step_to_move['r']))
        else:
            action_seq.append('l')
            agent = list(np.array(agent) + np.array(step_to_move['l']))
            # print("#1: ", agent)

    elif agent[1] in [0, 1] and ooi_reg == 'r':
        agent, seg_a = reach_ooi(agent, [2, 1], obs)
        action_seq.extend(seg_a)
        try:
            assert agent == [2, 1]
        except AssertionError:
            print("Problem with Expert Traj")
            sys.exit(-1)

        # Then Cross the bridge
        seg_a = ['r', 'r']
        action_seq.extend(seg_a)
        for a in seg_a:
            agent = list(np.array(agent) + np.array(step_to_move[a]))

    elif agent[1] in [3, 4] and ooi_reg == 'l':
        agent, seg_a = reach_ooi(agent, [2, 3], obs)
        action_seq.extend(seg_a)
        try:
            assert agent == [2, 3]
        except AssertionError:
            print("Problem with Expert Traj")
            sys.exit(-1)

        # Then Cross the bridge
        seg_a = ['l', 'l']
        action_seq.extend(seg_a)
        for a in seg_a:
            agent = list(np.array(agent) + np.array(step_to_move[a]))

    agent, seg_a = reach_ooi(agent, ooi, obs)
    action_seq.extend(seg_a)
    return agent, action_seq


def generate_demos(num_demos=100):

    grid_posns = [[x, y] for x in range(grid_length_x) for y in range(grid_length_y)]
    valid_grid_posns = [list(pos) for pos in grid_posns if pos not in obstacles]

    data_states = []
    data_latent_modes = []
    data_actions = []
    data_top_latent_modes = []
    episode_lengths = []

    for _ in range(num_demos):

        state = np.zeros((5, 5, 6))
        for obstacle in obstacles:
            state[obstacle[0], obstacle[1], 5] = 1

        goal_region, obj1_region, obj2_region, obj3_region = random.choice(list(permutations(range(1 + num_objects), 1 + num_objects)))

        goal_pos = sample_xy(grid_regions[goal_region])
        state[goal_pos[0], goal_pos[1], 4] = 1
        obj1_pos = sample_xy(grid_regions[obj1_region])
        state[obj1_pos[0], obj1_pos[1], 1] = 1
        obj2_pos = sample_xy(grid_regions[obj2_region])
        state[obj2_pos[0], obj2_pos[1], 2] = 1
        obj3_pos = sample_xy(grid_regions[obj3_region])
        state[obj3_pos[0], obj3_pos[1], 3] = 1
        objs_pos = [obj1_pos, obj2_pos, obj3_pos]

        _agent = sample_xy([pos for pos in valid_grid_posns if pos not in [goal_pos, obj1_pos, obj2_pos, obj3_pos]])
        print("Agent: ", _agent, " Obj1: ", obj1_pos, " Obj2: ", obj2_pos, " Obj3: ", obj3_pos, " Goal: ",
              goal_pos)

        state[_agent[0], _agent[1], 0] = 1

        object_ids = random.choice(list(permutations(range(num_objects), num_objects)))

        # For fixed order: Obj2 -> Obj1 -> Obj3
        print("Generating Fixed Latent Mode Trajectories")
        object_ids = (1, 0, 2)

        traj_actions = []
        traj_states = [state]
        traj_latent_modes = []
        traj_top_latent_modes = []
        for c in object_ids:
            obj_ooi = objs_pos[c]
            print("Object Being picked: ", obj_ooi)
            # Pick-Up
            temp_obs = obstacles + [pos for pos in objs_pos if pos != obj_ooi]
            _agent, seg_actions = reach_roi(agent=_agent, ooi=obj_ooi, obs=temp_obs)
            seg_latent_mode = [2*c for _ in range(len(seg_actions))]
            seg_top_latent_mode = [c for _ in range(len(seg_actions))]
            traj_actions.extend(seg_actions)
            traj_latent_modes.extend(seg_latent_mode)
            traj_top_latent_modes.extend(seg_top_latent_mode)

            # Drop
            _agent, seg_actions = reach_roi(agent=_agent, ooi=goal_pos, obs=temp_obs)
            seg_latent_mode = [2*c+1 for _ in range(len(seg_actions))]
            seg_top_latent_mode = [c for _ in range(len(seg_actions))]
            traj_actions.extend(seg_actions)
            traj_latent_modes.extend(seg_latent_mode)
            traj_top_latent_modes.extend(seg_top_latent_mode)

        # VIMP: do not keep the variable same throughout this loop as when it changes it also changes the existing
        # elements already in the list since they all refer to the same state
        curr_state = copy.deepcopy(state)
        for a in traj_actions:
            next_state = apply_action(curr_state, step_to_id[a])
            traj_states.append(next_state)
            curr_state = copy.deepcopy(next_state)

        try:
            assert _find(curr_state, 0) == _find(curr_state, 1) == _find(curr_state, 2) == _find(curr_state, 3) == _find(curr_state, 4)
        except AssertionError:
            print("Terminal State not Reached!")
            sys.exit(-1)

        data_states.append(np.array(traj_states[:-1]))  # No Terminal State
        data_actions.append(np.array([one_hot_encode(np.array(step_to_id[a], dtype=np.int), dim=num_actions)[0]
                                      for a in traj_actions]))
        data_latent_modes.append(np.array([one_hot_encode(np.array(c, dtype=np.int), dim=num_objects*2)[0]
                                           for c in traj_latent_modes]))
        data_top_latent_modes.append(np.array([one_hot_encode(np.array(c, dtype=np.int), dim=num_objects)[0]
                                               for c in traj_top_latent_modes]))

        episode_lengths.append(len(traj_states[:-1]))

    # Save Data as a dict
    data = {
        's': np.array(data_states),
        'a': np.array(data_actions),
        'c': np.array(data_latent_modes),  # Separate indicator for pick/drop of each object
        'top_c': np.array(data_top_latent_modes)  # Same indicator for pick/drop of every object
    }
    # Save the data in python-2 compatible format
    data_dir = '/Users/apple/Desktop/PhDCoursework/COMP641/InfoGAIL/training_data/StackBoxWorld'
    with open(os.path.join(data_dir, "stateaction_fixed_latent_dynamic_test.pkl"), 'wb') as o:
        pkl.dump(data, o, protocol=2)

    # Dataset Information
    print("------------------------ Dataset Information ------------------------")
    print("Average Episode Length: ", sum(episode_lengths)/num_demos)
    print("Max Episode Length: ", max(episode_lengths))
    print("Min Episode Length: ", min(episode_lengths))


generate_demos()






