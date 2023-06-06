import os
import pickle as pkl
import sys
import time
import numpy as np
# from config import training_data_config


class Dict2(dict):
    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)
        self.__dict__ = self


def one_hot_encode(vector, dim=None):
    if dim is None:
        one_hot_encoded = np.zeros((vector.size, vector.max() + 1))
        one_hot_encoded[np.arange(vector.size), vector] = 1
    else:
        one_hot_encoded = np.zeros((vector.size, dim))
        one_hot_encoded[np.arange(vector.size), vector] = 1
    return one_hot_encoded


def yield_batched_indexes(start, b_size, n_samples):
    while True:
        if start + b_size < n_samples:
            yield list(range(start, start+b_size))
            start = start + b_size
        else:
            yield list(range(start, n_samples)) + list(range(0, (start + b_size) % n_samples))
            start = (start + b_size) % n_samples


def causally_parse_dynamic_data_v2(traj_data, lower_bound=0, upper_bound=100, window_size=4):
    """
    Returns: s_t, [s_{t-3}; s_{t-2}; s_{t-1}; s_t], a_{t-1}, a_t, c_t, c_{t-1}, [t-3;t-2;t-1;t]
    """
    demo_data = {}

    if upper_bound - lower_bound:
        # Read the data in desirable format
        traj_sa = []
        for (traj_s, traj_a) in zip(traj_data['s'][lower_bound:upper_bound], traj_data['a'][lower_bound:upper_bound]):
            traj = []
            for (s, a) in zip(traj_s, traj_a):
                traj.append((s, a))
            traj_sa.append(traj)

        traj_c = traj_data['c'][lower_bound:upper_bound]

        # a_dim = 4
        temporal_positions = []
        data_curr_s = []
        data_curr_stack_s = []
        data_prev_a = []
        data_next_a = []
        data_curr_c = []
        data_prev_c = []
        # trajectory_start_points, trajectory_end_points = [], []
        # temp = 0

        # For each trajectory
        for traj_id in range(upper_bound-lower_bound):
            demo = traj_sa[traj_id]
            curr_stack_s_t = []
            temp_pos_t = []
            if len(demo) > 1:
                # Pad the trajectory of states with the initial state
                padded_s = [demo[0][0] for _ in range(window_size - 1)] + \
                               [demo[_id][0] for _id in range(len(demo))]
                temp_t = [np.array([0, ]) for _ in range(window_size - 1)] + \
                         [np.array([_id, ]) for _id in range(len(demo))]

                curr_s = [demo[_id][0] for _id in range(1, len(demo))]
                curr_a = [demo[_id][1] for _id in range(1, len(demo))]
                prev_a = [demo[_id][1] for _id in range(0, len(demo)-1)]
                curr_c = [traj_c[traj_id][_id] for _id in range(1, len(demo))]
                prev_c = [traj_c[traj_id][_id] for _id in range(0, len(demo) - 1)]

                # Get s_t to s_{t+window_size} states (Ignore the first stack i.e. (for window size=4)
                # [s0;s0;s0;s0] since we are ignoring s0 as ot does not have previous action)
                _id = 1
                while _id + window_size <= len(padded_s):
                    s_temporal = padded_s[_id: _id + window_size]
                    s_temporal = np.concatenate(s_temporal)
                    curr_stack_s_t.append(s_temporal)

                    pos_temporal = np.concatenate(temp_t[_id: _id + window_size])
                    temp_pos_t.append(pos_temporal)
                    _id += 1

                data_curr_s += curr_s
                data_curr_stack_s += curr_stack_s_t
                data_prev_a += prev_a
                data_next_a += curr_a
                data_curr_c += curr_c
                data_prev_c += prev_c
                temporal_positions += temp_pos_t

        demo_data = {
            'curr_states': np.array(data_curr_s),
            # For multi-dim states, the stacked version is (w_size x dim1, dim2, ..)
            'curr_stack_states': np.array(data_curr_stack_s),
            'prev_actions': np.array(data_prev_a),
            'next_actions': np.array(data_next_a),
            'curr_latent_states': np.array(data_curr_c),
            'prev_latent_states': np.array(data_prev_c),
            # Encodes temporal position of each state in the corresponding stack (n_samples x w_size)
            'temporal_positions': np.array(temporal_positions)
        }
    return demo_data

#
# def parse_data(env, trajectories, latent_modes_traj, lower_bound=0, upper_bound=training_data_config.num_traj):
#     total_traj = len(trajectories)
#     expert_demos = trajectories[lower_bound:upper_bound]
#     latent_modes_traj = latent_modes_traj[lower_bound:upper_bound]
#
#     states = []
#     actions = []
#     latent_modes = []
#     for traj_id in range(upper_bound-lower_bound):
#         demo = expert_demos[traj_id]
#         for sa_pair_id in range(len(demo)):
#             sa_pair = demo[sa_pair_id]
#             states.append(env.inv_states_dict[sa_pair[0]])
#             actions.append(sa_pair[1])
#             latent_modes.append(
#                 env.latent_state_dict[tuple(latent_modes_traj[traj_id][sa_pair_id])]
#             )
#     demo_data = {
#         'states': np.array(states),
#         'actions': np.array(one_hot_encode(np.array(actions), dim=len(env.action_dict))),
#         'latent_states': np.array(one_hot_encode(np.array(latent_modes), dim=len(env.latent_dict)))
#     }
#     return demo_data


def debug_time_taken(start_time, msg):
    print("Time Taken {}: ".format(msg), round(time.time()-start_time, 3))


def calc_bc_loss(act_gt, act_pred, eps=10e-9):
    xe_loss = np.average(-np.sum(np.multiply(act_gt, np.log(act_pred+eps)), axis=1))
    return np.array([xe_loss])


def check_and_load(path):
    if os.path.exists(path):
        with open(path, 'rb') as o:
            obj = pkl.load(o)
        return obj
    return None


def save(obj, path):
    with open(path, 'wb') as o:
        pkl.dump(obj, o, protocol=2)