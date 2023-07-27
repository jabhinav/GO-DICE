import os
import random
from collections import Counter
from typing import Optional, List

import numpy as np
import tensorflow as tf

from her.rollout import RolloutWorker
from utils.custom import logger
from utils.env import save_env_img

# from domains.stackBoxWorld import _find, stackBoxWorld
# from utils.misc import causally_parse_dynamic_data_v2, one_hot_encode


def get_index_of_all_max(preds):
	sample_max_indexes = []
	for _i in range(preds.shape[0]):
		sample_max_indexes.append(set(np.where(preds[_i] == np.max(preds[_i]))[0]))
	return sample_max_indexes


def pred_binary_error(_gt, _pred, weights=None):
	#  Choose maximum prob output which matches GT if multiple actions are predicted with high and equal prob
	_gt_max_indexes = get_index_of_all_max(_gt)
	_pred_max_indexes = get_index_of_all_max(_pred)
	num_t = len(_gt)
	count = 0
	if weights is not None:
		for i in range(num_t):
			if _gt_max_indexes[i] & _pred_max_indexes[i]:
				count += 1. * weights[i]
		return 1. - count
	else:
		for i in range(num_t):
			if _gt_max_indexes[i] & _pred_max_indexes[i]:
				count += 1.
		return 1. - (count / num_t)


# def train_test_mismatch():
#     env_name = 'StackBoxWorld'
#     root_dir = '/Users/apple/Desktop/PhDCoursework/COMP641/InfoGAIL/'
#     data_dir = os.path.join(root_dir, 'training_data/{}'.format(env_name))
#     train_data_path = os.path.join(data_dir, 'stateaction_fixed_latent_dynamic.pkl')
#     test_data_path = os.path.join(data_dir, 'stateaction_fixed_latent_dynamic_test.pkl')
#
#     def get_initial_configs(data_path):
#         unique_init_configs = []
#         with open(data_path, 'rb') as f:
#             traj_sac = pkl.load(f)
#         for traj_s in traj_sac['s']:
#             s_init = traj_s[0]
#             agent, obj1, obj2, obj3, goal = _find(s_init, 0), _find(s_init, 1), _find(s_init, 2), _find(s_init, 3), _find(s_init, 4)
#             init_config = agent + obj1 + obj2 + obj3 + goal
#             if init_config not in unique_init_configs:
#                 unique_init_configs.append(init_config)
#         return unique_init_configs
#
#     train_unique_init_configs = get_initial_configs(train_data_path)
#     test_unique_init_configs = get_initial_configs(test_data_path)
#
#     test_not_in_train = [config for config in test_unique_init_configs if config not in train_unique_init_configs]
#
#     print("-----------------------------------------------------------------------------------------------------------")
#     print("Number of Unique Train Configs =", len(train_unique_init_configs))
#     print("Train Configs:", train_unique_init_configs)
#     print("Number of Unique Test Configs =", len(test_unique_init_configs))
#     print("Test Configs:", test_unique_init_configs)
#     print("Number of unseen Test Configs: ", len(test_not_in_train))


# def evaluate_model_continuous(agent, model_class, test_traj_sac, train_config, file_txt_results_path):
#
#     # ################################################ Declare models ############################################## #
#     if model_class == "vae":
#         encoder = agent.encoder
#         classifier = agent.classifier
#         cond_prior = agent.cond_prior
#         actor = agent.decoder
#     elif model_class == "GAIL":
#         encoder = agent.posterior.encoder
#         classifier = agent.posterior.classifier
#         cond_prior = agent.cond_prior
#         actor = agent.actor
#     else:
#         raise NotImplementedError
#
#     # ################################################ Parse Data ############################################## #
#     demos = causally_parse_dynamic_data_v2(test_traj_sac, lower_bound=0, upper_bound=train_config["num_traj"],
#                                            window_size=train_config["w_size"])
#
#     num_samples = demos['curr_states'].shape[0]
#
#     # ############################################# Compute Static Metrics ########################################### #
#     if not train_config['perc_supervision']:
#         permutations = []
#         permutations_x = [list(perm) for perm in itertools.permutations(list(range(agent.g_dim)))]
#
#         for perm in permutations_x:
#             permutations.append(perm)
#     else:
#         permutations = [list(range(agent.g_dim))]
#
#     print("-------------------------------- RESULTS --------------------------------",
#           file=open(file_txt_results_path, 'a'))
#     curr_encodes_d = demos['curr_latent_states']
#     prev_encodes_d = demos['prev_latent_states']
#     perm_pred_l2_dis = []
#     perm_pred_latent_error = []
#     with tqdm(total=len(permutations), position=0, leave=True) as pbar:
#         for i, perm in enumerate(permutations):
#             # print "{}/{}".format(_id, len(permutations))
#             curr_encodes_d_perm = curr_encodes_d[:, perm]
#             prev_encodes_d_perm = prev_encodes_d[:, perm]
#
#             # x -> z -> y
#             num_z = 10  # More samples lead to better approx. of q(y|x)
#             [z_mu, z_std] = encoder(demos['curr_stack_states'], prev_encodes_d_perm)
#             multiple_z = multi_sample_normal_np(z_mu.numpy(), z_std.numpy(), agent.z_dim, k=num_z)
#             qy_zk = [classifier(z).numpy() for z in multiple_z]
#             qy_x = sum(qy_zk)/float(num_z)
#
#             # z = sample_normal_np(z_mu, z_std, self.z_dim)
#             # qy_z = self.classifier.predict([z])
#             decoding_error = pred_binary_error(curr_encodes_d_perm, qy_x)
#             perm_pred_latent_error.append(decoding_error)
#
#             # z -> x
#             next_act_pred_mu = actor(demos['curr_states'], z_mu.numpy())
#             avg_l2_loss = np.linalg.norm(demos['next_actions'] - next_act_pred_mu.numpy())/num_samples
#             perm_pred_l2_dis.append(avg_l2_loss)
#
#             pbar.refresh()
#             pbar.set_description("Perm ID {}".format(i + 1))
#             pbar.set_postfix(Decoding_Error=decoding_error)
#             pbar.update(1)
#
#     best_perm_index = int(np.argmin(np.array(perm_pred_latent_error)))
#     best_perm = permutations[best_perm_index]
#     print("Best Permutation: ", best_perm, file=open(file_txt_results_path, 'a'))
#     print("Best Avg. L2 Loss: ", perm_pred_l2_dis[best_perm_index], file=open(file_txt_results_path, 'a'))
#     print("Best Avg. Latent-Decoding Error: ", perm_pred_latent_error[best_perm_index],
#           file=open(file_txt_results_path, 'a'))
#
#     # ############################################# Compute Dynamic Metrics ########################################## #
#     print("-------------------------------- RESULTS Success Rate --------------------------------",
#           file=open(file_txt_results_path, 'a'))
#
#     # Get starting configurations
#     observations = []
#     for traj_s, traj_c in zip(test_traj_sac['s'], test_traj_sac['c']):
#         s_init = traj_s[0]
#         c_init = traj_c[0]
#         observations.append((s_init, c_init))
#
#     action_scale = np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
#     # action_scale = np.array([[0.14631747, 0.15090476, 0.11856036, 0.02602926]], dtype=np.float32)
#     success_rate = 0
#     env = gym.make('FetchPickAndPlace-v1')
#     env_max_steps = env._max_episode_steps
#     with tqdm(total=train_config['num_traj'], position=0, leave=True) as pbar:
#         for _id in range(train_config["num_traj"]):
#             # print("\n-------------- Trajectory ID{}: Initial Config --------------".format(_id),
#             #       file=open(file_txt_results_path, 'a'))
#
#             # Get Initial State and Latent mode ()
#             found_env = False
#             while not found_env:
#                 init_obs = env.reset()
#                 achieved_goal = init_obs['achieved_goal']
#                 desired_goal = init_obs['desired_goal']
#                 goal_distance = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
#                 if goal_distance > 0.05:
#                     found_env = True
#                 else:
#                     print("Skipping Env! Obj at Goal posn already!")
#
#             env.render()
#             step = 0
#             init_state = np.concatenate([init_obs['observation'], init_obs['desired_goal']])
#             init_latent_mode = one_hot_encode(np.array(0, dtype=np.int), dim=2)[0]
#
#             init_latent_mode = init_latent_mode[best_perm]
#             init_state = tf.expand_dims(init_state, axis=0)
#             init_encode_y = tf.expand_dims(init_latent_mode, axis=0)
#             _, init_scale_z, z_k = cond_prior(init_encode_y, k=10)
#             z_k = tf.squeeze(z_k, axis=1)  # (k, 1, z_dim) -> (k, z_dim)
#
#             p_x_k = actor(tf.repeat(init_state, repeats=10, axis=0), z_k)
#             init_action_mu = tf.reduce_mean(p_x_k, axis=0)
#             init_action_dis = Normal(loc=init_action_mu, scale=action_scale)
#             action = init_action_dis.sample()
#
#             prev_encode_y = init_encode_y
#             while step < env_max_steps + 1:
#                 next_obs, reward, done, info = env.train(action.numpy()[0])
#                 env.render()
#                 step += 1
#
#                 obj_achieved = info['is_success']
#                 if done:
#                     if obj_achieved:
#                         success_rate += 1
#                     break
#
#                 curr_state = np.concatenate([next_obs['observation'], next_obs['desired_goal']])
#                 curr_state = tf.expand_dims(curr_state, axis=0)
#
#                 # Predict next encode by sampling z from q(z|s_t, c_{t-1}) and averaging q(c_t|z^(i))
#                 [curr_z_mu, curr_z_std] = encoder(curr_state, prev_encode_y)
#                 sampled_zs = [_z for _z in multi_sample_normal_tf(curr_z_mu, curr_z_std, k=10)]
#                 qy_z_k = [(lambda x: classifier(x))(_z) for _z in sampled_zs]
#                 qy_z_k = tf.concat(values=[tf.expand_dims(qy_z, axis=0) for qy_z in qy_z_k], axis=0)
#                 qy_x = tf.reduce_mean(qy_z_k, axis=0)
#                 dist = Categorical(probs=qy_x, dtype=tf.float32)
#                 curr_encode_y = dist.sample().numpy()
#                 # If current encode has changed to 1 i.e. object has been picked, use a different action scale
#                 # if curr_encode_y:
#                 #     action_scale = np.array([[1.87254752e-01, 1.88611527e-01, 1.86052738e-01, 1.47017815e-18]],
#                 #                             dtype=np.float32)
#                 curr_encode_y = tf.one_hot(curr_encode_y, agent.g_dim)
#
#                 # Take action
#                 action_mu = actor(curr_state, curr_z_mu)
#                 dist = Normal(loc=action_mu, scale=action_scale)
#                 action = dist.sample()
#
#                 # The current encode becomes previous encode for the next iteration
#                 prev_encode_y = curr_encode_y
#
#             pbar.refresh()
#             pbar.set_description("Trajectory {}".format(_id + 1))
#             pbar.update(1)
#
#     print("\nSuccess rate: {}/{}".format(success_rate, train_config['num_traj']),
#           file=open(file_txt_results_path, 'a'))
#
#
# def evaluate_sup_model_continuous(agent, model_class, test_traj_sac, train_config, file_txt_results_path):
#
#     # ################################################ Declare models ############################################## #
#     if model_class == "vae":
#         actor = agent.decoder
#     elif model_class == "GAIL":
#         actor = agent.actor
#     else:
#         raise NotImplementedError
#
#     # ################################################ Parse Data ############################################## #
#     demos = causally_parse_dynamic_data_v2(test_traj_sac, lower_bound=0, upper_bound=train_config["num_traj"],
#                                            window_size=train_config["w_size"])
#     num_samples = demos['curr_states'].shape[0]
#
#     # ############################################# Compute Static Metrics ########################################### #
#
#     print("-------------------------------- RESULTS Metrics --------------------------------",
#           file=open(file_txt_results_path, 'a'))
#
#     next_act_pred = actor(demos['curr_states'], demos['curr_latent_states'])
#     avg_l2_loss = np.linalg.norm(demos['next_actions'] - next_act_pred) / num_samples
#
#     print("Best Avg. 0-1 Loss: ", avg_l2_loss, file=open(file_txt_results_path, 'a'))
#
#     # ############################################# Compute Dynamic Metrics ########################################## #
#     print("-------------------------------- RESULTS Success Rate --------------------------------",
#           file=open(file_txt_results_path, 'a'))
#
#     success_rate = 0
#     env = gym.make('FetchPickAndPlace-v1')
#     env_max_steps = env._max_episode_steps
#     with tqdm(total=train_config['num_traj'], position=0, leave=True) as pbar:
#         for _id in range(train_config['num_traj']):
#             # print("\n-------------- Trajectory ID{}: Initial Config --------------".format(_id),
#             #       file=open(file_txt_results_path, 'a'))
#
#             # Get Initial State and Latent mode
#             # Get Initial State and Latent mode
#             found_env = False
#             while not found_env:
#                 init_obs = env.reset()
#                 achieved_goal = init_obs['achieved_goal']
#                 desired_goal = init_obs['desired_goal']
#                 goal_distance = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
#                 if goal_distance > 0.05:
#                     found_env = True
#                 else:
#                     print("Skipping Env! Obj at Goal posn already!")
#
#             env.render()
#             step = 0
#             init_state = np.concatenate([init_obs['observation'], init_obs['desired_goal']])
#             init_latent_mode = one_hot_encode(np.array(0, dtype=np.int), dim=2)[0]
#
#             curr_state = tf.expand_dims(init_state, axis=0)
#             curr_encode_y = tf.expand_dims(init_latent_mode, axis=0)
#
#             # Until Object is picked
#             action_scale = np.array([[0.14631747, 0.15090476, 0.11856036, 0.02602926]], dtype=np.float32)
#             while step < env_max_steps:
#
#                 # Take action
#                 action_mu = actor(curr_state, curr_encode_y)
#                 dist = Normal(loc=action_mu, scale=action_scale)
#                 action = dist.sample()
#
#                 next_obs, reward, done, info = env.train(action.numpy()[0])
#                 env.render()
#                 step += 1
#
#                 obj_achieved = info['is_success']
#                 if done:
#                     if obj_achieved:
#                         success_rate += 1
#                     break
#
#                 curr_state = np.concatenate([next_obs['observation'], next_obs['desired_goal']])
#                 curr_state = tf.expand_dims(curr_state, axis=0)
#
#                 # The current encode should change if the object has been picked up
#                 # [We manually do that since env is not returning any gym response over latent mode]
#                 object_rel_pos = next_obs['observation'][6:9]
#                 if np.linalg.norm(object_rel_pos) < 0.005:
#                     curr_encode_y = one_hot_encode(np.array(1, dtype=np.int), dim=2)[0]
#                     curr_encode_y = tf.expand_dims(curr_encode_y, axis=0)
#                     # When object is picked
#                     action_scale = np.array([[1.87254752e-01, 1.88611527e-01, 1.86052738e-01, 1.47017815e-18]],
#                                             dtype=np.float32)
#
#             pbar.refresh()
#             pbar.set_description("Trajectory {}".format(_id + 1))
#             pbar.update(1)
#
#     print("\nSuccess rate: {}/{}".format(success_rate, train_config['num_traj']),
#           file=open(file_txt_results_path, 'a'))
#
#
# def evaluate_model_discrete(agent, model_class, test_traj_sac, train_config, file_txt_results_path):
#
#     # ################################################ Declare models ############################################## #
#     if model_class == "vae":
#         encoder = agent.encoder
#         classifier = agent.classifier
#         cond_prior = agent.cond_prior
#         actor = agent.decoder
#     elif model_class == "GAIL":
#         encoder = agent.posterior.encoder
#         classifier = agent.posterior.classifier
#         cond_prior = agent.cond_prior
#         actor = agent.actor
#     else:
#         raise NotImplementedError
#
#     # ################################################ Parse Data ############################################## #
#     demos = causally_parse_dynamic_data_v2(test_traj_sac, lower_bound=0, upper_bound=train_config["num_traj"],
#                                            window_size=train_config["w_size"])
#
#     # ############################################# Compute Static Metrics ########################################### #
#     if not train_config['perc_supervision']:
#         permutations = []
#         permutations_x = [list(perm) for perm in itertools.permutations(list(range(agent.g_dim)))]
#
#         for perm in permutations_x:
#             permutations.append(perm)
#     else:
#         permutations = [list(range(agent.g_dim))]
#
#     print("-------------------------------- RESULTS Metrics --------------------------------",
#           file=open(file_txt_results_path, 'a'))
#     curr_encodes_d = demos['curr_latent_states']
#     prev_encodes_d = demos['prev_latent_states']
#     perm_pred_0_1_runs = []
#     perm_pred_latent_error = []
#     with tqdm(total=len(permutations), position=0, leave=True) as pbar:
#         for _id, perm in enumerate(permutations):
#             # print "{}/{}".format(_id, len(permutations))
#             curr_encodes_d_perm = curr_encodes_d[:, perm]
#             prev_encodes_d_perm = prev_encodes_d[:, perm]
#
#             # x -> z -> y
#             num_z = 10  # More samples lead to better approx. of q(y|x)
#             [z_mu, z_std] = encoder(demos['curr_stack_states'], prev_encodes_d_perm)
#             multiple_z = multi_sample_normal_np(z_mu.numpy(), z_std.numpy(), agent.z_dim, k=num_z)
#             qy_zk = [classifier(z).numpy() for z in multiple_z]
#             qy_x = sum(qy_zk) / float(num_z)
#
#             # z = sample_normal_np(z_mu, z_std, self.z_dim)
#             # qy_z = self.classifier.predict([z])
#             decoding_error = pred_binary_error(curr_encodes_d_perm, qy_x)
#             perm_pred_latent_error.append(decoding_error)
#
#             # z -> x
#             next_act_pred = actor(demos['curr_states'], z_mu.numpy())
#             loss = pred_binary_error(demos['next_actions'], next_act_pred.numpy(), weights=None)
#             perm_pred_0_1_runs.append(loss)
#
#             pbar.refresh()
#             pbar.set_description("Perm ID {}".format(_id + 1))
#             pbar.set_postfix(Decoding_Error=decoding_error)
#             pbar.update(1)
#
#     best_perm_index = int(np.argmin(np.array(perm_pred_latent_error)))
#     best_perm = permutations[best_perm_index]
#     print("Best Permutation: ", best_perm, file=open(file_txt_results_path, 'a'))
#     print("Best Avg. 0-1 Loss: ", perm_pred_0_1_runs[best_perm_index], file=open(file_txt_results_path, 'a'))
#     print("Best Avg. Latent-Decoding Error: ", perm_pred_latent_error[best_perm_index],
#           file=open(file_txt_results_path, 'a'))
#
#     # ############################################# Compute Dynamic Metrics ########################################## #
#     print("-------------------------------- RESULTS Success Rate --------------------------------",
#           file=open(file_txt_results_path, 'a'))
#
#     # Get starting configurations
#     observations = []
#     for traj_s, traj_c in zip(test_traj_sac['s'], test_traj_sac['c']):
#         s_init = traj_s[0]
#         c_init = traj_c[0]
#         observations.append((s_init, c_init))
#
#     success_rate = 0
#     env = stackBoxWorld()
#     with tqdm(total=train_config['num_traj'], position=0, leave=True) as pbar:
#         for _id, (state, latent_mode) in enumerate(observations):
#             print("\n-------------- Trajectory ID{}: Initial Config --------------".format(_id),
#                   file=open(file_txt_results_path, 'a'))
#             print("Agent: {}, Obj1: {}, Obj2: {}, Obj3: {}, Goal: {}".format(_find(state, 0), _find(state, 1),
#                                                                              _find(state, 2), _find(state, 3),
#                                                                              _find(state, 4)),
#                   file=open(file_txt_results_path, 'a'))
#             print("Latent Mode:", latent_mode, file=open(file_txt_results_path, 'a'))
#
#             actions_taken = []
#
#             latent_mode = latent_mode[best_perm]
#             _, _, _, _ = env.force_set(state, np.argmax(latent_mode))
#             init_state = tf.expand_dims(state, axis=0)
#             init_encode_y = tf.expand_dims(latent_mode, axis=0)
#             _, _, z_k = cond_prior(init_encode_y, k=10)
#             z_k = tf.squeeze(z_k, axis=1)  # (k, 1, z_dim) -> (k, z_dim)
#
#             p_x_k = actor(tf.repeat(init_state, repeats=10, axis=0), z_k)
#             init_action_prob = tf.reduce_mean(p_x_k, axis=0)
#             init_action_dis = Categorical(probs=init_action_prob)
#             action = init_action_dis.sample()
#             actions_taken.append(env.steps_ref[action.numpy()])
#
#             prev_encode_y = init_encode_y
#             while not env.episode_ended:
#                 next_obs, raw, done, info = env.step(np.array(action))
#                 obj_achieved = info['is_success']
#
#                 curr_state = next_obs['grid_world'].astype(np.float32)
#                 curr_state = tf.expand_dims(curr_state, axis=0)
#                 if done:
#                     if obj_achieved:
#                         success_rate += 1
#                     _, _, _, _ = env.reset()
#                     break
#
#                 # Predict next encode by sampling z from q(z|s_t, c_{t-1}) and averaging q(c_t|z^(i))
#                 [curr_z_mu, curr_z_std] = encoder(curr_state, prev_encode_y)
#                 sampled_zs = [_z for _z in multi_sample_normal_tf(curr_z_mu, curr_z_std, k=10)]
#                 qy_z_k = [(lambda x: classifier(x))(_z) for _z in sampled_zs]
#                 qy_z_k = tf.concat(values=[tf.expand_dims(qy_z, axis=0) for qy_z in qy_z_k], axis=0)
#                 qy_x = tf.reduce_mean(qy_z_k, axis=0)
#                 dist = Categorical(probs=qy_x, dtype=tf.float32)
#                 curr_encode_y = dist.sample().numpy()
#                 curr_encode_y = tf.one_hot(curr_encode_y, agent.g_dim)
#
#                 # Take action
#                 action_prob = actor(curr_state, curr_z_mu)
#                 dist = Categorical(probs=action_prob)
#                 action = dist.sample()
#                 actions_taken.append(env.steps_ref[action.numpy()[0]])
#
#                 # The current encode becomes previous encode for the next iteration
#                 prev_encode_y = curr_encode_y
#
#             print("Actions Taken {}: ".format(len(actions_taken)), actions_taken,
#                   file=open(file_txt_results_path, 'a'))
#             pbar.refresh()
#             pbar.set_description("Trajectory {}".format(_id + 1))
#             pbar.update(1)
#
#     print("\nSuccess rate: {}/{}".format(success_rate, train_config['num_traj']),
#           file=open(file_txt_results_path, 'a'))
#
#
# def evaluate_sup_model_discrete(agent, model_class, test_traj_sac, train_config, file_txt_results_path):
#
#     # ################################################ Declare models ############################################## #
#     if model_class == "vae":
#         actor = agent.decoder
#     elif model_class == "GAIL":
#         actor = agent.actor
#     else:
#         raise NotImplementedError
#
#     # ################################################ Parse Data ############################################## #
#     demos = causally_parse_dynamic_data_v2(test_traj_sac, lower_bound=0, upper_bound=train_config["num_traj"],
#                                            window_size=train_config["w_size"])
#
#     # ############################################# Compute Static Metrics ########################################### #
#
#     print("-------------------------------- RESULTS Metrics --------------------------------",
#           file=open(file_txt_results_path, 'a'))
#
#     next_act_pred = actor(demos['curr_states'], demos['curr_latent_states'])
#     loss = pred_binary_error(demos['next_actions'], next_act_pred.numpy(), weights=None)
#
#     print("Best Avg. 0-1 Loss: ", loss, file=open(file_txt_results_path, 'a'))
#
#     # ############################################# Compute Dynamic Metrics ########################################## #
#     print("-------------------------------- RESULTS Success Rate --------------------------------",
#           file=open(file_txt_results_path, 'a'))
#
#     # Get starting configurations
#     observations = []
#     for traj_s, traj_c in zip(test_traj_sac['s'], test_traj_sac['c']):
#         s_init = traj_s[0]
#         c_init = traj_c[0]
#         observations.append((s_init, c_init))
#
#     success_rate = 0
#     env = stackBoxWorld()
#     with tqdm(total=train_config['num_traj'], position=0, leave=True) as pbar:
#         for _id, (state, latent_mode) in enumerate(observations):
#             print("\n-------------- Trajectory ID{}: Initial Config --------------".format(_id),
#                   file=open(file_txt_results_path, 'a'))
#             print("Agent: {}, Obj1: {}, Obj2: {}, Obj3: {}, Goal: {}".format(_find(state, 0), _find(state, 1),
#                                                                              _find(state, 2), _find(state, 3),
#                                                                              _find(state, 4)),
#                   file=open(file_txt_results_path, 'a'))
#             print("Latent Mode:", latent_mode, file=open(file_txt_results_path, 'a'))
#
#             actions_taken = []
#
#             _, _, _, _ = env.force_set(state, np.argmax(latent_mode))
#             curr_state = tf.expand_dims(state, axis=0)
#             curr_encode_y = tf.expand_dims(latent_mode, axis=0)
#             while not env.episode_ended:
#                 # Take action
#                 action_prob = actor(curr_state, curr_encode_y)
#                 dist = Categorical(probs=action_prob)
#                 action = dist.sample()
#                 actions_taken.append(env.steps_ref[action.numpy()[0]])
#
#                 # Step Env
#                 next_obs, raw, done, info = env.step(np.array(action))
#                 obj_achieved = info['is_success']
#
#                 curr_state = next_obs['grid_world'].astype(np.float32)
#                 curr_state = tf.expand_dims(curr_state, axis=0)
#                 curr_encode_y = next_obs['latent_mode'].astype(np.float32)
#                 curr_encode_y = tf.expand_dims(curr_encode_y, axis=0)
#                 if done:
#                     if obj_achieved:
#                         success_rate += 1
#                     _, _, _, _ = env.reset()
#                     break
#
#             print("Actions Taken {}: ".format(len(actions_taken)), actions_taken,
#                   file=open(file_txt_results_path, 'a'))
#             pbar.refresh()
#             pbar.set_description("Trajectory {}".format(_id + 1))
#             pbar.update(1)
#
#     print("\nSuccess rate: {}/{}".format(success_rate, train_config['num_traj']),
#           file=open(file_txt_results_path, 'a'))


def evaluate(actor, env, num_episodes=100):
	"""Evaluates the policy.
	
	Args:
	  actor: A policy to evaluate.
	  env: Environment to evaluate the policy on.
	  num_episodes: A number of episodes to average the policy on.
	
	Returns:
	  Averaged reward and a total number of steps.
	"""
	total_timesteps = 0
	total_returns = 0
	
	for _ in range(num_episodes):
		state = env.reset()
		done = False
		while not done:
			action, _, _ = actor(np.array([state]))
			action = action[0].numpy()
			
			next_state, reward, done, _ = env.step(action)
			
			total_returns += reward
			total_timesteps += 1
			state = next_state
	
	return total_returns / num_episodes, total_timesteps / num_episodes


def evaluate_worker(
		worker: RolloutWorker,
		num_episodes,
		log_traj: bool = False,
		log_dir: Optional[str] = None,
		resume_states: Optional[List[dict]] = None,
		subgoal_reward: int = 0,
):
	"""Evaluates the policy.

		Args:
		  worker: Rollout worker.
		  num_episodes: A number of episodes to average the policy on.
		  log_traj: Whether to log the skills or not (default: False).
		  log_dir: Directory to log (default: None).
		  resume_states: List of Resume states (default: None).
		  subgoal_reward: Subgoal reward (default: None).
		Returns:
		  Bunch of metrics.
	"""
	
	total_timesteps = []
	total_returns = []
	avg_final_goal_dist = []
	avg_perc_decrease = []
	avg_subgoal_achievement_reward = []
	subgoals_avg_info = {}
	subgoals_dist_info = {}
	
	i = 0
	exception_count = 0
	while i < num_episodes:
		
		if resume_states is None:
			# logger.info("No resume init state provided. Randomly initializing the env.")
			resume_state_dict = None
		else:
			logger.info(f"Resume init state provided! Check if you intended to do so.")
			# sample a random state from the list of resume states
			resume_state_dict = random.choice(resume_states)
		
		try:
			episode, stats = worker.generate_rollout(resume_state_dict=resume_state_dict)
		except Exception as e:
			exception_count += 1
			logger.info(f"Exception occurred: {e}")
			if exception_count < 10:
				continue
			else:
				raise e
		
		# Metric 1: Success Rate
		success = stats['ep_success'].numpy() if isinstance(stats['ep_success'], tf.Tensor) else stats['ep_success']
		total_returns.append(success)
		
		# Metric 2: Episode Length
		length = stats['ep_length'].numpy() if isinstance(stats['ep_length'], tf.Tensor) else stats['ep_length']
		total_timesteps.append(length)
		
		# Metric 3: Final Goal Distance
		final_goal_dist = episode['distances'][0][-1]
		final_goal_dist = final_goal_dist.numpy() if isinstance(final_goal_dist, tf.Tensor) else final_goal_dist
		avg_final_goal_dist.append(final_goal_dist)
		
		# Metric 4: Percentage Decrease in Goal Distance
		init_goal_dist = episode['distances'][0][0]
		init_goal_dist = init_goal_dist.numpy() if isinstance(init_goal_dist, tf.Tensor) else init_goal_dist
		perc_decrease = (init_goal_dist - final_goal_dist) / init_goal_dist
		avg_perc_decrease.append(perc_decrease)
		
		# Metric 5: SubGoal Achievement Indicators
		subgoal_info, subgoal_distances = worker.env.get_subgoal_info()  # Do after the episode is done and before the reset
		for subgoal, info in subgoal_info.items():
			if subgoal not in subgoals_avg_info:
				subgoals_avg_info[subgoal] = []
			subgoals_avg_info[subgoal].append(info)
		
		for subgoal, dist in subgoal_distances.items():
			if subgoal not in subgoals_dist_info:
				subgoals_dist_info[subgoal] = []
			subgoals_dist_info[subgoal].append(dist)

		avg_subgoal_achievement_reward.append(len([k for k, v in subgoal_info.items() if v]) * subgoal_reward)
		
		if log_traj:
			
			# Save the last state of the episode as image
			save_env_img(worker.env, path_to_save=os.path.join(log_dir, f"episode_{i}_last_state.png"))
			
			prev_skills = episode['prev_skills'].numpy() \
				if isinstance(episode['prev_skills'], tf.Tensor) else episode['prev_skills']
			prev_skills = np.argmax(prev_skills[0], axis=1).tolist()
			prev_skills: List[int] = [skill for skill in prev_skills]
			
			curr_skills = episode['curr_skills'].numpy() \
				if isinstance(episode['curr_skills'], tf.Tensor) else episode['curr_skills']
			curr_skills = np.argmax(curr_skills[0], axis=1).tolist()
			curr_skills: List[int] = [skill for skill in curr_skills]
			
			actions = episode['actions'].numpy() if isinstance(episode['actions'], tf.Tensor) else episode['actions']
			actions = actions[0]
			logger.info(f"\n[Episode Num]: {i}")
			
			# Log the trajectory in the form <time_step, prev_skill, curr_skill, action>
			for t in range(len(prev_skills)):
				logger.info(f"<{t}: {prev_skills[t]} -> {curr_skills[t]} -> {actions[t]}>")
			
			# Compute the frequency of each skill
			skill_freq = Counter(prev_skills)
			
			# Log the success and length of the episode
			logger.info(f"[Success]: {success}, [Length]: {length}, [Skill Freq]: {skill_freq}")
		
		i += 1
	
	subgoals_avg_info = {k: np.mean(v) for k, v in subgoals_avg_info.items()}
	subgoals_dist_info = {k: np.mean(v) for k, v in subgoals_dist_info.items()}
	
	return np.mean(total_returns), np.mean(total_timesteps), np.mean(avg_final_goal_dist), np.mean(avg_perc_decrease), \
		np.mean(avg_subgoal_achievement_reward), subgoals_avg_info, subgoals_dist_info
