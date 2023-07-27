import copy
import datetime
import json
import os
import pickle
import sys
import time
import numpy as np
from collections import OrderedDict
from typing import Dict, Tuple

import logging
import tensorflow as tf

from configs.DICE import get_DICE_args
from her.replay_buffer import ReplayBufferTf
from her.transitions import sample_transitions
from models.demoDICE import Agent as Agent_demoDICE
from models.skilledDemoDICE import Agent as Agent_skilledDemoDICE
from models.GoFar import Agent as Agent_GoFar
from models.BC import Agent as Agent_BC
from utils.buffer import get_buffer_shape
from utils.custom import state_to_goal, repurpose_skill_seq


get_config = {
	'BC': get_DICE_args,
	# 'goalOptionBC': get_goalGuidedOptionBC_args,
	# 'ValueDICE': get_DICE_args,
	'DemoDICE': get_DICE_args,
	'GoFar': get_DICE_args,
	'SkilledDemoDICE': get_DICE_args,
}

Agents = {
	'BC': Agent_BC,
	# 'goalOptionBC': run_goalOptionBC,
	# 'ValueDICE': run_valueDICE,
	'DemoDICE': Agent_demoDICE,
	'GoFar': Agent_GoFar,
	'SkilledDemoDICE': Agent_skilledDemoDICE,
}


def verify(algo: str):
	current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
	
	# Set the random seeds randomly for more randomness
	np.random.seed(int(time.time()))
	tf.random.set_seed(int(time.time()))
	
	tf.config.run_functions_eagerly(True)
	
	log_dir = os.path.join('./logging', 'verify' + current_time)
	if not os.path.exists(log_dir):
		os.makedirs(log_dir, exist_ok=True)
	
	logging.basicConfig(filename=os.path.join(log_dir, 'logs.txt'), filemode='w',
						format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
						datefmt='%m/%d/%Y %H:%M:%S',
						level=logging.INFO)
	logger = logging.getLogger(__name__)
	
	logger.info("# ################# Verifying ################# #")

	args = get_config[algo](log_dir, db=True)
	args.algo = algo
	args.log_dir = log_dir
	
	logger.info("---------------------------------------------------------------------------------------------")
	config: dict = vars(args)
	config = {key: str(value) for key, value in config.items()}
	config = OrderedDict(sorted(config.items()))
	logger.info(json.dumps(config, indent=4))
	
	# List Model Directories
	model_dirs = []
	for root, dirs, files in os.walk('./logging/offlineILPnPOneExp/SkilledDemoDICE_full'):
		for name in dirs:
			if 'run' in name:
				model_dirs.append(os.path.join(root, name, 'models'))
	
	
	for model_dir in model_dirs:
		
		print("Verifying Model: {}".format(model_dir))
		
		# Create log directory within args.log_dir
		log_dir = os.path.join(args.log_dir, os.path.basename(os.path.dirname(model_dir)))
		if not os.path.exists(log_dir):
			os.makedirs(log_dir, exist_ok=True)
		
		args.log_dir = log_dir
		args.log_wandb = False
		args.visualise_test = True
		
		# ############################################# Verifying ################################################## #
		agent = Agents[algo](args)
		print("\n------------- Verifying Actor at {} -------------".format(model_dir))
		logger.info("Loading Model Weights from {}".format(model_dir))
		agent.load_model(dir_param=model_dir)
		
		# resume_files = os.listdir('./pnp_data/two_obj_0_1_train_env_states')
		# resume_files = [f for f in resume_files if f.endswith('.pkl')]
		# resume_states = []
		# for f in resume_files:
		# 	with open(os.path.join('./pnp_data/two_obj_0_1_train_env_states', f), 'rb') as file:
		# 		resume_states.append(pickle.load(file))
		
		agent.visualise(
			use_expert_skill=False,
			use_expert_action=False,
			resume_states=None,
			num_episodes=5,
		)


def run(debug: bool, algo: str):
	current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
	
	# Set the random seeds randomly for more randomness
	np.random.seed(int(time.time()))
	tf.random.set_seed(int(time.time()))
	
	if debug:
		print("Running in Debug Mode. (db=True)")
	
	tf.config.run_functions_eagerly(debug)
	
	log_dir = os.path.join('./logging', algo, '{}'.format('debug' if debug else 'run') + current_time)
	if not os.path.exists(log_dir):
		os.makedirs(log_dir, exist_ok=True)
	
	logging.basicConfig(filename=os.path.join(log_dir, 'logs.txt'), filemode='w',
						format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
						datefmt='%m/%d/%Y %H:%M:%S',
						level=logging.INFO)
	logger = logging.getLogger(__name__)
	
	logger.info("# ################# Working on Model: \"{}\" ################# #".format(algo))
	
	args = get_config[algo](log_dir, db=debug)
	args.algo = algo
	args.log_dir = log_dir
	
	logger.info("---------------------------------------------------------------------------------------------")
	config: dict = vars(args)
	config = {key: str(value) for key, value in config.items()}
	config = OrderedDict(sorted(config.items()))
	logger.info(json.dumps(config, indent=4))
	
	# # For Debugging [#2]
	# if args.fix_goal and args.fix_object:
	# 	data_prefix = 'fOfG_'
	# elif args.fix_goal and not args.fix_object:
	# 	data_prefix = 'dOfG_'
	# elif args.fix_object and not args.fix_goal:
	# 	data_prefix = 'fOdG_'
	# else:
	# 	data_prefix = 'dOdG_'
	
	# Clear tensorflow graph and cache
	tf.keras.backend.clear_session()
	tf.compat.v1.reset_default_graph()
	
	# ######################################################################################################## #
	# ############################################# DATA LOADING ############################################# #
	# ######################################################################################################## #
	# Load Buffer to store expert data
	n_objs = args.num_objs
	buffer_shape: Dict[str, Tuple[int, ...]] = get_buffer_shape(args)
	
	expert_buffer = ReplayBufferTf(
		buffer_shape, args.buffer_size, args.horizon,
		sample_transitions(args.trans_style, state_to_goal=state_to_goal(n_objs), num_options=args.c_dim),
	)
	offline_buffer = ReplayBufferTf(
		buffer_shape, args.buffer_size, args.horizon,
		sample_transitions(args.trans_style, state_to_goal=state_to_goal(n_objs), num_options=args.c_dim)
	)
	if n_objs == 3:
		if args.stacking:
			expert_data_file = 'stack_three_obj_{}_train.pkl'.format(args.expert_behaviour)
			offline_data_file = 'stack_three_obj_{}_offline.pkl'.format(args.expert_behaviour)
		else:
			expert_data_file = 'three_obj_{}_train.pkl'.format(args.expert_behaviour)
			offline_data_file = 'three_obj_{}_offline.pkl'.format(args.expert_behaviour)
	elif n_objs == 2:
		expert_data_file = 'two_obj_{}_train.pkl'.format(args.expert_behaviour)
		offline_data_file = 'two_obj_{}_offline.pkl'.format(args.expert_behaviour)
	elif n_objs == 1:
		expert_data_file = 'single_obj_train.pkl'
		offline_data_file = 'single_obj_offline.pkl'
	else:
		raise NotImplementedError
	
	expert_data_path = os.path.join(args.dir_data, expert_data_file)
	if args.wandb_project.endswith('Exp'):
		offline_data_path = os.path.join(args.dir_data, expert_data_file)  # For EXP
	else:
		offline_data_path = os.path.join(args.dir_data, offline_data_file)
	
	if not os.path.exists(expert_data_path):
		logger.error("Expert data not found at {}. Please run the data generation script first.".format(expert_data_path))
		sys.exit(-1)
	
	if not os.path.exists(offline_data_path):
		logger.error("Offline data not found at {}. Please run the data generation script first.".format(offline_data_path))
		sys.exit(-1)
	
	# Store the expert data in the expert buffer -> D_E
	logger.info("Loading Expert Demos from {} into Expert Buffer for training.".format(expert_data_path))
	with open(expert_data_path, 'rb') as handle:
		buffered_data = pickle.load(handle)
	
	# [Optional] Reformat the G.T. skill sequences
	curr_skills = repurpose_skill_seq(args, buffered_data['curr_skills'])
	prev_skills = repurpose_skill_seq(args, buffered_data['prev_skills'])
	buffered_data['curr_skills'] = curr_skills
	buffered_data['prev_skills'] = prev_skills
	# Add a new key "has_gt_skill" indicating that the skill is G.T.
	buffered_data['has_gt_skill'] = tf.ones_like(buffered_data['successes'], dtype=tf.float32)
	buffered_data['skill_dec_confidence'] = tf.ones_like(buffered_data['successes'], dtype=tf.float32)
	expert_buffer.load_data_into_buffer(buffered_data=buffered_data, num_demos_to_load=args.expert_demos)
	
	# Store the offline data in the policy buffer for DemoDICE -> D_O
	logger.info("Loading Offline Demos from {} into Offline Buffer for training.".format(offline_data_path))
	with open(offline_data_path, 'rb') as handle:
		buffered_data = pickle.load(handle)
	
	# [Optional] Reformat the G.T. skill sequences
	curr_skills = repurpose_skill_seq(args, buffered_data['curr_skills'])
	prev_skills = repurpose_skill_seq(args, buffered_data['prev_skills'])
	buffered_data['curr_skills'] = curr_skills
	buffered_data['prev_skills'] = prev_skills
	# Add a new key "has_gt_skill" indicating that the skill is G.T.
	buffered_data['has_gt_skill'] = tf.ones_like(buffered_data['successes'], dtype=tf.float32)
	buffered_data['skill_dec_confidence'] = tf.ones_like(buffered_data['successes'], dtype=tf.float32)
	offline_buffer.load_data_into_buffer(buffered_data=buffered_data, num_demos_to_load=args.offline_demos)

	# ########################################################################################################### #
	# ############################################# TRAINING #################################################### #
	# ########################################################################################################### #
	start = time.time()
	
	agent = Agents[args.algo](args, expert_buffer, offline_buffer)
	
	# logger.info("Load Actor Policy from {}".format(args.dir_pre))
	# agent.load_actor(dir_param=args.dir_pre)
	# print("Actor Loaded")
	
	logger.info("Training .......")
	agent.learn()
	logger.info("Done Training in {}".format(str(datetime.timedelta(seconds=time.time() - start))))


if __name__ == "__main__":
	num_runs = 5
	for i in range(num_runs):
		run(debug=False, algo='SkilledDemoDICE')
	# verify(algo='SkilledDemoDICE')

