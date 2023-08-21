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
from models.GODICE import Agent as Agent_GODICE
from models.GoFar import Agent as Agent_GoFar
from models.BC import Agent as Agent_BC
from models.Expert import Agent as Agent_Expert
from utils.buffer import get_buffer_shape
from utils.custom import state_to_goal, repurpose_skill_seq
from forced_configs import get_multiple_configs


get_config = {
	'BC': get_DICE_args,
	# 'goalOptionBC': get_goalGuidedOptionBC_args,
	# 'ValueDICE': get_DICE_args,
	'DemoDICE': get_DICE_args,
	'GoFar': get_DICE_args,
	'GODICE': get_DICE_args,
	'Expert': get_DICE_args,
}

Agents = {
	'BC': Agent_BC,
	# 'goalOptionBC': run_goalOptionBC,
	# 'ValueDICE': run_valueDICE,
	'DemoDICE': Agent_demoDICE,
	'GoFar': Agent_GoFar,
	'GODICE': Agent_GODICE,
	'Expert': Agent_Expert,
}


def record(algo: str, root_dir):
	current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
	
	# Set the random seeds randomly for more randomness
	np.random.seed(int(time.time()))
	tf.random.set_seed(int(time.time()))
	
	tf.config.run_functions_eagerly(True)
	
	dir_root_log = os.path.join('./logging', 'record' + current_time)
	if not os.path.exists(dir_root_log):
		os.makedirs(dir_root_log, exist_ok=True)
	
	logging.basicConfig(filename=os.path.join(dir_root_log, 'logs.txt'), filemode='w',
						format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
						datefmt='%m/%d/%Y %H:%M:%S',
						level=logging.INFO)
	logger = logging.getLogger(__name__)
	
	logger.info("# ################# Recording ################# #")
	
	args = get_config[algo](algo, dir_root_log, debug=True, forced_config=None)
	args.algo = algo
	args.log_wandb = False
	args.visualise_test = False
	
	logger.info("---------------------------------------------------------------------------------------------")
	config: dict = vars(args)
	config = {key: str(value) for key, value in config.items()}
	config = OrderedDict(sorted(config.items()))
	logger.info(json.dumps(config, indent=4))
	
	# List Model Directories
	model_dirs = []
	for root, dirs, files in os.walk(root_dir):
		for name in dirs:
			if 'run' in name:
				model_dirs.append(os.path.join(root, name, 'models'))
	
	for model_dir in model_dirs:
		
		print("Recording Demos by Model: {}".format(model_dir))
		
		# Create log directory within args.log_dir
		model_log_dir = os.path.join(dir_root_log, os.path.basename(os.path.dirname(model_dir)))
		if not os.path.exists(model_log_dir):
			os.makedirs(model_log_dir, exist_ok=True)
		
		args.log_dir = model_log_dir
		
		# ############################################# Verifying ################################################## #
		agent = Agents[algo](args)
		logger.info("Loading Model Weights from {}".format(model_dir))
		agent.load_model(dir_param=model_dir)
		
		# resume_files = os.listdir('./pnp_data/two_obj_0_1_train_env_states')
		# resume_files = [f for f in resume_files if f.endswith('.pkl')]
		# resume_states = []
		# for f in resume_files:
		# 	with open(os.path.join('./pnp_data/two_obj_0_1_train_env_states', f), 'rb') as file:
		# 		resume_states.append(pickle.load(file))
		
		agent.record(
			use_expert_options=False,
			use_expert_action=False,
			num_episodes=2,
		)
		
		break


def verify(algo: str):
	current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
	
	# Set the random seeds randomly for more randomness
	np.random.seed(int(time.time()))
	tf.random.set_seed(int(time.time()))
	
	tf.config.run_functions_eagerly(False)
	
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
		args.visualise_test = False
		
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
			use_expert_options=False,
			use_expert_action=False,
			resume_states=None,
			num_episodes=5,
		)


def evaluate(algo: str, num_eval_demos=100, eval_with_expert_assist: bool = False):
	current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
	
	# Set the random seeds randomly for more randomness
	np.random.seed(int(time.time()))
	tf.random.set_seed(int(time.time()))
	
	tf.config.run_functions_eagerly(False)
	
	log_dir = os.path.join('./logging', 'evaluate' + current_time)
	if not os.path.exists(log_dir):
		os.makedirs(log_dir, exist_ok=True)
	
	logging.basicConfig(filename=os.path.join(log_dir, 'logs.txt'), filemode='w',
						format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
						datefmt='%m/%d/%Y %H:%M:%S',
						level=logging.INFO)
	logger = logging.getLogger(__name__)
	
	logger.info("# ################# Evaluating ################# #")
	
	args = get_config[algo](algo, log_dir, debug=False, forced_config=None)
	args.log_dir = log_dir
	args.log_wandb = False
	args.visualise_test = False
	
	logger.info("---------------------------------------------------------------------------------------------")
	config: dict = vars(args)
	config = {key: str(value) for key, value in config.items()}
	config = OrderedDict(sorted(config.items()))
	
	logger.info(json.dumps(config, indent=4))
	
	# Clear tensorflow graph and cache
	tf.keras.backend.clear_session()
	tf.compat.v1.reset_default_graph()
	
	# List Model Directories
	model_dirs = []
	for root, dirs, files in os.walk('./logging/offlineILPnPTwoExp/GODICE_none(0.25)_6'):
		for name in dirs:
			if 'run' in name:
				model_dirs.append(os.path.join(root, name, 'models'))
	
	agent = Agents[args.algo](args)
	
	# # To evaluate expert's low level policy with learned option transitions. The way experts are implemented for PnP
	# # tasks uses the same logic of Pick-Grab-Drop across multiple objects. So we can get by using the n-object Agent
	# # with expert assist. For any other implementation, would need to use zero_shot script which targets at one-obj
	# # low level policy reuse in higher order tasks
	if eval_with_expert_assist:
		agent.model.act_w_expert_action = True
		logger.info("IMP: Evaluating Expert's Low Level Policy with Learned Option Transitions")
	
	multi_model_returns = []
	for model_dir in model_dirs:
		logger.info("---------------------------------------------------------------------------------------------")
		logger.info("Evaluating Model: {}".format(model_dir))
		
		agent.load_model(dir_param=model_dir)
		
		# Average Return across multiple episodes
		avg_return, std_dev_return = agent.compute_avg_return(eval_demos=num_eval_demos, avg_of_reward=True)
		multi_model_returns.append(avg_return)
		
		# Option Activation across multiple episodes
		agent.args.dir_plot = os.path.join(agent.args.log_dir, 'plots_' + os.path.basename(os.path.dirname(model_dir)))
		agent.evaluate_trajectory_option_activation(eval_demos=num_eval_demos)
	
	logger.info("Average Returns across multiple models: {}+={}".format(
		np.mean(multi_model_returns),
		np.std(multi_model_returns))
	)
	print("Average Returns across multiple models: {} +- {}".format(
		np.mean(multi_model_returns),
		np.std(multi_model_returns))
	)
	

def run(debug: bool, algo: str, forced_config: dict = None):
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
	
	# First clear previous logging configurations to create new logs for each run
	for handler in logging.root.handlers[:]:
		logging.root.removeHandler(handler)
	logging.basicConfig(filename=os.path.join(log_dir, 'logs.txt'), filemode='w',
						format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
						datefmt='%m/%d/%Y %H:%M:%S',
						level=logging.INFO)
	logger = logging.getLogger(__name__)
	
	logger.info("# ################# Working on Model: \"{}\" ################# #".format(algo))
	
	args = get_config[algo](algo, log_dir, debug=debug, forced_config=forced_config)
	args.log_dir = log_dir
	
	logger.info("---------------------------------------------------------------------------------------------")
	config: dict = vars(args)
	config = {key: str(value) for key, value in config.items()}
	config = OrderedDict(sorted(config.items()))
	
	logger.info(json.dumps(config, indent=4))
	
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
		print("Using Perfect Expert Data for Offline Data from {}".format(expert_data_path))
	else:
		offline_data_path = os.path.join(args.dir_data, offline_data_file)
		print("Using Imperfect Data for Offline Data from {}".format(offline_data_path))
	
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
	
	logger.info("Training .......")
	agent.learn()
	logger.info("Done Training in {}".format(str(datetime.timedelta(seconds=time.time() - start))))


if __name__ == "__main__":
	# _forced_configs = get_multiple_configs()
	# for _config in _forced_configs:
	# 	print("Running with Config: {}".format(_config))
	# 	num_runs = 3
	# 	for i in range(num_runs):
	# 		run(debug=False, algo=_config['algo'], forced_config=_config)
	
	# num_runs = 1
	# for i in range(num_runs):
	# 	run(debug=False, algo='SkilledDemoDICE')
	
	# evaluate(algo='SkilledDemoDICE')
	# verify(algo='SkilledDemoDICE')
	
	record(algo='DemoDICE', root_dir='./logging/offlineILPnPTwoExp/DemoDICE')

