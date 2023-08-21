import argparse
import os
from utils.env import add_env_config


def get_goalGuidedOptionBC_args(log_dir):
	parser = argparse.ArgumentParser()
	
	parser.add_argument('--do_train', type=bool, default=True,
						help='Whether to train the Goal/Skill Predictor [Assumes pretraining is done]')
	parser.add_argument('--do_eval', type=bool, default=True)
	parser.add_argument('--expert_demos', type=int, default=100, help='Use 100 (GOAL GAIL usage)')
	parser.add_argument('--eval_demos', type=int, default=5, help='Use 5 (unseen demos to validate trained pol)')
	parser.add_argument('--test_demos', type=int, default=5, help='Use 5 (seen demos to verify trained pol)')
	parser.add_argument('--perc_train', type=int, default=0.9)
	
	# Specify Skill Configuration
	parser.add_argument('--num_skills', type=int, default=3)
	parser.add_argument('--teacher_forcing_ratio', type=float, default=1.0)
	parser.add_argument('--temp_min', type=float, default=0.01, help='Minimum temperature for Gumbel Softmax')
	parser.add_argument('--temp_max', type=float, default=10, help='Maximum temperature for Gumbel Softmax')
	parser.add_argument('--temp_decay', type=float, default=0.0005, help='Temperature decay for Gumbel Softmax')
	
	# Specify Environment Configuration
	parser.add_argument('--two_object', type=bool, default=True)
	parser.add_argument('--env_name', type=str, default='OpenAIPickandPlace')
	parser.add_argument('--full_space_as_goal', type=bool, default=False)
	parser.add_argument('--expert_behaviour', type=str, default='0', choices=['0', '1', '2'],
						help='Expert behaviour in two_object env')
	parser.add_argument('--stacking', type=bool, default=False)
	parser.add_argument('--target_in_the_air', type=bool, default=False,
						help='Is only valid in two object task')
	
	# Specify Data Collection Configuration
	parser.add_argument('--horizon', type=int, default=125,
						help='Set 50 for one_obj, 125 for two_obj:0, two_obj:1 and 150 for two_obj:2')
	parser.add_argument('--rollout_terminate', type=bool, default=True,
						help='We retain the success flag=1 for states which satisfy goal condition, '
							 'if set to False success flag will be 0 across traj.')
	parser.add_argument('--buffer_size', type=int, default=int(1e6), help='--')
	
	# Specify Training configuration
	parser.add_argument('--num_epochs', type=int, default=500, help='Recommended Use:- 1000')
	parser.add_argument('--n_batches', type=int, default=50,
						help='Recommended Use:- round it off nearest multiple of 50 (num_trans/batch_size)')
	parser.add_argument('--expert_batch_size', type=int, default=256,
						help='No. of trans to sample from expert_buffer for Policy Training  (GOAL-GAIL uses 96)')
	
	# Logging Configuration
	parser.add_argument('--log_interval', type=int, default=-1, help='Recommended Use num_epochs/10')
	
	# Parameters
	parser.add_argument('--kl_coeff', type=float, default=0.0, help='KL Divergence Coefficient (For Skill Predictor)')
	parser.add_argument('--grad_norm_clip', type=float, default=1.0, help='Gradient Norm Clip')
	parser.add_argument('--underflow_eps', type=int, default=1e-20, help='Avoid log underflow')
	parser.add_argument('--vae_lr', type=float, default=0.001)
	
	# Specify Misc configuration
	parser.add_argument('--use_norm', type=bool, default=False)
	parser.add_argument('--clip_obs', type=float, default=200.0,
						help='Un-normalised i.e. raw Observed Values (State and Goals) are clipped to this value')
	parser.add_argument('--clip_norm', type=float, default=5.0,
						help='Normalised Observed Values (State and Goals) are clipped to this value')
	parser.add_argument('--eps_norm', type=float, default=0.01,
						help='A small value used in the normalizer to avoid numerical instabilities')
	
	# Specify Path Configurations
	parser.add_argument('--dir_data', type=str, default='./pnp_data')
	parser.add_argument('--dir_root_log', type=str, default=log_dir)
	parser.add_argument('--dir_summary', type=str, default=os.path.join(log_dir, 'summary'))
	parser.add_argument('--dir_plot', type=str, default=os.path.join(log_dir, 'plots'))
	parser.add_argument('--dir_param', type=str, default=os.path.join(log_dir, 'models'))
	parser.add_argument('--dir_test', type=str, default='./finetuned_models',
						help='Provide the <path_to_models>')
	
	args = parser.parse_args()
	
	# Load the environment config
	args = add_env_config(args)
	
	# Other Configurations
	args.train_demos = int(args.expert_demos * args.perc_train)
	args.val_demos = args.expert_demos - args.train_demos
	
	return args
