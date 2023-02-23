import argparse
import os
from utils.env import get_config_env


def get_DICE_args(log_dir):
	parser = argparse.ArgumentParser()
	
	parser.add_argument('--do_train', type=bool, default=True,
						help='Whether to train the Goal/Skill Predictor [Assumes pretraining is done]')
	parser.add_argument('--do_eval', type=bool, default=True)
	parser.add_argument('--expert_demos', type=int, default=10, help='Use 100 (GOAL GAIL usage)')
	parser.add_argument('--eval_demos', type=int, default=10, help='Use 5 (unseen demos to validate trained pol)')
	parser.add_argument('--test_demos', type=int, default=5, help='Use 5 (seen demos to verify trained pol)')
	parser.add_argument('--perc_train', type=int, default=0.9)
	
	# # Specify Skill Configuration
	# parser.add_argument('--num_skills', type=int, default=3)
	# parser.add_argument('--temp_min', type=float, default=0.01, help='Minimum temperature for Gumbel Softmax')
	# parser.add_argument('--temp_max', type=float, default=10, help='Maximum temperature for Gumbel Softmax')
	# parser.add_argument('--temp_decay', type=float, default=0.0005, help='Temperature decay for Gumbel Softmax')
	
	# Specify Environment Configuration
	parser.add_argument('--env_name', type=str, default='OpenAIPickandPlace')
	parser.add_argument('--full_space_as_goal', type=bool, default=False)
	parser.add_argument('--two_object', type=bool, default=False)
	parser.add_argument('--expert_behaviour', type=str, default='0', choices=['0', '1', '2'],
						help='Expert behaviour in two_object env')
	parser.add_argument('--stacking', type=bool, default=False)
	parser.add_argument('--target_in_the_air', type=bool, default=False,
						help='Is only valid in two object task')
	parser.add_argument('--fix_goal', type=bool, default=True,
						help='Fix the goal position for one object task')
	parser.add_argument('--fix_object', type=bool, default=False,
						help='Fix the object position for one object task')
	

	parser.add_argument('--horizon', type=int, default=100,
						help='Set 50 for one_obj, 125 for two_obj:0, two_obj:1 and 150 for two_obj:2, '
							 '100 for fOfG and dOfG')

	# Specify Data Collection Configuration
	parser.add_argument('--buffer_size', type=int, default=int(2e5),
						help='Number of transitions to store in buffer (max_time_steps)')
	
	# Specify Training configuration
	parser.add_argument('--max_time_steps', type=int, default=2e5,
						help='Number of time steps to run')
	parser.add_argument('--start_training_timesteps', type=int, default=0,
						help='Number of time steps before starting training')
	parser.add_argument('--updates_per_step', type=int, default=2,
						help='Number of updates per time step')
	parser.add_argument('--num_random_actions', type=int, default=1e4,
						help='Number of steps to do exploration and then exploit - 2e3')
	parser.add_argument('--batch_size', type=int, default=256,
						help='No. of trans to sample from expert_buffer for Policy Training')
	
	# Logging Configuration
	parser.add_argument('--eval_interval', type=int, default=100, help='Recommended Use num_epochs/10')
	
	# Parameters
	parser.add_argument('--discount', type=float, default=0.99, help='Discount used for returns.')
	parser.add_argument('--replay_regularization', type=float, default=0.05,
						help='Replay Regularization Coefficient. Used by both ValueDICE (0.1) and DemoDICE (0.05)')
	
	parser.add_argument('--nu_grad_penalty_coeff', type=float, default=1e-4,
						help='Nu Net Gradient Penalty Coefficient. ValueDICE uses 10.0, DemoDICE uses 1e-4')
	parser.add_argument('--cost_grad_penalty_coeff', type=float, default=10,
						help='Cost Net Gradient Penalty Coefficient')
	
	parser.add_argument('--actor_lr', type=float, default=3e-3)
	parser.add_argument('--critic_lr', type=float, default=3e-4)
	parser.add_argument('--disc_lr', type=float, default=3e-4)
	parser.add_argument('--clip_obs', type=float, default=200.0,
						help='Un-normalised i.e. raw Observed Values (State and Goals) are clipped to this value')
	
	# Specify Path Configurations
	parser.add_argument('--dir_data', type=str, default='./pnp_data')
	parser.add_argument('--dir_root_log', type=str, default=log_dir)
	parser.add_argument('--dir_summary', type=str, default=os.path.join(log_dir, 'summary'))
	parser.add_argument('--dir_plot', type=str, default=os.path.join(log_dir, 'plots'))
	parser.add_argument('--dir_param', type=str, default=os.path.join(log_dir, 'models'))
	parser.add_argument('--dir_post', type=str, default='./finetuned_models',
						help='Provide the <path_to_models>')
	parser.add_argument('--dir_pre', type=str, default='./pretrained_models',
						help='Provide the <path_to_models>')
	
	args = parser.parse_args()
	
	# Load the environment config
	args = get_config_env(args, ag_in_env_goal=False)
	
	# Other Configurations
	args.train_demos = int(args.expert_demos * args.perc_train)
	args.val_demos = args.expert_demos - args.train_demos
	
	return args
