import argparse
import os
from utils.env import get_config_env


def get_ClassicVAE_args(log_dir):
    parser = argparse.ArgumentParser()

    parser.add_argument('--do_train', type=bool, default=True)
    parser.add_argument('--do_eval', type=bool, default=True)
    parser.add_argument('--expert_demos', type=int, default=100, help='Use 100 (GOAL GAIL usage)')
    parser.add_argument('--eval_demos', type=int, default=5, help='Use 5 (trained pol)')
    parser.add_argument('--perc_train', type=int, default=0.9)

    # Specify Environment Configuration
    parser.add_argument('--env_name', type=str, default='OpenAIPickandPlace')
    parser.add_argument('--full_space_as_goal', type=bool, default=False)
    parser.add_argument('--two_object', type=bool, default=False)
    parser.add_argument('--expert_behaviour', type=str, default='2', choices=['0', '1', '2'],
                        help='Expert behaviour in two_object env')
    parser.add_argument('--stacking', type=bool, default=False)
    parser.add_argument('--target_in_the_air', type=bool, default=False,
                        help='Is only valid in two object task')

    # Specify Data Collection Configuration
    parser.add_argument('--horizon', type=int, default=50,
                        help='Set 50 for one_obj, 100 for two_obj:0, two_obj:1 and 150 for two_obj:2')
    parser.add_argument('--rollout_terminate', type=bool, default=True,
                        help='We retain the success flag=1 for states which satisfy goal condition, '
                             'if set to False success flag will be 0 across traj.')
    parser.add_argument('--buffer_size', type=int, default=int(1e6), help='--')

    # Specify Training configuration
    parser.add_argument('--num_epochs', type=int, default=500, help='Recommended Use 500')
    parser.add_argument('--n_batches', type=int, default=50,
                        help='Recommended Use round it off nearest multiple of 100 (num_trans/batch_size)')
    parser.add_argument('--expert_batch_size', type=int, default=128,
                        help='No. of trans to sample from expert_buffer for Policy Training  (GOAL-GAIL uses 96)')
    
    # Logging Configuration
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--log_interval', type=int, default=10, help='Recommended Use num_epochs/10')
    
    # overload_eps
    parser.add_argument('--underflow_eps', type=int, default=1e-20, help='Avoid log underflow')

    # Specify Optimiser/Loss Configuration
    parser.add_argument('--vae_lr', type=float, default=0.001)

    # Specify HER Transition/Transitional Data Sampling configuration
    parser.add_argument('--relabel_for_policy', type=bool, default=True,
                        help='True for gail_her, False ow')  # TODO: Implement the ow case
    parser.add_argument('--replay_strategy', type=str, default='future')

    # Specify Misc configuration
    parser.add_argument('--clip_obs', type=float, default=200.0,
                        help='Un-normalised i.e. raw Observed Values (State and Goals) are clipped to this value')
    parser.add_argument('--clip_norm', type=float, default=5.0,
                        help='Normalised Observed Values (State and Goals) are clipped to this value')
    parser.add_argument('--eps_norm', type=float, default=0.01,
                        help='A small value used in the normalizer to avoid numerical instabilities')

    # Specify Path Configurations
    parser.add_argument('--root_log_dir', type=str, default=log_dir)
    parser.add_argument('--summary_dir', type=str, default=os.path.join(log_dir, 'summary'))
    parser.add_argument('--plot_dir', type=str, default=os.path.join(log_dir, 'plots'))
    parser.add_argument('--param_dir', type=str, default=os.path.join(log_dir, 'models'))
    parser.add_argument('--test_param_dir', type=str, help='Provide the <path_to_models>')

    args = parser.parse_args()

    # Load the environment config
    args = get_config_env(args)

    return args
