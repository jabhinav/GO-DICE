import argparse
import os
from utils.env import get_config_env


def get_GoalGAIL_args(log_dir):
    parser = argparse.ArgumentParser()

    parser.add_argument('--do_train', type=bool, default=False)
    parser.add_argument('--do_eval', type=bool, default=False)
    parser.add_argument('--do_test', type=bool, default=True)
    parser.add_argument('--eval_demos', type=int, default=10, help='Use 10 (trained pol)')
    parser.add_argument('--test_demos', type=int, default=10, help='Use 10 (trained pol)')
    parser.add_argument('--expert_demos', type=int, default=100, help='Use 20 (GOAL GAIL usage)')

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
    parser.add_argument('--random_eps', type=float, default=0.3,
                        help='prob of taking a random action (ddpg pol). Note do not confuse this with logit for the '
                             'binomial distribution since if considered as a logit i.e.'
                             ' log-odds = log(p/1-p) = 0.3 > 0 would mean p > 0.5 for random action. This is wrong!')
    parser.add_argument('--noise_eps', type=float, default=0.15,
                        help='multiplier of gauss noise added to pred actions as a percentage of max_u (ddpg pol)')
    parser.add_argument('--rollout_terminate', type=bool, default=True,
                        help='We retain the success flag=1 for states which satisfy goal condition, '
                             'if set to False success flag will be 0 across traj.')
    parser.add_argument('--terminate_bootstrapping', type=bool, default=True,
                        help='Used by DDPG to compute target values. i.e. whether to use (1-done) in '
                             'y = r + gamma*(1-done)*Q. '
                             'WE WILL ALWAYS USE (1-done). Thus omitting the use of this control var.')
    parser.add_argument('--buffer_size', type=int, default=int(1e6), help='--')

    # Specify Training configuration
    parser.add_argument('--outer_iters', type=int, default=200, help='Recommended Use 1000')
    parser.add_argument('--num_epochs', type=int, default=5, help='Recommended Use 5')
    parser.add_argument('--num_cycles', type=int, default=50, help='Recommended Use 50')
    parser.add_argument('--rollout_batch_size', type=int, default=1, help='Num_eps_to_collect/cycle. Use 2/MPI_thread')
    parser.add_argument('--n_batches', type=int, default=40, help='Recommended Use 40')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='No. of trans to sample from off_policy_buffer for Policy Training (GOAL-GAIL uses 256)')
    parser.add_argument('--expert_batch_size', type=int, default=96,
                        help='No. of trans to sample from expert_buffer for Policy Training  (GOAL-GAIL uses 96)')
    parser.add_argument('--disc_batch_size', type=int, default=256,
                        help='Same as ddpg Pol batch_size. These many trans we sample from on_policy_buffer and '
                             'expert_buffer for Discriminator Training  (GOAL-GAIL uses 256)')

    # Specify Discriminator Configuration
    parser.add_argument('--use_disc', type=bool, default=True)
    parser.add_argument('--n_batches_disc', type=int, default=20, help='Use 0 for bc/her else use 20')
    parser.add_argument('--train_dis_per_rollout', type=bool, default=True)
    parser.add_argument('--rew_type', type=str, default='negative', choices=['negative', 'normalized', 'gail', 'airl'])
    parser.add_argument('--lambd', type=float, default=10.0, help='gradient penalty coefficient for wgan.')

    # Specify Optimiser/Loss Configuration
    parser.add_argument('--a_lr', type=float, default=1e-4)
    parser.add_argument('--c_lr', type=float, default=1e-4)
    parser.add_argument('--d_lr', type=float, default=1e-4)
    parser.add_argument('--polyak_tau', type=float, default=0.95, help='polyak averaging coefficient for soft-updates')
    parser.add_argument('--l2_action_penalty', type=float, default=0.,
                        help='L2 regularize for policy network, GOAL-GAIL uses 0, DDPG-Expert uses 1.')
    # parser.add_argument('--anneal_coeff_BC', type=float, default=0., help='Keep it to 0 and do not anneal BC')
    parser.add_argument('--BC_Loss_coeff', type=float, default=0.01,
                        help='Weight BC Loss, GOAL-GAIL uses 0, DDPG-Expert uses 1/Num_expert_demos')
    parser.add_argument('--Actor_Loss_coeff', type=float, default=1.,
                        help='Weight Actor Loss,  GOAL-GAIL uses 1, DDPG-Expert uses 0.001')

    # Specify HER Transition/Transitional Data Sampling configuration
    parser.add_argument('--relabel_for_policy', type=bool, default=True,
                        help='True for gail_her, False ow')  # TODO: Implement the ow case
    parser.add_argument('--replay_strategy', type=str, default='future')
    parser.add_argument('--replay_k', type=int, default=4)
    parser.add_argument('--gail_weight', type=float, default=0.1, help="the weight before gail reward")
    parser.add_argument('--anneal_disc', type=bool, default=False, help='Whether to anneal disc. reward')
    parser.add_argument('--two_rs', type=bool, default=False, help='set to True when anneal_disc is set to True')
    parser.add_argument('--annealing_coeff', type=float, default=1., help='0.9 if her_bc else 1. Choose < 1 for anneal')
    parser.add_argument('--q_annealing', type=float, default=1., help='Choose < 1 for anneal as 1^(_) = 1')

    # Specify Misc configuration
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--relative_goals', type=bool, default=False)
    parser.add_argument('--clip_obs', type=float, default=200.0,
                        help='Un-normalised i.e. raw Observed Values (State and Goals) are clipped to this value')
    parser.add_argument('--clip_norm', type=float, default=5.0,
                        help='Normalised Observed Values (State and Goals) are clipped to this value')
    parser.add_argument('--eps_norm', type=float, default=0.01,
                        help='A small value used in the normalizer to avoid numerical instabilities')

    # Specify Path Configurations
    parser.add_argument('--summary_dir', type=str, default=os.path.join(log_dir, 'summary'))
    parser.add_argument('--param_dir', type=str, default=os.path.join(log_dir, 'models'))
    parser.add_argument('--test_param_dir', type=str, help='Provide the <path_to_models>')

    args = parser.parse_args()

    # Load the environment config
    args = get_config_env(args)

    return args
