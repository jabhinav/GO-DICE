import os
import logging
import argparse
from models.GoalGAIL import run
from utils.env import get_config_env

logging.basicConfig(filename="./logging/logs", filemode='w',
                    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def get_GoalGAIL_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--do_train', type=bool, default=True)
    parser.add_argument('--do_test', type=bool, default=False)
    parser.add_argument('--test_demos', type=int, default=10, help='Use 10 (trained pol)')

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
    parser.add_argument('--num_demos', type=int, default=100, help='Use 20 (expert pol)')
    parser.add_argument('--collect_episodes', type=int, default=1, help='Use 1 (ddpg pol)')
    parser.add_argument('--random_eps', type=float, default=0.3, help='% of time a random action is taken (ddpg pol)')
    parser.add_argument('--noise_eps', type=float, default=0.15, help='std of gauss noise added to non-random actions '
                                                                      'as a percentage of max_u (ddpg pol)')

    # Specify Training configuration
    parser.add_argument('--outer_iters', type=int, default=100, help='Use 1000')
    parser.add_argument('--num_epochs', type=int, default=5, help='Use 5')
    parser.add_argument('--num_cycles', type=int, default=50, help='Use 50')
    parser.add_argument('--n_batches', type=int, default=40, help='Use 40')
    parser.add_argument('--batch_size', type=int, default=64, help='Try 96/128/256')
    parser.add_argument('--buffer_size', type=int, default=int(1e6), help='--')

    # Specify Optimiser/Loss Configuration
    parser.add_argument('--a_lr', type=float, default=1e-3)
    parser.add_argument('--c_lr', type=float, default=1e-3)
    parser.add_argument('--d_lr', type=float, default=1e-4)
    parser.add_argument('--polyak_tau', type=float, default=0.95, help='polyak averaging coefficient for soft-updates')
    parser.add_argument('--l2_action_penalty', type=float, default=0., help='L2 regularize for policy network')
    parser.add_argument('--anneal_coeff_BC', type=float, default=0., help='Keep it to 0 and do not anneal BC')
    parser.add_argument('--BC_Loss_coeff', type=float, default=0., help='Weight BC Loss')

    # Specify Discriminator Configuration
    parser.add_argument('--n_batches_disc', type=int, default=20, help='Use 0 for bc/her else use 20')
    parser.add_argument('--train_dis_per_rollout', type=bool, default=True)
    parser.add_argument('--rew_type', type=str, default='negative', choices=['negative', 'normalized', 'gail', 'airl'])
    parser.add_argument('--lambd', type=float, default=10.0, help='gradient penalty coefficient for wgan.')

    # Specify HER Transition/Transitional Data Sampling configuration
    parser.add_argument('--relabel_for_policy', type=bool, default=True,
                        help='True for gail_her, False ow')  # TODO: Implement the ow case
    parser.add_argument('--replay_strategy', type=str, default='future')
    parser.add_argument('--replay_k', type=int, default=4)
    parser.add_argument('--gail_weight', type=float, default=0.1, help="the weight before gail reward")
    parser.add_argument('--anneal_disc', type=bool, default=False, help='Whether to anneal disc. reward')
    parser.add_argument('--two_rs', type=bool, default=False, help='set to True when anneal_disc is set to True')
    parser.add_argument('--annealing_coeff', type=float, default=1., help='0.9 if her_bc else 1.')
    parser.add_argument('--q_annealing', type=float, default=1.)

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
    parser.add_argument('--summary_dir', type=str, default='./logging/summary')
    parser.add_argument('--param_dir', type=str, default='./logging/models')

    args = parser.parse_args()

    # Load the environment config
    args = get_config_env(args)

    return args


if __name__ == "__main__":
    args = get_GoalGAIL_args()
    store_data_at = os.path.join(os.getcwd(), 'pnp_data/two_obj_fickle_start.pkl')
    run(args, store_data_path=None)
