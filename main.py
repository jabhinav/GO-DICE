import os
import logging
import argparse
from models.GoalGAIL import run

logging.basicConfig(filename="./logs", filemode='w', format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def get_GoalGAIL_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_num', type=int, default=0)

    # Specify Environment Configuration
    parser.add_argument('--env_name', type=str, default='OpenAIPickandPlace')
    parser.add_argument('--full_space_as_goal', type=bool, default=False)
    parser.add_argument('--two_object', type=bool, default=True)
    parser.add_argument('--expert_behaviour', type=str, default='2', choices=['0', '1', '2'],
                        help='Expert behaviour in two_object env')
    parser.add_argument('--stacking', type=bool, default=False)
    parser.add_argument('--target_in_the_air', type=bool, default=False,
                        help='Is only valid in two object task')

    # Specify Data Collection Configuration
    parser.add_argument('--horizon', type=int, default=150,
                        help='Set 50 for one_obj, 100 for two_obj:0, two_obj:1 and 150 for two_obj:2')
    parser.add_argument('--num_demos', type=int, default=100, help='Use 20 (expert pol)')
    parser.add_argument('--collect_episodes', type=int, default=1, help='Use 1 (ddpg pol)')
    parser.add_argument('--random_eps', type=float, default=0.3, help='% of time a random action is taken (ddpg pol)')
    parser.add_argument('--noise_eps', type=float, default=0.15, help='std of gauss noise added to non-random actions '
                                                                      'as a percentage of max_u (ddpg pol)')

    # Specify Training configuration
    parser.add_argument('--outer_iters', type=int, default=2, help='Use 1000')
    parser.add_argument('--num_epochs', type=int, default=2, help='Use 5')
    parser.add_argument('--num_cycles', type=int, default=2, help='Use 50')
    parser.add_argument('--n_batches', type=int, default=2, help='Use 40')
    parser.add_argument('--batch_size', type=int, default=96, help='Try 96/128/256')
    parser.add_argument('--buffer_size', type=int, default=int(1e6), help='--')
    parser.add_argument('--ac_iter', type=int, default=5)
    parser.add_argument('--a_lr', type=float, default=4e-4)
    parser.add_argument('--c_lr', type=float, default=3e-3)
    parser.add_argument('--tau', type=float, default=0.005, help='soft update for target network')

    # Specify Discriminator Configuration
    parser.add_argument('--n_batches_disc', type=int, default=20, help='Use 0 for bc/her else use 20')
    parser.add_argument('--train_dis_per_rollout', type=bool, default=True)
    parser.add_argument('--d_iter', type=int, default=10)
    parser.add_argument('--d_lr', type=float, default=5e-4)
    parser.add_argument('--rew_type', type=str, default='negative', choices=['negative', 'normalized', 'gail', 'airl'])

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

    # Specify Policy configuration
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lambda', type=float, default=0.95)
    parser.add_argument('--relative_goals', type=bool, default=False)
    parser.add_argument('--clip_obs', type=float, default=200.0)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_GoalGAIL_args()
    store_data_at = os.path.join(os.getcwd(), 'pnp_data/two_obj_fickle_start.pkl')
    run(args, store_data_path=store_data_at)
