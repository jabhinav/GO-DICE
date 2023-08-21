import argparse
import os

from utils.env import add_env_config, get_horizon


def get_DICE_args(algo: str, log_dir: str, debug: bool = False, forced_config: dict = None):
    parser = argparse.ArgumentParser()

    # Force Environment Configuration
    env_name = 'OpenAIPickandPlace'
    num_objs = 3
    stacking: bool = True
    horizon = get_horizon(num_objs, stack=stacking)

    # General configuration
    parser.add_argument('--algo', type=str, default=algo, choices=['BC', 'GoFar', 'DemoDICE', 'GODICE',
                                                                   'Expert'])
    parser.add_argument('--log_wandb', type=bool, default=not debug)
    # parser.add_argument('--log_wandb', type=bool, default=False)
    parser.add_argument('--wandb_project', type=str, default='GODICE_offlineILPnPStackThreeExp',
                        choices=['GODICE_offlineILPnPDEBUGExp',
                                 'GODICE_offlineILPnPThreeExp',
                                 'GODICE_offlineILPnPStackThreeExp',
                                 'GODICE_offlineILPnPStackThree'
                                 ])

    # Viterbi configuration
    parser.add_argument('--skill_supervision', type=str, default='semi',
                        choices=['full',
                                 'semi',
                                 'none'],
                        help='Type of supervision for latent skills. '
                             'full: Use ground truth skills for offline data.'
                             'semi:x: Use Viterbi to update latent skills for offline data.'
                             'none: Use Viterbi to update latent skills for expert and offline data.')
    parser.add_argument('--num_skills', type=int, default=None,
                        help='Number of skills to use for agent, if provided, will override expert skill set. '
                             'Use only when skill supervision is "none"')
    parser.add_argument('--wrap_level', type=str, default='0', choices=['0', '1', '2'],
                        help='consumed by multi-object expert to determine how to wrap effective skills.'
                             '0: no wrapping, 1: wrap at skill level, 2: wrap at object level')
    parser.add_argument('--skill_dec_conf_interval', type=list, default=[0.01, 1.0],
                        help='Confidence interval for skill decoder, used as multiplier for policy loss')
    # parser.add_argument('--update_skills_interval', type=int, default=1,
    # 					help='Number of time steps after which latent skills will be updated using Viterbi [Unused]')

    parser.add_argument('--expert_demos', type=int, default=50)
    parser.add_argument('--offline_demos', type=int, default=50)
    parser.add_argument('--perc_train', type=int, default=1.0)
    parser.add_argument('--buffer_size', type=int, default=int(2e5),
                        help='Number of transitions to store in buffer (max_time_steps)')

    # Specify Environment Configuration
    parser.add_argument('--env_name', type=str, default=env_name, choices=['OpenAIPickandPlace',
                                                                           'MyPointMassDropNReach'])
    parser.add_argument('--num_objs', type=int, default=num_objs)
    parser.add_argument('--horizon', type=int, default=horizon)
    parser.add_argument('--stacking', type=bool, default=stacking)
    parser.add_argument('--expert_behaviour', type=str, default='0', choices=['0'],
                        help='Expert behaviour in two_object env')
    parser.add_argument('--full_space_as_goal', type=bool, default=False)
    parser.add_argument('--fix_goal', type=bool, default=False,
                        help='[Debugging] Fix the goal position for one object task')
    parser.add_argument('--fix_object', type=bool, default=False,
                        help='[Debugging] Fix the object position for one object task')

    # Specify Training configuration
    parser.add_argument('--max_pretrain_time_steps', type=int, default=0 if not debug else 0,
                        help='No. of time steps to run pretraining - actor, director on expert data. Set to 0 to skip')
    parser.add_argument('--max_time_steps', type=int, default=10000 if not debug else 100,
                        help='No. of time steps to run. Recommended 5k for one_obj, 10k for two_obj')
    parser.add_argument('--batch_size', type=int, default=3 * num_objs * 256,
                        help='No. of trans to sample from buffer for each update')
    parser.add_argument('--trans_style', type=str, default='random_unsegmented',
                        choices=['random_unsegmented', 'random_segmented'],
                        help='How to sample transitions from expert buffer')

    # Polyak
    parser.add_argument('--update_target_interval', type=int, default=50,
                        help='Number of time steps after which target networks will be updated using polyak averaging')
    parser.add_argument('--actor_polyak', type=float, default=0.50,
                        help='Polyak averaging coefficient for actor.')
    parser.add_argument('--director_polyak', type=float, default=0.50,
                        help='Polyak averaging coefficient for director.')
    parser.add_argument('--critic_polyak', type=float, default=0.50,
                        help='Polyak averaging coefficient for critic.')

    # Logging Configuration
    parser.add_argument('--eval_demos', type=int, default=1 if debug else 10)
    parser.add_argument('--test_demos', type=int, default=0)
    parser.add_argument('--eval_interval', type=int, default=200)
    parser.add_argument('--visualise_test', type=bool, default=False, help='Visualise test episodes?')
    parser.add_argument('--subgoal_reward', type=int, default=1, help='Reward for achieving subgoals')

    # # DICE Parameters
    parser.add_argument('--replay_regularization', type=float, default=0.05,
                        help='Replay Regularization Coefficient. Used by both ValueDICE (0.1) and DemoDICE (0.05)')
    parser.add_argument('--discount', type=float, default=0.99, help='Discount used for returns.')
    parser.add_argument('--nu_grad_penalty_coeff', type=float, default=1e-4,
                        help='Nu Net Gradient Penalty Coefficient. ValueDICE uses 10.0, DemoDICE uses 1e-4')
    parser.add_argument('--cost_grad_penalty_coeff', type=float, default=10,
                        help='Cost Net Gradient Penalty Coefficient')
    parser.add_argument('--actor_lr', type=float, default=3e-3)
    parser.add_argument('--critic_lr', type=float, default=3e-4)
    parser.add_argument('--disc_lr', type=float, default=3e-4)
    parser.add_argument('--clip_obs', type=float, default=200.0,
                        help='Un-normalised i.e. raw Observed Values (State and Goals) are clipped to this value')

    # # BC specific parameters
    parser.add_argument('--BC_beta', type=float, default=0.0,
                        help='Coefficient for BC Loss on Offline relative to Expert. Set to 1 for only offline data')

    # Specify Path Configurations
    parser.add_argument('--dir_data', type=str, default=f'./pnp_data/{env_name}/')
    parser.add_argument('--dir_root_log', type=str, default=log_dir)
    parser.add_argument('--dir_summary', type=str, default=os.path.join(log_dir, 'summary'))
    parser.add_argument('--dir_plot', type=str, default=os.path.join(log_dir, 'plots'))
    parser.add_argument('--dir_param', type=str, default=os.path.join(log_dir, 'models'))
    parser.add_argument('--dir_post', type=str, default='./finetuned_models',
                        help='Provide the <path_to_models>')
    parser.add_argument('--dir_pre', type=str, default='./pretrained_models',
                        help='Provide the <path_to_models>')

    args = parser.parse_args()

    # Update the args with forced_config
    if forced_config is not None:
        for key, value in forced_config.items():
            assert hasattr(args, key), f'Forced config key {key} not found in args'
            setattr(args, key, value)

    # Load the environment config
    args = add_env_config(args, ag_in_env_goal=True)

    # Other Configurations
    args.train_demos = int(args.expert_demos * args.perc_train)
    args.val_demos = args.expert_demos - args.train_demos

    # Set number of skills [For unsupervised skill learning]
    if args.num_skills is not None and \
            'none' in args.skill_supervision:
        print('Overriding c_dim with specified %d options' % args.num_skills)
        args.c_dim = args.num_skills

    # Set number of skills [For full or semi-supervised skill learning]
    if args.env_name == 'OpenAIPickandPlace' and \
            args.wrap_level != '0' and \
            'none' not in args.skill_supervision:
        print('Overriding c_dim based on %s' % args.wrap_level)
        if args.wrap_level == '1':
            args.c_dim = 3  # Skill Level Wrap
        elif args.wrap_level == '2':
            args.c_dim = args.num_objs  # Object Level Wrap
        else:
            raise NotImplementedError('Wrap level %s not implemented' % args.wrap_level)

    return args
