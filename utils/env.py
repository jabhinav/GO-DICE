import collections
import numpy as np
from utils.normalise import Normalizer
from domains.PnP import MyPnPEnvWrapperForGoalGAIL


# def get_env_params(env):
#     obs = env.reset()
#     params = {'obs': obs['observation'].shape[0], 'goal': obs['desired_goal'].shape[0],
#               'action': env.action_space.shape[0], 'action_max': env.action_space.high[0],  # Highest (lowest) values
#               'max_timesteps': env._max_episode_steps, 'latent_mode': 2}
#     return params
#
#
# def get_env(env_name):
#     env = gym.make('FetchPickAndPlace-v1')
#     # env = PnPEnv()
#     env_config = get_env_params(env)
#     return env, env_config


def get_PnP_env(args):
    env = MyPnPEnvWrapperForGoalGAIL(args.full_space_as_goal, two_obj=args.two_object,
                                     stacking=args.stacking, target_in_the_air=args.target_in_the_air)
    return env


def get_config_env(args):
    env = get_PnP_env(args)
    obs, ag, g = env.reset()

    args.g_dim = len(env.current_goal)
    args.s_dim = obs.shape[0]
    args.a_dim = env.action_space.shape[0]
    args.action_max = float(env.action_space.high[0])
    return args


def preprocess_robotic_demos(demos, env_params, window_size=1, clip_range=5):
    # Declare Normaliser
    norm_o = Normalizer(size=env_params['obs'], default_clip_range=clip_range)
    norm_o.update(demos['curr_states'][:, :env_params['obs']])
    norm_o.recompute_stats()
    norm_g = Normalizer(size=env_params['goal'], default_clip_range=clip_range)
    norm_g.update(demos['curr_states'][:, env_params['obs']:])
    norm_g.recompute_stats()

    # Normalise state
    def wrap_normalise(_states):
        _state_obs = norm_o.normalize(_states[:, :env_params['obs']])
        _state_g = norm_g.normalize(_states[:, env_params['obs']:])
        return _state_obs, _state_g

    state_obs, state_g = wrap_normalise(demos['curr_states'])
    normalised_s = np.concatenate([state_obs, state_g], axis=1)

    # Normalise stack state
    normalise_stack_s = []
    s_dim = env_params['obs'] + env_params['goal']
    for w in range(window_size):
        states = demos['curr_stack_states'][:, w*s_dim:(w+1)*s_dim]
        state_obs, state_g = wrap_normalise(states)
        normalise_stack_s.append(state_obs)
        normalise_stack_s.append(state_g)
    normalise_stack_s = np.concatenate(normalise_stack_s, axis=1)

    # Update demos
    demos['curr_states'] = normalised_s
    demos['curr_stack_states'] = normalise_stack_s

    return demos
