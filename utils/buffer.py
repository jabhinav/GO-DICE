def get_buffer_shape(args):
    buffer_shape = {
        'states': (args.horizon + 1, args.s_dim),
        'achieved_goals': (args.horizon+1, args.g_dim),
        'states_2': (args.horizon, args.s_dim),
        'achieved_goals_2': (args.horizon, args.g_dim),
        'goals': (args.horizon, args.g_dim),
        'actions': (args.horizon, args.a_dim),
        'successes': (args.horizon,),
        'distances': (args.horizon, ),
        'latent_modes': (args.horizon, args.c_dim)
    }
    return buffer_shape

