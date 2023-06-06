from typing import Dict, Tuple


def get_buffer_shape(args) -> Dict[str, Tuple[int, ...]]:
    buffer_shape = {
        'prev_goals': (args.horizon, args.ag_dim),
        'prev_skills': (args.horizon, args.c_dim),
        'states': (args.horizon + 1, args.s_dim),
        'env_goals': (args.horizon + 1, args.g_dim),
        'curr_goals': (args.horizon, args.ag_dim),
        'curr_skills': (args.horizon, args.c_dim),
        # 'obj_identifiers': (args.horizon,),
        'states_2': (args.horizon, args.s_dim),
        'actions': (args.horizon, args.a_dim),
        'successes': (args.horizon,),
        'distances': (args.horizon,),
    }
    return buffer_shape

