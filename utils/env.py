from domains.PnP import MyPnPEnvWrapper


def get_PnP_env(args):
	env = MyPnPEnvWrapper(args.full_space_as_goal, two_obj=args.two_object,
						  stacking=args.stacking, target_in_the_air=args.target_in_the_air,
						  fix_goal=args.fix_goal, fix_object=args.fix_object, )
	return env


def save_env_img(env: MyPnPEnvWrapper, path_to_save="./env_curr_state.png"):
	img = env._env.render(mode="rgb_array", width=2000, height=2000)
	from PIL import Image
	im = Image.fromarray(img)
	im.save(path_to_save)


def get_config_env(args, ag_in_env_goal=False):
	env = get_PnP_env(args)
	obs, ag, g = env.reset()
	
	args.g_dim = len(env.current_goal)
	args.s_dim = obs.shape[0]
	args.a_dim = env.action_space.shape[0]
	args.action_max = float(env.action_space.high[0])
	
	args.c_dim = env.latent_dim  # In a way defines number of skills
	if ag_in_env_goal:
		args.ag_dim = args.g_dim  # Achieved Goal in the same space as Env Goal
	else:
		args.ag_dim = 3  # 3D position of the goal in the space of objects
	
	return args
