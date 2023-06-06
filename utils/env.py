from domains.PnP import MyPnPEnvWrapper
from domains.PnPExpert import PnPExpertOneObjImitator, PnPExpertTwoObjImitator, PnPExpertMultiObjImitator

def get_PnP_env(args):
	env = MyPnPEnvWrapper(args.full_space_as_goal, num_objs=args.num_objs, stacking=args.stacking,
						  fix_goal=args.fix_goal, fix_object=args.fix_object)  # For Debugging
	return env


def save_env_img(env: MyPnPEnvWrapper, path_to_save="./env_curr_state.png"):
	img = env._env.render(mode="rgb_array", width=2000, height=2000)
	from PIL import Image
	im = Image.fromarray(img)
	im.save(path_to_save)


def get_config_env(args, ag_in_env_goal):
	"""
	:param args: Namespace object
	:param ag_in_env_goal: If True, then achieved goal is in the same space as env goal
	"""
	env = get_PnP_env(args)
	obs, ag, g = env.reset()
	
	args.g_dim = len(env.current_goal)
	args.s_dim = obs.shape[0]
	args.a_dim = env.action_space.shape[0]
	args.action_max = float(env.action_space.high[0])
	
	# We will overwrite this later by what expert's latent dim is
	# args.c_dim = env.latent_dim  # In a way defines number of effective skills
	
	# Specify the expert's latent skill dimension [Default]
	# Define number of skills, this could be different from agent's practiced skill dimension
	assert hasattr(args, 'num_objs')
	args.c_dim = 3 * args.num_objs
	
	if ag_in_env_goal:
		args.ag_dim = args.g_dim  # Achieved Goal in the same space as Env Goal
	else:
		args.ag_dim = 3  # Goal/Object position in the 3D space
	
	return args


def get_expert(num_objects: int, args):
	if num_objects == 1:
		expert = PnPExpertOneObjImitator(args)
	elif num_objects == 2:
		expert = PnPExpertTwoObjImitator(args)
	elif num_objects == 3:
		expert = PnPExpertMultiObjImitator(args)
	else:
		raise NotImplementedError
	return expert
