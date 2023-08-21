from domains.PnP import MyPnPEnvWrapper
from utils.env import save_env_img
import time

exp_env = MyPnPEnvWrapper(full_space_as_goal=False, num_objs=3,
						  stacking=True, fix_goal=False, fix_object=False)

old_obs = exp_env._env.reset()

# Generate random actions and step before saving the image (creates more interesting images)
for n in range(1):
	action = exp_env._env.action_space.sample()
	obs, reward, done, info = exp_env._env.step(action)
	time.sleep(0.1)
	
save_env_img(exp_env, path_to_save='./PnPx3Stack.png')
# old_state = exp_env._env.sim.get_state()
# old_goal = exp_env._env._current_goal
#
# new_obs = exp_env._env.reset()
# save_env_img(exp_env, path_to_save='./new_env.png')
# new_state = exp_env._env.sim.get_state()
#
# # Reset
# exp_env._env.sim.set_state(old_state)
# exp_env._env.sim.forward()
# exp_env._env._current_goal = old_goal
# reset_obs = exp_env._env._get_obs()
# save_env_img(exp_env, path_to_save='./old_env_reset.png')
#
# print("Old Obs: {}\nNew Obs: {}\nReset Obs: {}".format(old_obs, new_obs, reset_obs))
# print("Reset==New: {}, Reset==Old: {}".format(reset_obs['observation'] == new_obs['observation'], reset_obs['observation'] == old_obs['observation']))
