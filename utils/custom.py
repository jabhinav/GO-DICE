import numpy as np
import tensorflow as tf
from her.rollout import RolloutWorker


def state_to_goal(num_objs: int):
	"""
	Converts state to goal. (Achieved Goal Space)
	If obj_identifiers is not None, then it further filters the achieved goals based on the object id.
	"""
	
	@tf.function(experimental_relax_shapes=True)  # Imp otherwise code will be very slow
	def get_goal(states: tf.Tensor, obj_identifiers: tf.Tensor = None):
		goals = tf.map_fn(lambda x: x[3: 3 + num_objs * 3], states, fn_output_signature=tf.float32)
		# Above giving ValueError: Shape () must have rank at least 1, correct!
		
		
		if obj_identifiers is not None:
			# Further filter the achieved goals
			goals = tf.map_fn(lambda x: x[0][x[1] * 3: 3 + x[1] * 3], (goals, obj_identifiers),
							  fn_output_signature=tf.float32)
		
		return goals
	
	return get_goal
	

def evaluate(actor, env, num_episodes=10):
	"""Evaluates the policy.
	
	Args:
	  actor: A policy to evaluate.
	  env: Environment to evaluate the policy on.
	  num_episodes: A number of episodes to average the policy on.
	
	Returns:
	  Averaged reward and a total number of steps.
	"""
	total_timesteps = 0
	total_returns = 0
	
	for _ in range(num_episodes):
		state = env.reset()
		done = False
		while not done:
			action, _, _ = actor(np.array([state]))
			action = action[0].numpy()
			
			next_state, reward, done, _ = env.step(action)
			
			total_returns += reward
			total_timesteps += 1
			state = next_state
	
	return total_returns / num_episodes, total_timesteps / num_episodes


def evaluate_worker(worker: RolloutWorker, num_episodes, expert_assist: bool = False):
	"""Evaluates the policy.

		Args:
		  worker: Rollout worker.
		  num_episodes: A number of episodes to average the policy on.
		  expert_assist: Whether to use expert assistance or not (default: False).
		Returns:
		  Averaged reward and a total number of steps.
		"""

	total_timesteps = []
	total_returns = []
	avg_goal_dist = []
	
	for _ in range(num_episodes):
		episode, stats = worker.generate_rollout(epsilon=0.0, stddev=0.0, expert_assist=expert_assist)
		
		success = stats['ep_success'].numpy() if isinstance(stats['ep_success'], tf.Tensor) else stats['ep_success']
		length = stats['ep_length'].numpy() if isinstance(stats['ep_length'], tf.Tensor) else stats['ep_length']
		goal_dist = episode['distances'][0][-1]
		goal_dist = goal_dist.numpy() if isinstance(goal_dist, tf.Tensor) else goal_dist
		
		total_returns.append(success)
		total_timesteps.append(length)
		avg_goal_dist.append(goal_dist)
	
	return np.mean(total_returns), np.mean(total_timesteps), np.mean(avg_goal_dist)


def debug(fn_name, do_debug=False):
	if do_debug:
		print("Tracing", fn_name)
		tf.print("Executing", fn_name)
