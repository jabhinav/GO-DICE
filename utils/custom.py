import sys

import tensorflow as tf

import logging

logger = logging.getLogger(__name__)


def state_to_goal(num_objs: int):
	"""
	Converts state to goal. (Achieved Goal Space)
	If obj_identifiers is not None, then it further filters the achieved goals based on the object/skill id.
	"""
	
	@tf.function(experimental_relax_shapes=True)  # Imp otherwise code will be very slow
	def get_goal(states: tf.Tensor, obj_identifiers: tf.Tensor = None):
		
		
		# Get achieved goals
		goals = tf.map_fn(lambda x: x[3: 3 + num_objs * 3], states, fn_output_signature=tf.float32)
		# Above giving ValueError: Shape () must have rank at least 1, correct!
		
		# if obj_identifiers is not None:
		# 	# Further filter the achieved goals [Not required as we will operate in full achieved goal space]
		# 	goals = tf.map_fn(lambda x: x[0][x[1] * 3: 3 + x[1] * 3], (goals, obj_identifiers),
		# 					  fn_output_signature=tf.float32)
		
		return goals
	
	return get_goal


def get_some_tensor(x, y):
	"""
	x: tf.Tensor T x prev_dim x curr_dim
	y: tf.Tensor on-hot (prev_dim,)
	op: tf.Tensor (curr_dim,)
	"""
	# Get t = 0, from x first
	op0 = x[0]  # prev_dim x curr_dim
	# Multiply with y and sum over prev_dim
	y = tf.reshape(y, (-1, 1))  # prev_dim x 1
	op0 = tf.reduce_sum(tf.multiply(op0, y), axis=0)  # curr_dim,


def debug(fn_name, do_debug=False):
	if do_debug:
		print("Tracing", fn_name)
		tf.print("Executing", fn_name)


def _update_pbar_msg(args, pbar, total_timesteps):
	"""Update the progress bar with the current training phase."""
	if total_timesteps < args.start_training_timesteps:
		msg = 'Not Training'
	else:
		msg = 'Training'
	if total_timesteps < args.num_random_actions:
		msg += ' - Exploration'
	else:
		msg += ' - Exploitation'
	if pbar.desc != msg:
		pbar.set_description(msg)


def repurpose_skill_seq(args, skill_seq):
	"""
	Repurpose the skill sequence to be used for training the policy. Use value of wrap_skill_id
	= "0": no change
	= "1": wrap pick/grab/drop:obj_id to pick/grab/drop
	= "2": wrap pick:obj_id to pick/grab/drop:obj_id to obj_id
	:param skill_seq: one-hot skill sequence of shape (n_trajs, horizon, c_dim)
	:return: tensor of shape (n_trajs, horizon, c_dim) and type same as skill_seq
	"""
	if args.env_name != 'OpenAIPickandPlace':
		tf.print("Wrapping skill sequence is currently only supported for PnP tasks!")
		sys.exit(-1)
	
	if args.wrap_level == "0":
		return skill_seq
	elif args.wrap_level == "1":
		# wrap by i = j % 3 where i is the new position of skill originally at j. Dim changes from c_dim to 3
		skill_seq = tf.argmax(skill_seq, axis=-1)
		skill_seq = skill_seq % 3
		# Convert back to one-hot
		skill_seq = tf.one_hot(skill_seq, depth=3)
		return skill_seq
	elif args.wrap_level == "2":
		# wrap such that 0/1/2 -> 0, 3/4/5 -> 1, 6/7/8 -> 2 ... Dim changes from c_dim to self.args.num_objs
		skill_seq = tf.argmax(skill_seq, axis=-1)
		skill_seq = skill_seq // 3
		# Convert back to one-hot
		skill_seq = tf.one_hot(skill_seq, depth=args.num_objs)
		return skill_seq
	else:
		raise NotImplementedError("Invalid value for wrap_skill_id: {}".format(args.wrap_level))