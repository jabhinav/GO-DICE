import logging
import random
from typing import Optional, List

import numpy as np
import tensorflow as tf

from her.rollout import RolloutWorker

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


def evaluate(actor, env, num_episodes=100):
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


def evaluate_worker(
        worker: RolloutWorker,
        num_episodes,
        log_traj: bool = False,
        resume_states: Optional[List[dict]] = None
):
    """Evaluates the policy.

        Args:
          worker: Rollout worker.
          num_episodes: A number of episodes to average the policy on.
          log_traj: Whether to log the skills or not (default: False).
          resume_states: List of Resume states (default: None).
        Returns:
          Averaged reward and a total number of steps.
        """

    total_timesteps = []
    total_returns = []
    avg_final_goal_dist = []
    avg_perc_decrease = []

    i = 0
    exception_count = 0
    while i < num_episodes:

        if resume_states is None:
            # logger.info("No resume init state provided. Randomly initializing the env.")
            resume_state_dict = None
        else:
            logger.info(f"Resume init state provided! Check if you intended to do so.")
            # sample a random state from the list of resume states
            resume_state_dict = random.choice(resume_states)

        try:
            episode, stats = worker.generate_rollout(resume_state_dict=resume_state_dict)
        except Exception as e:
            exception_count += 1
            logger.info(f"Exception occurred: {e}")
            if exception_count < 10:
                continue
            else:
                raise e

        success = stats['ep_success'].numpy() if isinstance(stats['ep_success'], tf.Tensor) else stats['ep_success']
        length = stats['ep_length'].numpy() if isinstance(stats['ep_length'], tf.Tensor) else stats['ep_length']
        init_goal_dist = episode['distances'][0][0]
        final_goal_dist = episode['distances'][0][-1]
        init_goal_dist = init_goal_dist.numpy() if isinstance(init_goal_dist, tf.Tensor) else init_goal_dist
        final_goal_dist = final_goal_dist.numpy() if isinstance(final_goal_dist, tf.Tensor) else final_goal_dist

        perc_decrease = (init_goal_dist - final_goal_dist) / init_goal_dist

        total_returns.append(success)
        total_timesteps.append(length)
        avg_final_goal_dist.append(final_goal_dist)
        avg_perc_decrease.append(perc_decrease)

        if log_traj:
            prev_skills = episode['prev_skills'].numpy() \
                if isinstance(episode['prev_skills'], tf.Tensor) else episode['prev_skills']
            prev_skills = np.argmax(prev_skills[0], axis=1).tolist()
            prev_skills = [skill for skill in prev_skills]

            curr_skills = episode['curr_skills'].numpy() \
                if isinstance(episode['curr_skills'], tf.Tensor) else episode['curr_skills']
            curr_skills = np.argmax(curr_skills[0], axis=1).tolist()
            curr_skills = [skill for skill in curr_skills]

            actions = episode['actions'].numpy() if isinstance(episode['actions'], tf.Tensor) else episode['actions']
            actions = actions[0]
            logger.info(f"\nEpisode Num: {i}")
            # Log the trajectory in the form <time_step, prev_skill, curr_skill, action>
            for t in range(len(prev_skills)):
                logger.info(f"<{t}: {prev_skills[t]} -> {curr_skills[t]} -> {actions[t]}>")

        i += 1

    return np.mean(total_returns), np.mean(total_timesteps), np.mean(avg_final_goal_dist), np.mean(avg_perc_decrease)


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
