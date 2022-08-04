import logging
import sys

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


def make_sample_her_transitions(replay_strategy, replay_k, reward_fun=None, env=None, discriminator=None,
                                gail_weight=0., sample_g_first=False, zero_action_p=0., dis_bound=np.inf, two_rs=False,
                                with_termination=True):
    """
    Creates a sample function that can be used for HER experience replay.

    Args:
        replay_strategy (in ['future', 'none']): the HER replay strategy; if set to 'none',
            regular DDPG experience replay is used
        replay_k (int): the ratio between HER replays and regular replays (e.g. k = 4 -> 4 times
            as many HER replays as regular replays are used)
        reward_fun (function): function to re-compute the reward with substituted goals [TODO: ADD LATER IF REQD. !!!!]
        env: Self-explanatory
        discriminator: Discriminator Network provided to compute Disc. reward
        gail_weight: Disc. reward weight
        sample_g_first: If True, select goal at some intermediate time-step and then select time-step preceding the goal
                        else select some time-step and then select goal from future time-step
        zero_action_p: prob. of deciding whether to induce transitions in batched data with actions = [0., 0., 0., 0.]
        dis_bound: bounds on discriminator reward
        two_rs: Whether to retain disc_reward and completion_reward separately else add them up
        with_termination: True to compute completion_reward based on updated goal and achieved goal using the env.'s
                          threshold of reaching the goal
                          else a threshold of 1e-6 is used which is so small that success flag is pre-dominantly set to
                          False (this is not useful for HER)
                          This flag is controlled by rollout_terminate i.e. while rolling out whether we have obtained
                          success flag from env. or voluntarily set to 0 across traj
    """
    if replay_strategy not in ['future', 'none']:
        raise ValueError("Invalid replay strategy: {}".format(replay_strategy))

    # Compute hindsight probability.
    if replay_strategy == 'future':
        future_p = 1 - (1. / (1 + replay_k))
    else:
        future_p = 0.

    def _sample_her_transitions(episodic_data, batch_size_in_transitions):
        """
        We will first select num_episodes (randomly picked) = batch_size_in_transitions from buffer,
        We then decide which of these selected episodes will go through HER (using future_p)
        Args:
            episodic_data: {key: array(buffer_size x T x dim_key)}
            batch_size_in_transitions: self-explanatory

        Returns:
        """
        T = episodic_data['actions'].shape[1]
        rollout_batch_size = episodic_data['actions'].shape[0]  # Total number of episodes
        batch_size = batch_size_in_transitions  # Number of transitions to sample

        # -------------------------------------------------------------------------------------------------------------
        # ------------------------------------- 1) Select which episodes to use -------------------------------------
        successes = episodic_data['successes']
        # Get idx of the transitions at which the episode terminated
        terminate_idxes = np.argmax(np.array(successes), axis=1)
        # If no success, set to last transition
        terminate_idxes[np.logical_not(np.any(np.array(successes), axis=1))] = T - 1
        # Get episode index for each transition to sample
        episode_idxs = np.random.choice(np.arange(rollout_batch_size), size=batch_size,
                                        p=(terminate_idxes + 1) / np.sum(terminate_idxes + 1))
        # Get the terminate index for the selected episodes
        terminate_idxes = terminate_idxes[episode_idxs]

        # -------------------------------------------------------------------------------------------------------------
        # --------------------------------- 2) Select which time steps + goals to use ---------------------------------
        if not sample_g_first:
            # First select some time-step for transitions and then sample goals [for sampled her_indexes]
            # Select future time indexes proportional with probability future_p. These
            # will be used for HER replay by substituting in future goals.

            # Select samples (episodes in the batch) which will go through her
            her_indexes = np.where(np.random.uniform(size=batch_size) < future_p)
            # Get the current time step
            t_samples = np.rint(np.random.random(terminate_idxes.shape) * terminate_idxes).astype(int)
            future_offset = np.random.uniform(size=batch_size) * (terminate_idxes - t_samples)
            # Get the offset for the future time step
            future_offset = future_offset.astype(int)
            # Get the future time steps for those episodes selected for HER
            future_t = (t_samples + future_offset)[her_indexes]
        else:
            # Select goal at some intermediate time-step and then select time-step preceding the goal
            her_indexes = np.arange(batch_size)
            future_t = np.rint(np.random.random(terminate_idxes.shape) * terminate_idxes).astype(int)
            t_samples = np.rint(np.random.random(future_t.shape) * future_t).astype(int)
        # assert ((t_samples <= terminate_idxes).all(), t_samples, terminate_idxes)

        # -------------------------------------------------------------------------------------------------------------
        # ----------------- 3) Select the batch of transitions corresponding to the current time steps -----------------
        transitions = {key: episodic_data[key][episode_idxs, t_samples].copy()
                       for key in episodic_data.keys()}

        # -------------------------------------------------------------------------------------------------------------
        # ------------------ 4) Replace goal with achieved goal but only for the previously-selected ------------------
        # HER transitions (as defined by her_indexes). For the other transitions,
        # keep the original goal.
        if future_p:

            # Replace the goals of the her-selected episodes with achieved goals
            future_ag = episodic_data['achieved_goals'][episode_idxs[her_indexes], future_t]
            transitions['goals'][her_indexes] = future_ag

            # Update the success flag accordingly
            if with_termination:
                transitions['successes'] = np.linalg.norm(transitions['goals'] - transitions['achieved_goals_2'],
                                                          axis=-1) < env.terminal_eps
            else:
                transitions['successes'] = np.linalg.norm(transitions['goals'] - transitions['achieved_goals_2'],
                                                          axis=-1) < 1e-6

        if zero_action_p > 0:
            zero_action_indexes = np.where(np.random.uniform(size=batch_size) < zero_action_p)
            transitions['states_2'][zero_action_indexes] = transitions['states'][zero_action_indexes]
            transitions['goals'][zero_action_indexes] = transitions['achieved_goals'][zero_action_indexes]
            transitions['achieved_goals_2'][zero_action_indexes] = transitions['achieved_goals'][zero_action_indexes]
            transitions['actions'][zero_action_indexes] = 0.

        # -------------------------------------------------------------------------------------------------------------
        # ------------------------ 5) Re-compute reward since we may have substituted the goal ------------------------
        # Here we will actually reward those transitions which achieved some goal in hindsight
        completion_rew = tf.cast(transitions['successes'] * env.goal_weight, tf.float32)
        # completion_rew = tf.reshape(completion_rew, (-1, 1))

        if discriminator is not None and gail_weight != 0.:

            disc_rew = discriminator.get_reward(transitions['states'], transitions['goals'], transitions['actions'])
            disc_rew = gail_weight * tf.clip_by_value(disc_rew, -dis_bound, dis_bound)
            disc_rew = tf.squeeze(disc_rew, axis=1)

            if two_rs:
                transitions['rewards_disc'] = disc_rew
                transitions['rewards'] = completion_rew

            else:
                transitions['rewards'] = disc_rew + completion_rew
        else:
            transitions['rewards'] = completion_rew

        transitions = {
            k: tf.reshape(tf.cast(transitions[k], dtype=tf.float32), shape=(batch_size, *transitions[k].shape[1:]))
            for k in transitions.keys()
        }

        # transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:])
        #                for k in transitions.keys()}
        try:
            assert (transitions['actions'].shape[0] == batch_size_in_transitions)
        except AssertionError:
            logger.error("Something wrong in sampling HER transitions. Check|")
            sys.exit(-1)

        return transitions

    return _sample_her_transitions
