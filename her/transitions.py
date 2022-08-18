import logging
import tensorflow_probability as tfp
import numpy as np
import tensorflow as tf
from utils.debug import debug

logger = logging.getLogger(__name__)


@tf.function(experimental_relax_shapes=True)
def sample_her_fwd_bkd_transitions(episodic_data, batch_size_in_transitions=None, do_bkd_transitioning=False):
    debug(fn_name="_sample_her_transitions")
    
    T = episodic_data['actions'].shape[1]
    batch_size = batch_size_in_transitions  # Number of transitions to sample
    
    # -------------------------------------------------------------------------------------------------------------
    # ------------------------------------- 1) Select which episodes to use -------------------------------------
    successes = episodic_data['successes']
    # Get index at which episode terminated
    terminate_idxes = tf.math.argmax(successes, axis=-1)
    # If no success, set to last index
    mask_no_success = tf.math.equal(terminate_idxes, 0)
    terminate_idxes += tf.multiply((T - 1) * tf.ones_like(terminate_idxes),
                                   tf.cast(mask_no_success, terminate_idxes.dtype))
    # Get episode idx for each transition to sample: more likely to sample from episodes which didn't end in success
    p = (terminate_idxes + 1) / tf.reduce_sum(terminate_idxes + 1)
    episode_idxs = tfp.distributions.Categorical(probs=p).sample(sample_shape=(batch_size,))
    episode_idxs = tf.cast(episode_idxs, dtype=terminate_idxes.dtype)
    # Get terminate index for the selected episodes
    terminate_idxes = tf.gather(terminate_idxes, episode_idxs)

    # -------------------------------------------------------------------------------------------------------------
    # --------------------------------- 2) Select which time steps + goals to use ---------------------------------
    # Get the current time step
    t_samples_frac = tf.experimental.numpy.random.random(size=(batch_size,))
    t_samples = t_samples_frac * tf.cast(terminate_idxes, dtype=t_samples_frac.dtype)
    t_samples = tf.cast(tf.round(t_samples), dtype=terminate_idxes.dtype)

    # FWD: Get the offset for the future time-step
    future_offset_frac = tf.experimental.numpy.random.uniform(size=(batch_size,))
    future_offset = future_offset_frac * tf.cast((terminate_idxes - t_samples), future_offset_frac.dtype)
    future_offset = tf.cast(future_offset, terminate_idxes.dtype)
    # Get the future time steps for those episodes selected for HER
    future_t = t_samples + future_offset

    # BKD: Get the offset for the past time-step
    past_offset_frac = tf.experimental.numpy.random.uniform(size=(batch_size,))
    past_t = past_offset_frac * tf.cast(t_samples, dtype=past_offset_frac.dtype)
    past_t = tf.cast(tf.round(past_t), dtype=terminate_idxes.dtype)

    # -------------------------------------------------------------------------------------------------------------
    # ----------------- 3) Select the batch of transitions corresponding to the current time steps ----------------
    curr_indices = tf.stack((episode_idxs, t_samples), axis=-1)
    curr_transitions = {}
    for key in episodic_data.keys():
        curr_transitions[key] = tf.gather_nd(episodic_data[key], indices=curr_indices)
    
    # # -------------------------------------------------------------------------------------------------------------
    # # ---------------------------------- 4.1) Fwd: Replace goal with achieved goal --------------------------------
    future_indices = tf.stack((episode_idxs, future_t), axis=-1)
    future_ag = tf.gather_nd(episodic_data['achieved_goals'], indices=future_indices)
    curr_transitions['inter_goals'] = future_ag
    
    # # ---------------------------------- 4.2) Bkd: Compute Delta_G and past states --------------------------------
    # # NOTE that 'goals' should not be altered and must match terminal goals of each episode
    if do_bkd_transitioning:
        past_transitions = {}
        past_indices = tf.stack((episode_idxs, past_t), axis=-1)
        for key in episodic_data.keys():
            past_transitions[key] = tf.gather_nd(episodic_data[key], indices=past_indices)
        retro_future_ag = tf.gather_nd(episodic_data['achieved_goals'], indices=curr_indices)
        past_transitions['inter_goals'] = retro_future_ag
        
        # Combine the curr, past transitions
        transitions = {}
        for key in curr_transitions.keys():
            transitions[key] = tf.concat((curr_transitions[key], past_transitions[key]), axis=0)
    else:
        transitions = curr_transitions
    
    return transitions


def sample_all_consecutive_transitions(episodic_data, batch_size_in_transitions=None):
    debug(fn_name="unroll_transitions")
    
    num_episodes = episodic_data['actions'].shape[0]
    T = episodic_data['actions'].shape[1]

    successes = episodic_data['successes']
    # Get index at which episode terminated
    terminate_idxes = tf.math.argmax(successes, axis=-1)
    # If no success, set to last index
    mask_no_success = tf.math.equal(terminate_idxes, 0)
    terminate_idxes += tf.multiply((T - 1) * tf.ones_like(terminate_idxes),
                                   tf.cast(mask_no_success, terminate_idxes.dtype))
    
    indices = tf.TensorArray(dtype=terminate_idxes.dtype, size=0, dynamic_size=True)
    
    transition_idx = 0
    
    # TODO: This is giving ValueError: None values not supported. (while tracing num_episodes=None)
    for ep in tf.range(num_episodes, dtype=terminate_idxes.dtype):
        for t in tf.range(terminate_idxes[ep], dtype=terminate_idxes.dtype):
            index = tf.stack((ep, t), axis=-1)
            indices = indices.write(transition_idx, index)
            transition_idx += 1
    
    indices = indices.stack()
    transitions = {}
    for key in episodic_data.keys():
        transitions[key] = tf.gather_nd(episodic_data[key], indices=indices)

    return transitions


@tf.function(experimental_relax_shapes=True)
def sample_rnd_consecutive_transitions(episodic_data, batch_size_in_transitions=None):
    debug(fn_name="_sample_vae_transitions")

    T = episodic_data['actions'].shape[1]
    batch_size = batch_size_in_transitions  # Number of transitions to sample

    # -------------------------------------------------------------------------------------------------------------
    # ------------------------------------- 1) Select which episodes to use -------------------------------------
    successes = episodic_data['successes']
    # Get index at which episode terminated
    terminate_idxes = tf.math.argmax(successes, axis=-1)
    # If no success, set to last index
    mask_no_success = tf.math.equal(terminate_idxes, 0)
    terminate_idxes += tf.multiply((T - 1) * tf.ones_like(terminate_idxes),
                                   tf.cast(mask_no_success, terminate_idxes.dtype))
    # Get episode idx for each transition to sample: equally likely to sample from any episode
    p = (T - 1) * tf.ones_like(terminate_idxes)
    p = (p + 1) / tf.reduce_sum(p + 1)
    episode_idxs = tfp.distributions.Categorical(probs=p).sample(sample_shape=(batch_size,))
    episode_idxs = tf.cast(episode_idxs, dtype=terminate_idxes.dtype)
    # Get terminate index for the selected episodes
    terminate_idxes = tf.gather(terminate_idxes, episode_idxs)

    # -------------------------------------------------------------------------------------------------------------
    # --------------------------------- 2) Select which time steps + goals to use ---------------------------------

    # Select previous state (s_{t-1}): We do not designate this as s_t directly as s_t can coincide with g_{t-1}
    t_samples_frac = tf.experimental.numpy.random.random(size=(batch_size,))
    t_samples = t_samples_frac * tf.cast(terminate_idxes, dtype=t_samples_frac.dtype)
    t_samples = tf.cast(tf.round(t_samples), dtype=terminate_idxes.dtype)

    # --------------------------------------------------------------------------------------------------------------
    # ----------------- 3) Select the batch of transitions corresponding to the selected time steps ----------------
    indices = tf.stack((episode_idxs, t_samples), axis=-1)
    transitions = {}
    for key in episodic_data.keys():
        transitions[key] = tf.gather_nd(episodic_data[key], indices=indices)
        
    # achieved_goal_indices = tf.stack((episode_idxs, future_t), axis=-1)
    # transitions['achieved_goals'] = tf.gather_nd(episodic_data['achieved_goals'], indices=achieved_goal_indices)
    
    return transitions


def make_sample_her_transitions_tf(replay_strategy, replay_k, reward_fun=None, discriminator=None, goal_weight=1.,
                                   gail_weight=0., terminal_eps=0.01, sample_g_first=False, zero_action_p=0.,
                                   dis_bound=np.inf, two_rs=False, with_termination=True):
    """
    Creates a sample function that can be used for HER experience replay.

    Args:
        replay_strategy (in ['future', 'none']): the HER replay strategy; if set to 'none',
            regular DDPG experience replay is used
        replay_k (int): the ratio between HER replays and regular replays (e.g. k = 4 -> 4 times
            as many HER replays as regular replays are used)
        reward_fun (function): function to re-compute the reward with substituted goals [TODO: ADD LATER IF REQD. !!!!]
        goal_weight: Weight given to goal_reached reward
        terminal_eps: Threshold for goal_reached
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

    # 1) experimental_relax_shapes = True (avoids retracing):
    #  it relaxes arg shapes of passing tensors (i.e. episodic_data) which can have different shapes as buffer increases
    # 2) Also Make sure batch_size_in_transitions passed here is a tf.constant to avoid retracing
    @tf.function(experimental_relax_shapes=True)
    def _sample_her_transitions(episodic_data, batch_size_in_transitions):
        """
        We will first select num_episodes (randomly picked) = batch_size_in_transitions from buffer,
        We then decide which of these selected episodes will go through HER (using future_p)
        Args:
            episodic_data: {key: array(buffer_size x T x dim_key)}
            batch_size_in_transitions: self-explanatory

        Returns:
        """
        debug(fn_name="_sample_her_transitions")

        T = episodic_data['actions'].shape[1]
        batch_size = batch_size_in_transitions  # Number of transitions to sample

        # -------------------------------------------------------------------------------------------------------------
        # ------------------------------------- 1) Select which episodes to use -------------------------------------
        successes = episodic_data['successes']
        # Get index at which episode terminated
        terminate_idxes = tf.math.argmax(successes, axis=-1)
        # If no success, set to last index
        mask_no_success = tf.math.equal(terminate_idxes, 0)
        terminate_idxes += tf.multiply((T - 1) * tf.ones_like(terminate_idxes),
                                       tf.cast(mask_no_success, terminate_idxes.dtype))
        # Get episode idx for each transition to sample: more likely to sample from episodes which didn't end in success
        p = (terminate_idxes + 1) / tf.reduce_sum(terminate_idxes + 1)
        episode_idxs = tfp.distributions.Categorical(probs=p).sample(sample_shape=(batch_size,))
        episode_idxs = tf.cast(episode_idxs, dtype=terminate_idxes.dtype)
        # Get terminate index for the selected episodes
        terminate_idxes = tf.gather(terminate_idxes, episode_idxs)

        # -------------------------------------------------------------------------------------------------------------
        # --------------------------------- 2) Select which time steps + goals to use ---------------------------------
        if not sample_g_first:
            # First select some time-step for transitions and then sample goals [by sampling her_indexes]
            # Select future time indexes proportional with probability future_p. These
            # will be used for HER replay by substituting in future goals.

            # Get the current time step
            t_samples_frac = tf.experimental.numpy.random.random(size=(batch_size,))
            t_samples = t_samples_frac * tf.cast(terminate_idxes, dtype=t_samples_frac.dtype)
            t_samples = tf.cast(tf.round(t_samples), dtype=terminate_idxes.dtype)

            # Select samples (samples in the batch) which will go through her
            prob_her = tf.experimental.numpy.random.uniform(size=(batch_size,))
            her_indexes = tf.squeeze(tf.where(prob_her < future_p), axis=-1)
            non_her_indexes = tf.squeeze(tf.where(prob_her >= future_p), axis=-1)

            # Get the offset for the future time step
            future_offset_frac = tf.experimental.numpy.random.uniform(size=(batch_size,))
            future_offset = future_offset_frac * tf.cast((terminate_idxes - t_samples), future_offset_frac.dtype)
            future_offset = tf.cast(future_offset, terminate_idxes.dtype)
            # Get the future time steps for those episodes selected for HER
            future_t = tf.gather(t_samples + future_offset, her_indexes)
        else:
            # Select goal at some intermediate time-step and then select time-step preceding the goal
            her_indexes = tf.range(batch_size)
            non_her_indexes = tf.constant([])

            future_t_frac = tf.experimental.numpy.random.uniform(shape=(batch_size,))
            future_t = future_t_frac * tf.cast(terminate_idxes, dtype=future_t_frac.dtype)
            future_t = tf.cast(tf.round(future_t), dtype=terminate_idxes.dtype)

            t_samples_frac = tf.experimental.numpy.random.random(size=(batch_size,))
            t_samples = t_samples_frac * tf.cast(future_t, dtype=t_samples_frac.dtype)
            t_samples = tf.cast(tf.round(t_samples), dtype=terminate_idxes.dtype)

        # -------------------------------------------------------------------------------------------------------------
        # ----------------- 3) Select the batch of transitions corresponding to the current time steps ----------------
        indices = tf.stack((episode_idxs, t_samples), axis=-1)
        transitions = {}
        for key in episodic_data.keys():
            transitions[key] = tf.gather_nd(episodic_data[key], indices=indices)

        # -------------------------------------------------------------------------------------------------------------
        # ------------------ 4) Replace goal with achieved goal but only for the previously-selected ------------------
        # HER transitions (as defined by her_indexes). For the other transitions,
        # keep the original goal.
        if future_p:
            # # # Replace the goals of the her-selected episodes with achieved goals # # #
            replace_ep_idxs = tf.gather(episode_idxs, her_indexes)
            replace_t_indices = tf.stack((replace_ep_idxs, future_t), axis=-1)
            future_ag = tf.gather_nd(episodic_data['achieved_goals'], indices=replace_t_indices)

            temp_indices = tf.stack((tf.stack((her_indexes, tf.zeros_like(her_indexes)), axis=-1),
                                     tf.stack((her_indexes, tf.ones_like(her_indexes)), axis=-1),
                                     tf.stack((her_indexes, 2*tf.ones_like(her_indexes)), axis=-1)),
                                    axis=1)
            future_ag = tf.scatter_nd(indices=temp_indices, updates=future_ag,
                                      shape=(batch_size, tf.shape(transitions['goals'])[-1]))

            if not tf.equal(tf.size(non_her_indexes), 0):
                retain_goals = tf.gather(transitions['goals'], indices=non_her_indexes)
                temp_indices = tf.stack((tf.stack((non_her_indexes, tf.zeros_like(non_her_indexes)), axis=-1),
                                         tf.stack((non_her_indexes, tf.ones_like(non_her_indexes)), axis=-1),
                                         tf.stack((non_her_indexes, 2 * tf.ones_like(non_her_indexes)), axis=-1)),
                                        axis=1)
                retain_goals = tf.scatter_nd(indices=temp_indices, updates=retain_goals,
                                             shape=(batch_size, tf.shape(transitions['goals'])[-1]))
                transitions['goals'] = retain_goals + future_ag
            else:
                transitions['goals'] = future_ag

            # # # Update the success flag accordingly # # #
            if with_termination:
                new_flags = tf.norm(transitions['goals'] - transitions['achieved_goals_2'], axis=-1) < terminal_eps
            else:
                new_flags = tf.norm(transitions['goals'] - transitions['achieved_goals_2'], axis=-1) < 1e-6

            transitions['successes'] = tf.cast(new_flags, transitions['successes'].dtype)

        # -------------------------------------------------------------------------------------------------------------
        # ------------------------ 5) Re-compute reward since we may have substituted the goal ------------------------
        # Here we will actually reward those transitions which achieved some goal in hindsight
        completion_rew = transitions['successes'] * goal_weight
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

        return transitions

    return _sample_her_transitions


