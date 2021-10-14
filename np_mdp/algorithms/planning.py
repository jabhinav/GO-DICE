"""Implements planning techniques for solving MDPs.

Provides a domain independent implementation of policy and value iteration
using numpy.
"""

from typing import Optional

import numpy as np
from tqdm import tqdm
import np_mdp.models.mdp as mdp_lib


def value_iteration(
    transition_model,
    reward_model,
    discount_factor = 0.95,
    max_iteration = 20,
    epsilon = 1e-6,
    v_value_initial = None,
):
  """Implements the value iteration algorithm.

  Args:
    transition_model: A transition model as a numpy 3-d array.
    reward_model: A reward model as a numpy 2-d array.
    discount_factor: MDP discount factor to be used for policy evaluation.
    max_iteration: Maximum number of iterations for policy evaluation.
    epsilon: Desired v-value threshold. Used for termination condition.
    v_value_initial: Optional. Initial guess for V value.

  Returns:
    A tuple of policy, v_value, and q_value.
  """
  num_states, num_actions, _ = transition_model.shape

  if v_value_initial is not None:
    assert v_value_initial.shape == (num_states, ), (
        "Initial V value has incorrect shape.")
    v_value = v_value_initial
  else:
    v_value = np.zeros(num_states)

  iteration_idx = 0
  delta_v = epsilon + 1.
  progress_bar = tqdm(total=max_iteration)
  while (iteration_idx < max_iteration) and (delta_v > epsilon):
    q_value = reward_model + discount_factor * np.einsum(
        'san,n -> sa', transition_model, v_value)
    new_v_value = q_value.max(axis=-1)
    delta_v = np.linalg.norm(new_v_value[:] - v_value[:])
    iteration_idx += 1
    v_value = new_v_value
  progress_bar.close()

  policy = mdp_lib.deterministic_policy_from_q_value(q_value)

  return (policy, v_value, q_value)


def policy_iteration(
    transition_model,
    reward_model,
    discount_factor,
    max_iteration,
    epsilon,
    policy_initial = None,
    v_value_initial = None,
):
  """Implements the policy iteration algorithm.

  Args:
    transition_model: A transition model as a numpy 3-d array.
    reward_model: A reward model as a numpy 2-d array.
    discount_factor: MDP discount factor to be used for policy evaluation.
    max_iteration: Maximum number of iterations for policy evaluation.
    epsilon: Desired v-value threshold. Used for termination condition.
    policy_initial: Optional. A deterministic policy.
    v_value_initial: Optional. Initial guess for V value.

  Returns:
    A tuple of policy, v_value, and q_value.
  """
  num_states, num_actions, _ = transition_model.shape

  if policy_initial is not None:
    policy = policy_initial
  else:
    policy = np.zeros(num_states, dtype=int)

  v_value = v_value_initial

  iteration_idx = 0
  delta_policy = 1
  progress_bar = tqdm(total=max_iteration)
  while (iteration_idx < max_iteration) and (delta_policy > 0):
    v_value = mdp_lib.v_value_from_policy(
        policy=policy,
        transition_model=transition_model,
        reward_model=reward_model,
        discount_factor=discount_factor,
        epsilon=epsilon,
        v_value_initial=v_value,
    )
    q_value = mdp_lib.q_value_from_v_value(
        v_value=v_value,
        transition_model=transition_model,
        reward_model=reward_model,
        discount_factor=discount_factor,
    )
    new_policy = mdp_lib.deterministic_policy_from_q_value(q_value)
    delta_policy = (policy != new_policy).sum()
    policy = new_policy
    iteration_idx += 1
  progress_bar.close()

  return policy, v_value, q_value
