"""Defines an abstract MDP class using numpy."""

import abc
from typing import Optional

from absl import logging
import numpy as np
import numpy.testing as npt
import scipy.special as sc
from tqdm import tqdm

from np_mdp.utils.mdp_utils import StateSpace, ActionSpace
from np_mdp.utils.mdp_utils import NpStateSpace, NpActionSpace


class MDP:
  """Abstract MDP class."""
  __metaclass__ = abc.ABCMeta

  def __init__(self, fast_cache_mode: bool = False):
    """Initializes MDP class.

      Args:
        fast_cache_mode: Enables fast caching by avoiding assertions. Note
          this should be true only after verifying the implementation is
          correct.
    """
    self.fast_cache_mode = fast_cache_mode

    # Define state space.
    self.init_statespace()
    self.init_statespace_helper_vars()

    # Define action space.
    self.init_actionspace()
    self.init_actionspace_helper_vars()

    # Define empty transition and reward numpy ndarrays.
    # We do not compute them here, as the computations might be costly.
    # Users can compute them by calling the respective properties.
    self._np_transition_model = None
    self._np_reward_model = None

  @abc.abstractmethod
  def init_statespace(self):
    """Defines MDP state space.

    The MDP class allows definition of factored state spaces.
    The method should define a dictionary mapping state feature indices
    to its state space. The source provides an example implementation.
    The joint state space will be automatically constructed using the
    helper method init_statespace_helper_vars.
    """
    self.dict_factored_statespace = {}
    s0_space = StateSpace()
    s1_space = StateSpace()
    self.dict_factored_statespace = {0: s0_space, 1: s1_space}

  def init_statespace_helper_vars(self):
    """Creates helper variables for the state space."""

    # Retrieve number of states and state factors.
    self.num_state_factors = len(self.dict_factored_statespace)
    self.list_num_states = []
    for idx in range(self.num_state_factors):
      self.list_num_states.append(
          self.dict_factored_statespace.get(idx).num_states)
    self.num_states = np.prod(self.list_num_states)
    logging.info("num_states= %d" % (self.num_states, ))

    # Create mapping from state to state index.
    # Mapping takes state value as inputs and outputs a scalar state index.
    np_list_idx = np.arange(self.num_states, dtype=np.int32)
    self.np_state_to_idx = np_list_idx.reshape(self.list_num_states)

    # Create mapping from state index to state.
    # Mapping takes state index as input and outputs a factored state.
    np_idx_to_state = np.zeros((self.num_states, self.num_state_factors),
                               dtype=np.int32)
    for state, idx in np.ndenumerate(self.np_state_to_idx):
      np_idx_to_state[idx] = state
    self.np_idx_to_state = np_idx_to_state

    # Store state space as numpy state space.
    # This is done to accelerate state-related computations.
    self.statespace = NpStateSpace(np_idx_to_state=self.np_idx_to_state,
                                   np_state_to_idx=self.np_state_to_idx)

    assert self.statespace.num_states == self.num_states, (
        'Encountered mismatch in the size of the state space.')

  @abc.abstractmethod
  def init_actionspace(self):
    """Defines MDP action space.

    The MDP class allows definition of factored action spaces.
    The method should define a dictionary mapping action feature indices
    to its action space. The source provides an example implementation.
    The joint action space will be automatically constructed using the
    helper method init_actionspace_helper_vars.
    """
    self.dict_factored_actionspace = {}
    a0_space = ActionSpace()
    a1_space = ActionSpace()
    self.dict_factored_actionspace = {0: a0_space, 1: a1_space}

  def init_actionspace_helper_vars(self):
    """Creates helper variables for the action space."""

    # Retrieve number of actions and action factors.
    self.num_action_factors = len(self.dict_factored_actionspace)
    self.list_num_actions = []
    for idx in range(self.num_action_factors):
      self.list_num_actions.append(
          self.dict_factored_actionspace.get(idx).num_actions)
    self.num_actions = np.prod(self.list_num_actions)
    logging.info("num_actions= %d" % (self.num_actions, ))

    # Create mapping from action to action index.
    # Mapping takes action value as inputs and outputs a scalar action index.
    np_list_idx = np.arange(self.num_actions, dtype=np.int32)
    self.np_action_to_idx = np_list_idx.reshape(self.list_num_actions)

    # Create mapping from action index to action.
    # Mapping takes action index as input and outputs a factored action.
    np_idx_to_action = np.zeros((self.num_actions, self.num_action_factors),
                                dtype=np.int32)
    for action, idx in np.ndenumerate(self.np_action_to_idx):
      np_idx_to_action[idx] = action
    self.np_idx_to_action = np_idx_to_action

    # Store action space as numpy action space.
    # This is done to accelerate action-related computations.
    self.actionspace = NpActionSpace(np_idx_to_action=self.np_idx_to_action,
                                     np_action_to_idx=self.np_action_to_idx)

    assert self.actionspace.num_actions == self.num_actions, (
        'Encountered mismatch in the size of the action space.')

  @abc.abstractmethod
  def transition_model(self, state_idx: int, action_idx: int) -> np.ndarray:
    """Defines MDP transition function.

      Args:
        state_idx: Index of an MDP state.
        action_idx: Index of an MDP action.

      Returns:
        A numpy array with two columns and at least one row.
        The first column corresponds to the probability for the next state.
        The second column corresponds to the index of the next state.
    """
    raise NotImplementedError
    return np_next_p_state_idx  # noqa: F821

  @property
  def np_transition_model(self) -> np.ndarray:
    """Returns transition model as a np ndarray."""

    # If already computed, return the computed value.
    # This model does not change after the MDP is defined.
    if self._np_transition_model is not None:
      return self._np_transition_model

    # Else: Compute using the transition model.
    self._np_transition_model = np.zeros(
        (self.num_states, self.num_actions, self.num_states), dtype=np.float32)
    for state in tqdm(range(self.num_states)):
      for action in range(self.num_actions):
        np_next_p_state_idx = self.transition_model(state, action)
        if self.fast_cache_mode:
          self._np_transition_model[state, action,
                                    tuple(np_next_p_state_idx[:, 1].astype(int)
                                          )] = (np_next_p_state_idx[:, 0])
        else:
          for (next_p, next_state) in np_next_p_state_idx:
            next_state = int(next_state)
            self._np_transition_model[state, action, next_state] = next_p
          npt.assert_almost_equal(
              actual=self._np_transition_model[state, action].sum(),
              desired=1,
              decimal=7,
              err_msg="Transition probabilities do not sum to one.",
          )
    return self._np_transition_model

  def transition(self, state_idx: int, action_idx: int) -> int:
    """Samples next state using the MDP transition function / model.

    Args:
      state_idx: Index of an MDP state.
      action_idx: Index of an MDP action.

    Returns:
      next_state_idx, the index of the next state.
    """
    next_state_distribution = self.transition_model(state_idx, action_idx)
    next_state_idx = np.random.choice(next_state_distribution[:, 1],
                                      size=1,
                                      replace=False,
                                      p=next_state_distribution[:, 0])
    return int(next_state_idx)

  @abc.abstractmethod
  def reward(self, state_idx: int, action_idx: int) -> float:
    """Defines MDP reward function.

      Args:
        state_idx: Index of an MDP state.
        action_idx: Index of an MDP action.

      Returns:
        A scalar reward.
    """
    raise NotImplementedError

  @property
  def np_reward_model(self):
    """Returns reward model as a np ndarray."""
    # If already computed, return the computed value.
    # This model does not change after the MDP is defined.
    if self._np_reward_model is not None:
      return self._np_reward_model

    # Else: Compute using the reward method.
    self._np_reward_model = np.zeros((self.num_states, self.num_actions))
    for state in tqdm(range(self.num_states)):
      for action in range(self.num_actions):
        self._np_reward_model[state, action] = self.reward(state, action)
    return self._np_reward_model


def v_value_from_q_value(q_value: np.ndarray) -> np.ndarray:
  """Computes V values given Q values.

  Args:
    q_value: A numpy 2-d array of Q values. First dimension should correspond
      to state, the second to action.

  Returns:
    value of a state, V(s), as a numpy 1-d array.
  """
  return q_value.max(axis=-1)


def q_value_from_v_value(
    v_value: np.ndarray,
    transition_model: np.ndarray,
    reward_model: np.ndarray,
    discount_factor: float = 0.95,
) -> np.ndarray:
  """Computes V values given a policy.

  Args:
    v_value: value of a state, V(s), as a numpy 1-d array.
    transition_model: A transition model as a numpy 3-d array.
    reward_model: A reward model as a numpy 2-d array.
    discount_factor: MDP discount factor to be used for policy evaluation.
    max_iteration: Maximum number of iterations for policy evaluation.

  Returns:
    value of a state and action pair, Q(s,a), as a numpy 2-d array.
  """
  q_value = reward_model + discount_factor * np.einsum(
      'san,n -> sa', transition_model, v_value)
  return q_value


def v_value_from_policy(
    policy: np.ndarray,
    transition_model: np.ndarray,
    reward_model: np.ndarray,
    discount_factor: float = 0.95,
    max_iteration: int = 20,
    epsilon: float = 1e-6,
    v_value_initial: Optional[np.ndarray] = None,
) -> np.ndarray:
  """Computes V values given a policy.

  Args:
    policy: A policy. Coule be either deterministic or stochastic.
    transition_model: A transition model as a numpy 3-d array.
    reward_model: A reward model as a numpy 2-d array.
    discount_factor: MDP discount factor to be used for policy evaluation.
    max_iteration: Maximum number of iterations for policy evaluation.
    epsilon: Desired v-value threshold. Used for termination condition.
    v_value_initial: Optional. Initial guess for V value.

  Returns:
    value of a state, V(s), as a numpy 1-d array.
  """
  num_states, num_actions, _ = transition_model.shape

  if policy.ndim == 1:
    stochastic_policy = np.zeros((num_states, num_actions))
    stochastic_policy[np.arange(num_states), policy] = 1.
  elif policy.ndim == 2:
    stochastic_policy = policy
  else:
    raise ValueError("Provided policy has incorrect dimension.")

  if v_value_initial is not None:
    assert v_value_initial.shape == (num_states, ), (
        "Initial V value has incorrect shape.")
    v_value = v_value_initial
  else:
    v_value = np.zeros((num_states))

  iteration_idx = 0
  delta_v = epsilon + 1.
  while (iteration_idx < max_iteration) and (delta_v > epsilon):
    q_value = reward_model + discount_factor * np.einsum(
        'san,n -> sa', transition_model, v_value)
    new_v_value = np.einsum('sa,sa->s', stochastic_policy, q_value)
    delta_v = np.linalg.norm(new_v_value[:] - v_value[:])
    iteration_idx += 1
    v_value = new_v_value

  return v_value


def q_value_from_policy(
    policy: np.ndarray,
    transition_model: np.ndarray,
    reward_model: np.ndarray,
    discount_factor: float = 0.95,
    max_iteration: int = 20,
    epsilon: float = 1e-6,
    v_value_initial: Optional[np.ndarray] = None,
) -> np.ndarray:
  """Computes V values given a policy.

  Args:
    policy: A policy. Coule be either deterministic or stochastic.
    transition_model: A transition model as a numpy 3-d array.
    reward_model: A reward model as a numpy 2-d array.
    discount_factor: MDP discount factor to be used for policy evaluation.
    max_iteration: Maximum number of iterations for policy evaluation.
    epsilon: Desired v-value threshold. Used for termination condition.
    v_value_initial: Optional. Initial guess for V value.

  Returns:
    value of a state and action pair, Q(s,a), as a numpy 2-d array.
  """

  v_value = v_value_from_policy(
      policy=policy,
      transition_model=transition_model,
      reward_model=reward_model,
      discount_factor=discount_factor,
      max_iteration=max_iteration,
      epsilon=epsilon,
      v_value_initial=v_value_initial,
  )

  q_value = q_value_from_v_value(
      v_value=v_value,
      transition_model=transition_model,
      reward_model=reward_model,
      discount_factor=discount_factor,
  )

  return q_value


def deterministic_policy_from_q_value(q_value: np.ndarray) -> np.ndarray:
  """Computes a deterministic policy given Q values.

  Args:
    q_value: A numpy 2-d array of Q values. First dimension should correspond
      to state, the second to action.

  Returns:
    action in a state, policy(s), as a numpy 1-d array.
  """
  return q_value.argmax(axis=-1)


def softmax_policy_from_q_value(q_value: np.ndarray,
                                temperature: float = 1.) -> np.ndarray:
  """Computes a stochastic softmax policy given Q values.

  Args:
    q_value: A numpy 2-d array of Q values. First dimension should correspond
      to state, the second to action.
    temperature: The temperature parameters while computing the softmax. For a
      high temperature, the policy will be uniform over action. For more
      details, see https://en.wikipedia.org/wiki/Softmax_function#Applications

  Returns:
    probability of an action in a state, policy(s,a), as a numpy 2-d array.
  """
  if temperature == 0:
    num_states, num_actions = q_value.shape
    policy = q_value.argmax(axis=-1)
    stochastic_policy = np.zeros((num_states, num_actions))
    stochastic_policy[np.arange(num_states), policy] = 1.
    return stochastic_policy
  else:
    return sc.softmax(q_value / temperature, axis=1)
