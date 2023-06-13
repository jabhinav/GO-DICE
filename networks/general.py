import tensorflow as tf
from keras.layers import Dense
from utils.sample import gumbel_softmax_tf
import tensorflow_probability as tfp
import numpy as np


class SGCEncoder(tf.keras.Model):
    """
    To encode previous goal, previous latent mode, and current state
    """
    def __init__(self, g_dim, c_dim):
        super(SGCEncoder, self).__init__()
        self.fc1 = Dense(units=256, activation=tf.nn.relu)
        self.fc2 = Dense(units=256, activation=tf.nn.relu)
        self.fc3 = Dense(units=128, activation=tf.nn.relu)
        
        # Goal prediction
        self.fc4 = Dense(units=128, activation=tf.nn.relu)
        self.g_out = Dense(units=g_dim, activation=tf.nn.tanh)
        
        # Latent mode prediction: predict logits
        self.fc5 = Dense(units=128, activation=tf.nn.relu)
        self.c_out = Dense(units=c_dim, activation=None)
        
    def call(self, *ip):
        x = tf.concat([*ip], axis=1)
        # Encode the state
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        
        # Predict the goal
        g = self.fc4(x)
        g = self.g_out(g)
        
        # Predict the latent mode
        c = self.fc5(x)
        c = self.c_out(c)
        
        return g, c


class GoalPredictor(tf.keras.Model):
    def __init__(self, g_dim):
        super(GoalPredictor, self).__init__()
        self.fc1 = Dense(units=256, activation=tf.nn.relu)
        self.fc2 = Dense(units=256, activation=tf.nn.relu)
        self.fc3 = Dense(units=128, activation=tf.nn.relu)
        self.g_out = Dense(units=g_dim, activation=tf.nn.tanh)

    def call(self, *ip):
        x = tf.concat([*ip], axis=1)
        h = self.fc1(x)
        h = self.fc2(h)
        h = self.fc3(h)
        g = self.g_out(h)
        return g
    

class BCSkillPredictor(tf.keras.Model):
    def __init__(self, c_dim):
        super(BCSkillPredictor, self).__init__()
        self.fc1 = Dense(units=256, activation=tf.nn.relu)
        self.fc2 = Dense(units=256, activation=tf.nn.relu)
        self.fc3 = Dense(units=128, activation=tf.nn.relu)
        self.c_out = Dense(units=c_dim, activation=None)

    def call(self, *ip):
        x = tf.concat([*ip], axis=1)
        h = self.fc1(x)
        h = self.fc2(h)
        h = self.fc3(h)
        c = self.c_out(h)
        return c
    
    
class SkillTerminationPredictor(tf.keras.Model):
    def __init__(self):
        super(SkillTerminationPredictor, self).__init__()
        self.fc1 = Dense(units=256, activation=tf.nn.relu)
        self.fc2 = Dense(units=256, activation=tf.nn.relu)
        self.fc3 = Dense(units=128, activation=tf.nn.relu)
        self.t_out = Dense(units=1, activation=tf.nn.sigmoid)

    def call(self, *ip):
        x = tf.concat([*ip], axis=1)
        h = self.fc1(x)
        h = self.fc2(h)
        h = self.fc3(h)
        t = self.t_out(h)
        return t
    
    
class Critic(tf.keras.Model):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = Dense(units=256, activation=tf.nn.relu, kernel_initializer='he_normal')
        self.fc2 = Dense(units=256, activation=tf.nn.relu, kernel_initializer='he_normal')
        self.fc3 = Dense(units=128, activation=tf.nn.relu, kernel_initializer='he_normal')
        self.v_out = Dense(units=1, activation=None, kernel_initializer='he_normal', use_bias=False)

    def call(self, *ip):
        x = tf.concat([*ip], axis=1)
        h = self.fc1(x)
        h = self.fc2(h)
        h = self.fc3(h)
        v = self.v_out(h)
        return v
    
    
class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = Dense(units=256, activation=tf.nn.relu, kernel_initializer='he_normal')
        self.fc2 = Dense(units=256, activation=tf.nn.relu, kernel_initializer='he_normal')
        self.fc3 = Dense(units=128, activation=tf.nn.relu, kernel_initializer='he_normal')
        self.v_out = Dense(units=1, activation=None, kernel_initializer='he_normal', use_bias=False)

    def call(self, *ip):
        x = tf.concat([*ip], axis=1)
        h = self.fc1(x)
        h = self.fc2(h)
        h = self.fc3(h)
        v = self.v_out(h)
        return v


class Actor(tf.keras.Model):
    def __init__(self, action_dim):
        super(Actor, self).__init__()
        # self.base = tf.keras.Sequential([
        #     Dense(units=256, activation=tf.nn.relu, kernel_initializer='he_normal'),
        #     Dense(units=256, activation=tf.nn.relu, kernel_initializer='he_normal'),
        #     Dense(units=128, activation=tf.nn.relu, kernel_initializer='he_normal'),
        #     Dense(units=action_dim, kernel_initializer='he_normal')
        # ])
        
        # Rewrite the base weights to initialise using Xavier(gain=1.0) and bias=0.0
        self.base = tf.keras.Sequential([
            Dense(units=256, activation=tf.nn.relu, kernel_initializer='glorot_uniform', bias_initializer='zeros'),
            Dense(units=256, activation=tf.nn.relu, kernel_initializer='glorot_uniform', bias_initializer='zeros'),
            Dense(units=128, activation=tf.nn.relu, kernel_initializer='glorot_uniform', bias_initializer='zeros'),
            Dense(units=action_dim, kernel_initializer='glorot_uniform', bias_initializer='zeros')
        ])
        
        self.MEAN_MIN, self.MEAN_MAX = -7, 7
        # self.LOG_STD_MIN, self.LOG_STD_MAX = -5, 2
        self.eps = np.finfo(np.float32).eps
        self.pi = tf.constant(np.pi)
        self.FIXED_STD = 0.05
        
        self.train = True
    
    # def get_dist_and_mode(self, states):
    #     out = self.base(states)
    #     mu, log_std = tf.split(out, num_or_size_splits=2, axis=1)
    #     mu, log_std = tf.nn.tanh(mu), tf.nn.tanh(log_std)
    #     mu = tf.clip_by_value(mu, self.MEAN_MIN, self.MEAN_MAX)
    #     log_std = tf.clip_by_value(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
    #     std = tf.exp(log_std)
    #
    #     # log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1)
    #     #
    #     # dist = tfp.distributions.TransformedDistribution(
    #     #     tfp.distributions.Sample(tfp.distributions.Normal(tf.zeros(mu.shape[:-1]), 1.0),
    #     #                              sample_shape=mu.shape[-1:]),
    #     #     tfp.bijectors.Chain([
    #     #         tfp.bijectors.Tanh(),
    #     #         tfp.bijectors.Shift(shift=mu),
    #     #         tfp.bijectors.ScaleMatvecDiag(scale_diag=std)])
    #     # )
    #
    #     return mu, std
    
    def get_log_prob(self, states, actions):
        """Evaluate log probs for actions conditioned on states.
        Args:
          states: A batch of states.
          actions: A batch of actions to evaluate log probs on.
        Returns:
          Log probabilities of actions.
        """
        mu = self.base(states)
        mu = tf.nn.tanh(mu)
        mu = tf.clip_by_value(mu, self.MEAN_MIN, self.MEAN_MAX)
        
        std = tf.ones_like(mu) * self.FIXED_STD
        
        actions = tf.clip_by_value(actions, -1 + self.eps, 1 - self.eps)
        
        # Get log probs from Gaussian distribution
        log_probs = -0.5 * tf.square((actions - mu) / std) - 0.5 * tf.math.log(2 * self.pi) - tf.math.log(std)
        log_probs = tf.reduce_sum(log_probs, axis=1, keepdims=False)
        
        return log_probs
    
    
    def call(self, states, training=None, mask=None):
        """Computes actions for given inputs.
        Args:
          states: A batch of states.
          training: Ignored
          mask: Ignored.
        Returns:
          A mode action, a sampled action and log probability of the sampled action.
        """
        mu = self.base(states)
        mu = tf.nn.tanh(mu)
        mu = tf.clip_by_value(mu, self.MEAN_MIN, self.MEAN_MAX)
        
        if self.train:
            # Sample actions from the distribution
            actions = tf.random.normal(shape=mu.shape, mean=mu, stddev=self.FIXED_STD)
        else:
            actions = mu
            
        # Compute log probs
        log_probs = self.get_log_prob(states, actions)
        # # Sample actions from the distribution
        # actions = dist.sample()
        #
        # # Compute log probs
        # log_probs = dist.log_prob(actions)
        log_probs = tf.expand_dims(log_probs, -1)  # To avoid broadcasting
        
        actions = tf.clip_by_value(actions, -1 + self.eps, 1 - self.eps)
        return mu, actions, log_probs


class Director(tf.keras.Model):
    def __init__(self, skill_dim):
        super(Director, self).__init__()

        # self.base = tf.keras.Sequential([
        #     Dense(units=256, activation=tf.nn.relu, kernel_initializer='he_normal'),
        #     Dense(units=256, activation=tf.nn.relu, kernel_initializer='he_normal'),
        #     Dense(units=128, activation=tf.nn.relu, kernel_initializer='he_normal'),
        #     Dense(units=skill_dim, kernel_initializer='he_normal')
        # ])
        
        # Rewrite the base weights to initialise using Xavier(gain=1.0) and bias=0.0
        self.base = tf.keras.Sequential([
            Dense(units=256, activation=tf.nn.relu, kernel_initializer='glorot_uniform', bias_initializer='zeros'),
            Dense(units=256, activation=tf.nn.relu, kernel_initializer='glorot_uniform', bias_initializer='zeros'),
            Dense(units=128, activation=tf.nn.relu, kernel_initializer='glorot_uniform', bias_initializer='zeros'),
            Dense(units=skill_dim, kernel_initializer='glorot_uniform', bias_initializer='zeros')
        ])
        
        self.train = True

    def get_dist_and_mode(self, states):
        logits = self.base(states)
        return logits
    
    def get_log_prob(self, states, curr_skills=None):
        """Evaluate log probs for current skill conditioned on states.
        Args:
          states: A batch of states.
          curr_skills: A batch of current skills to evaluate log probs on.
        Returns:
          Log probabilities of skills.
        """
        logits = self.get_dist_and_mode(states)
        log_probs = tf.nn.log_softmax(logits, axis=-1)  # (batch_size, skill_dim)
        if curr_skills is not None:
            # Current skills is a one-hot vector
            log_probs = tf.reduce_sum(log_probs * curr_skills, axis=-1)
            
        return log_probs
    
    def call(self, states, training=None, mask=None):
        """Computes skills for given inputs.
        Args:
          states: A batch of states.
          training: Ignored
          mask: Ignored.
          
        Returns:
             (1) skill probability distribution i.e. softmax of logits,
             (2) a sampled skill using Gumbel-Softmax trick and multinomial sampling,
             (3) log probability of the sampled skill.
        """
        logits = self.get_dist_and_mode(states)
        
        if self.train:
            # Sample skills from the distribution. Use Gumbel-Softmax trick to sample from a categorical distribution.
            skills = gumbel_softmax_tf(logits, hard=False)
        else:
            # Get skills from the distribution.
            skills = tf.one_hot(tf.argmax(logits, axis=-1), depth=logits.shape[-1])
            
        # Compute log probs
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        log_probs = tf.reduce_sum(log_probs * skills, axis=-1)
        log_probs = tf.expand_dims(log_probs, -1)  # To avoid broadcasting
        
        return tf.nn.softmax(logits), skills, log_probs
        
        
class SkilledActors(tf.keras.Model):
    def __init__(self, a_dim: int, c_dim: int):
        super(SkilledActors, self).__init__()
        
        self.action_dim = a_dim
        self.skill_dim = c_dim
        
        # For each skill, we have a separate actor
        self.actors = [Actor(a_dim) for _ in range(c_dim)]
        
        # Have a skill predictor for each skill (we call them directors)
        self.directors = [Director(c_dim) for _ in range(c_dim)]
        
        # Target Actors and Directors (Only for evaluation purposes)
        self.target_actors = [Actor(a_dim) for _ in range(c_dim)]
        self.target_directors = [Director(c_dim) for _ in range(c_dim)]
        
        # Set the trainable variables of reference networks to be non-trainable
        for trg_actor in self.target_actors:
            trg_actor.trainable = False
        for trg_director in self.target_directors:
            trg_director.trainable = False
        
        
    def get_variables(self):
        # Function to get variables of all the networks
        return self.trainable_variables
    
    def change_training_mode(self, training_mode: bool):
        """Change the training mode of the model.
        For categorical skills, it changes sampling mode (Gumbel to argmax).
        Args:
          training_mode: A boolean indicating whether to train or not.
        """
        for actor in self.actors:
            actor.train = training_mode
        for skill_predictor in self.directors:
            skill_predictor.train = training_mode

    def get_actor_log_probs(self, states, curr_skills=None, actions=None, use_ref=False):
        """Computes log probabilities of actions for given inputs.
            Args:
              states: A batch of current states.
              curr_skills: A batch of current skills.
              actions: A batch of actions.
              use_ref: A boolean indicating whether to use reference networks or not.
            Returns:
              A batch of log probabilities of the actions.
        """
        actors = self.target_actors if use_ref else self.actors
        op = [actor.get_log_prob(states, actions) for actor in actors]
        log_probs = tf.stack(op, axis=1)  # (batch_size, skill_dim, 1)
        if curr_skills is not None:
            log_probs = tf.reduce_sum(log_probs * curr_skills, axis=1)
        
        log_probs = tf.expand_dims(log_probs, -1)  # To avoid broadcasting
        return log_probs
    
    def call_actor(self, states, curr_skills=None):
        """Computes actions for given states.
        Args:
          states: A batch of current states.
          curr_skills: A batch of current skills.
        Returns:
          A mode action, a sampled action and log probability of the sampled action.
        """
        # Get the action by each skilled actor -> List of (mode, action, log_prob)
        op = [actor(states) for actor in self.actors]
        # Get the mode, action and log_prob for each skill
        modes, actions, log_probs = zip(*op)
        
        # Stack them to get a tensor of shape (batch_size, skill_dim, action_dim)
        modes = tf.stack(modes, axis=1)
        actions = tf.stack(actions, axis=1)
        log_probs = tf.stack(log_probs, axis=1)
        
        # Get the mode, action and log_prob for the given skill
        if curr_skills is not None:
            curr_skills = tf.expand_dims(curr_skills, -1)  # To avoid broadcasting
            modes = tf.reduce_sum(modes * curr_skills, axis=1)
            actions = tf.reduce_sum(actions * curr_skills, axis=1)
            log_probs = tf.reduce_sum(log_probs * curr_skills, axis=1)
        
        return modes, actions, log_probs

    def get_director_log_probs(self, states, prev_skills=None, curr_skills=None, use_ref=False):
        """Computes log probabilities of next skills for given inputs.
            Args:
              states: A batch of current states.
              prev_skills: A batch of previous skills.
              curr_skills: A batch of current skills.
              use_ref: A boolean indicating whether to use reference networks or not.
            Returns:
              A batch of log probabilities of the next skills.
        """
        directors = self.target_directors if use_ref else self.directors
        op = [director.get_log_prob(states, curr_skills) for director in directors]
        log_probs = tf.stack(op, axis=1)  # (batch_size, prev_skill_dim) or (batch_size, prev_skill_dim, curr_skill_dim)
        if prev_skills is not None:
            
            # If log probs is (batch_size, prev_skill_dim)
            if len(log_probs.shape) == 2:
                log_probs = tf.reduce_sum(log_probs * prev_skills, axis=1)  # (batch_size,)
                log_probs = tf.expand_dims(log_probs, -1)  # (batch_size, 1)
            
            # If log probs is (batch_size, prev_skill_dim, curr_skill_dim)
            else:
                log_probs = log_probs * tf.expand_dims(prev_skills, -1)  # (batch_size, prev_skill_dim, curr_skill_dim)
                log_probs = tf.reduce_sum(log_probs, axis=1)  # (batch_size, curr_skill_dim)
        return log_probs

    def call_director(self, states, prev_skills=None):
        """Directs skills for given states.
        Args:
          states: A batch of current states.
          prev_skills: A batch of previous skills.
        Returns:
          The next predicted skill probabilities, a sampled skill and log probability of the sampled skill
        """
        # Get the predicted skill by each skill predictor
        op = [director(states) for director in self.directors]
        # Unpack
        next_skill_probs, next_skills, next_skill_log_probs = zip(*op)
    
        # Stack them to get a tensor of shape (batch_size, (prev)skill_dim, (curr)skill_dim)
        next_skill_probs = tf.stack(next_skill_probs, axis=1)
        next_skills = tf.stack(next_skills, axis=1)
        next_skill_log_probs = tf.stack(next_skill_log_probs, axis=1)
    
        # Get the next skill probabilities, next skill and next skill log probability for the given skill
        if prev_skills is not None:
            prev_skills = tf.expand_dims(prev_skills, -1)  # (batch_size, prev_skill_dim, 1)
            next_skill_probs = tf.reduce_sum(next_skill_probs * prev_skills, axis=1)
            next_skills = tf.reduce_sum(next_skills * prev_skills, axis=1)
            next_skill_log_probs = tf.reduce_sum(next_skill_log_probs * prev_skills, axis=1)
    
        return next_skill_probs, next_skills, next_skill_log_probs

    
    def compute_max_path_viterbi(self, log_probs, init_skill):
        """
        Computes the Viterbi path. Use numpy
        """
        num_decoding_steps = log_probs.shape[0]
        init_skill = np.reshape(init_skill, (1, -1))  # 1 x (prev)skill_dim
        
        # Initialise the forward messages (gather t=0 corresponding to prev_skill=init_skill)
        mu = log_probs[0]  # (prev)skill_dim x (curr)skill_dim
        # Collect the current skill distribution based on prev_skill = init_skill
        mu = np.matmul(init_skill, mu)  # 1 x (curr)skill_dim
        
        mu_path = np.zeros((num_decoding_steps, self.skill_dim), dtype=np.float32)
        # Assign the vectorised initial skill to the max path at t=0
        mu_path[0] = np.argmax(init_skill)
        
        # Decode  c_{0} to c_{t = T-1} (total T steps)
        for t in range(1, num_decoding_steps):
            # Add mu(it's now corresponding to prev_skill) to the log probs to get the next forward message
            accumulate_log_prob_t = np.reshape(mu, (-1, 1)) + log_probs[t]  # (prev)skill_dim x (curr)skill_dim
            mu_path[t] = np.argmax(accumulate_log_prob_t, axis=-2)  # (curr)skill_dim
            mu = np.max(accumulate_log_prob_t, axis=-2)  # (curr)skill_dim
        
        # Backward pass to get the Viterbi path
        path = np.zeros((num_decoding_steps + 1, 1), dtype=np.int32)
        path[-1] = np.argmax(mu)
        log_prob_traj = np.max(mu)
        for t in range(num_decoding_steps, 0, -1):
            path[t-1] = mu_path[t - 1, path[t]]  # Here we are using the max path to get the previous skill
            
        return path, log_prob_traj
    
    def viterbi_decode(self, states, actions, init_skill, use_ref):
        """
        Computes the Viterbi path for the given
        > states (T x s_dim),
        > actions (T x a_dim)
        > init_skill (skill_dim,) one-hot vector
        """
        
        log_pis = self.get_actor_log_probs(states, None, actions, use_ref)  # T x (curr)skill_dim x 1
        log_pis = tf.reshape(log_pis, (states.shape[0], 1, self.skill_dim))  # T x 1 x (curr)skill_dim
        log_trs = self.get_director_log_probs(states, None, None, use_ref)  # T x (prev)skill_dim x (curr)skill_dim
        log_probs = log_pis + log_trs  # T x (prev)skill_dim x (curr)skill_dim
        
        path, log_prob_traj = tf.numpy_function(self.compute_max_path_viterbi,
                                                [log_probs, init_skill],
                                                [tf.int32, tf.float32])
        
        return path, log_prob_traj


class preTanhActor(tf.keras.Model):
    def __init__(self, action_dim):
        super(preTanhActor, self).__init__()
        self.base = tf.keras.Sequential([
            Dense(units=256, activation=tf.nn.relu, kernel_initializer='he_normal'),
            Dense(units=256, activation=tf.nn.relu, kernel_initializer='he_normal'),
            Dense(units=128, activation=tf.nn.relu, kernel_initializer='he_normal'),
            Dense(units=2 * action_dim, kernel_initializer='he_normal')
        ])
        self.MEAN_MIN, self.MEAN_MAX = -7, 7
        self.LOG_STD_MIN, self.LOG_STD_MAX = -5, 2
        self.eps = np.finfo(np.float32).eps
        
        self.train = True
    
    def get_dist_and_mode(self, states):
        out = self.base(states)
        mu, log_std = tf.split(out, num_or_size_splits=2, axis=1)
        
        mu = tf.clip_by_value(mu, self.MEAN_MIN, self.MEAN_MAX)
        log_std = tf.clip_by_value(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = tf.exp(log_std)
        
        pretanh_action_dist = tfp.distributions.MultivariateNormalDiag(mu, std)
        return pretanh_action_dist, mu
    
    def get_log_prob(self, states, actions):
        """Evaluate log probs for actions conditioned on states.
        Args:
          states: A batch of states.
          actions: A batch of actions to evaluate log probs on.
        Returns:
          Log probabilities of actions.
        """
        pretanh_action_dist, _ = self.get_dist_and_mode(states)
        
        pretanh_actions = tf.atanh(tf.clip_by_value(actions, -1 + self.eps, 1 - self.eps))
        pretanh_log_probs = pretanh_action_dist.log_prob(pretanh_actions)
        
        log_probs = pretanh_log_probs - tf.reduce_sum(tf.math.log(1 - actions ** 2 + self.eps), axis=-1)
        return log_probs
    
    def call(self, states, training=None, mask=None):
        """Computes actions for given inputs.
        Args:
          states: A batch of states.
          training: Ignored
          mask: Ignored.
        Returns:
          A mode action, a sampled action and log probability of the sampled action.
        """
        pretanh_action_dist, mode = self.get_dist_and_mode(states)
        
        if self.train:
            # Sample actions from the distribution
            pretanh_actions = pretanh_action_dist.sample()
            actions = tf.tanh(pretanh_actions)
            
            # Compute log probs
            pretanh_log_probs = pretanh_action_dist.log_prob(pretanh_actions)
            log_probs = pretanh_log_probs - tf.reduce_sum(tf.math.log(1 - actions ** 2 + self.eps), axis=-1)
        
        else:
            actions = tf.tanh(mode)
            
            # Compute log probs
            pretanh_log_probs = pretanh_action_dist.log_prob(
                tf.atanh(tf.clip_by_value(actions, -1 + self.eps, 1 - self.eps)))
            log_probs = pretanh_log_probs - tf.reduce_sum(tf.math.log(1 - actions ** 2 + self.eps), axis=-1)
        
        log_probs = tf.expand_dims(log_probs, -1)  # To avoid broadcasting
        
        return tf.tanh(mode), actions, log_probs
