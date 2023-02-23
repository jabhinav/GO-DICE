import tensorflow as tf
from keras.layers import Dense, Flatten
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
    

class SkillPredictor(tf.keras.Model):
    def __init__(self, c_dim):
        super(SkillPredictor, self).__init__()
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
        self.base = tf.keras.Sequential([
            Dense(units=256, activation=tf.nn.relu, kernel_initializer='he_normal'),
            Dense(units=256, activation=tf.nn.relu, kernel_initializer='he_normal'),
            Dense(units=128, activation=tf.nn.relu, kernel_initializer='he_normal'),
            Dense(units=2 * action_dim, kernel_initializer='he_normal')
        ])
        self.MEAN_MIN, self.MEAN_MAX = -7, 7
        self.LOG_STD_MIN, self.LOG_STD_MAX = -5, 2
        self.eps = np.finfo(np.float32).eps
    
    def get_dist_and_mode(self, states):
        out = self.base(states)
        mu, log_std = tf.split(out, num_or_size_splits=2, axis=1)
        
        mu = tf.clip_by_value(mu, self.MEAN_MIN, self.MEAN_MAX)
        log_std = tf.clip_by_value(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = tf.exp(log_std)

        pretanh_action_dist = tfp.distributions.MultivariateNormalDiag(mu, std)
        return pretanh_action_dist, mu
    
    @tf.function
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
        log_probs = tf.expand_dims(log_probs, -1)  # To avoid broadcasting
        return log_probs
    
    @tf.function
    def call(self, states):
        """Computes actions for given inputs.
        Args:
          states: A batch of states.
        Returns:
          A mode action, a sampled action and log probability of the sampled action.
        """
        pretanh_action_dist, mode = self.get_dist_and_mode(states)
        
        # Sample actions from the distribution
        pretanh_actions = pretanh_action_dist.sample()
        actions = tf.tanh(pretanh_actions)
        
        # Compute log probs
        pretanh_log_probs = pretanh_action_dist.log_prob(pretanh_actions)
        log_probs = pretanh_log_probs - tf.reduce_sum(tf.math.log(1 - actions ** 2 + self.eps), axis=-1)
        log_probs = tf.expand_dims(log_probs, -1)  # To avoid broadcasting
        
        return tf.tanh(mode), actions, log_probs
    
    
class oldActor(tf.keras.Model):
    def __init__(self, action_dim):
        super(oldActor, self).__init__()
        self.base = tf.keras.Sequential([
            Dense(units=256, activation=tf.nn.relu, kernel_initializer='orthogonal'),
            Dense(units=256, activation=tf.nn.relu, kernel_initializer='orthogonal'),
            Dense(units=128, activation=tf.nn.relu, kernel_initializer='orthogonal'),
            Dense(units=2 * action_dim, kernel_initializer='orthogonal')
            ])
        self.MEAN_MIN, self.MEAN_MAX = -7, 7
        self.LOG_STD_MIN, self.LOG_STD_MAX = -5, 2
        
    def get_dist_and_mode(self, states):
        out = self.base(states)
        mu, log_std = tf.split(out, num_or_size_splits=2, axis=1)
        
        mu, log_std = tf.nn.tanh(mu), tf.nn.tanh(log_std)
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1)
        std = tf.exp(log_std)
        
        dist = tfp.distributions.TransformedDistribution(
            tfp.distributions.Sample(tfp.distributions.Normal(tf.zeros(mu.shape[:-1]), 1.0), sample_shape=mu.shape[-1:]),
            tfp.bijectors.Chain([
                tfp.bijectors.Tanh(),
                tfp.bijectors.Shift(shift=mu),
                tfp.bijectors.ScaleMatvecDiag(scale_diag=std)])
            )
        
        return dist, mu

    @tf.function
    def get_log_prob(self, states, actions):
        """Evaluate log probs for actions conditioned on states.
        Args:
          states: A batch of states.
          actions: A batch of actions to evaluate log probs on.
        Returns:
          Log probabilities of actions.
        """
        dist, _ = self.get_dist_and_mode(states)
        log_probs = dist.log_prob(actions)
        log_probs = tf.expand_dims(log_probs, -1)  # To avoid broadcasting
        return log_probs

    @tf.function
    def call(self, states):
        """Computes actions for given inputs.
        Args:
          states: A batch of states.
        Returns:
          A mode action, a sampled action and log probability of the sampled action.
        """
        dist, mode = self.get_dist_and_mode(states)
        samples = dist.sample()
        log_probs = dist.log_prob(samples)
        log_probs = tf.expand_dims(log_probs, -1)  # To avoid broadcasting
        return mode, samples, log_probs
        
