import tensorflow as tf
from keras.layers import Dense


class Actor(tf.keras.Model):
    def __init__(self, a_dim, actions_max):
        super(Actor, self).__init__()

        self.max_actions = actions_max
        self.fc1 = Dense(units=256, activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.GlorotUniform())
        self.fc2 = Dense(units=256, activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.GlorotUniform())
        self.a_out = Dense(units=a_dim, activation=tf.nn.tanh, kernel_initializer=tf.keras.initializers.GlorotUniform())

    def call(self, curr_state, goal_state):
        ip = tf.concat([curr_state, goal_state], axis=1)
        h = self.fc1(ip)
        h = self.fc2(h)
        actions = self.a_out(h) * self.max_actions
        return actions


class Critic(tf.keras.Model):
    def __init__(self, actions_max):
        super(Critic, self).__init__()

        self.max_actions = actions_max
        self.fc1 = Dense(units=256, activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.GlorotUniform())
        self.fc2 = Dense(units=256, activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.GlorotUniform())
        self.q_out = Dense(units=1, activation=None, kernel_initializer=tf.keras.initializers.GlorotUniform())

    def call(self, state, goal, actions):
        ip = tf.concat([state, goal, actions / self.max_actions], axis=1)
        h = self.fc1(ip)
        h = self.fc2(h)
        q = self.q_out(h)
        return q
