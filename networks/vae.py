import tensorflow as tf
from keras.layers import Dense, Flatten


class Encoder(tf.keras.Model):
    def __init__(self, g_dim):
        super(Encoder, self).__init__()
        self.flatten = Flatten()
        self.fc1 = Dense(units=256, activation=tf.nn.relu)
        self.fc2 = Dense(units=256, activation=tf.nn.relu)

        self.locs_out = Dense(units=g_dim, activation=tf.nn.relu)
        self.std_out = Dense(units=g_dim, activation=tf.nn.softplus)

    def call(self, achieved_goal, curr_state, final_goal):
        ip = tf.concat([achieved_goal, curr_state, final_goal], axis=1)
        h = self.fc1(ip)
        h = self.fc2(h)

        locs = self.locs_out(h)

        # it is better to model std_dev as log(std_dev) as it is more numerically stable to take exponent compared to
        # computing log. Hence, our final KL divergence term is:
        scale = self.std_out(h)
        scale = tf.clip_by_value(scale, clip_value_min=1e-3, clip_value_max=1e3)
        return locs, scale


class Decoder(tf.keras.Model):
    def __init__(self, a_dim, actions_max):
        super(Decoder, self).__init__()

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