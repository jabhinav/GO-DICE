import tensorflow as tf
from keras.layers import Concatenate, Dense


class Discriminator(tf.keras.Model):
    def __init__(self, rew_type: str = 'negative'):
        super(Discriminator, self).__init__()
        kernel_init = tf.keras.initializers.Orthogonal(gain=1.0)
        self.concat = Concatenate()
        # self.normalise = BatchNormalization(axis=0)
        self.fc1 = Dense(units=256, activation=tf.nn.tanh, kernel_initializer=kernel_init)
        self.fc2 = Dense(units=256, activation=tf.nn.tanh, kernel_initializer=kernel_init)
        self.d_out = Dense(units=1, kernel_initializer=kernel_init)

        self.rew_type: str = rew_type

    def call(self, state, goal, action):
        ip = self.concat([state, goal, action])
        # ip = self.normalise(ip)

        h = self.fc1(ip)
        h = self.fc2(h)
        d_out = self.d_out(h)
        return d_out

    def get_reward(self, state, goal, action):
        # Compute the Discriminator Output
        ip = self.concat([state, goal, action])
        # ip = self.normalise(ip)

        h = self.fc1(ip)
        h = self.fc2(h)
        d_out = self.d_out(h)

        # Convert the output into reward
        if self.rew_type == 'airl':
            return d_out
        elif self.rew_type == 'gail':
            return -tf.math.log(1 - tf.nn.sigmoid(d_out) + 1e-8)
        elif self.rew_type == 'normalized':
            return tf.nn.sigmoid(d_out)
        elif self.rew_type == 'negative':
            return tf.math.log(tf.nn.sigmoid(d_out) + 1e-8)
        else:
            print("Specify the correct reward type")
            raise NotImplementedError
