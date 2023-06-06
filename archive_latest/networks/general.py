import tensorflow as tf
from keras.layers import Dense, Flatten


class GCEncoder(tf.keras.Model):
    def __init__(self, g_dim, c_dim):
        super(GCEncoder, self).__init__()
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
        

class Encoder(tf.keras.Model):
    def __init__(self, z_dim):
        super(Encoder, self).__init__()
        self.flatten = Flatten()
        self.fc1 = Dense(units=256, activation=tf.nn.relu)
        self.fc2 = Dense(units=256, activation=tf.nn.relu)
        self.fc3 = Dense(units=128, activation=tf.nn.relu)

        self.locs_out = Dense(units=z_dim, activation=tf.nn.relu)
        self.std_out = Dense(units=z_dim, activation=tf.nn.softplus)

    def call(self, *ip):
        x = tf.concat([*ip], axis=1)
        h = self.fc1(x)
        h = self.fc2(h)
        h = self.fc3(h)

        locs = self.locs_out(h)

        # it is better to model std_dev as log(std_dev) as it is more numerically stable to take exponent compared to
        # computing log. Hence, our final KL divergence term is:
        scale = self.std_out(h)
        scale = tf.clip_by_value(scale, clip_value_min=1e-3, clip_value_max=1e3)
        return locs, scale


class Decoder(tf.keras.Model):
    def __init__(self, g_dim):
        super(Decoder, self).__init__()

        self.fc1 = Dense(units=256, activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.GlorotUniform())
        self.fc2 = Dense(units=256, activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.GlorotUniform())
        self.fc3 = Dense(units=128, activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.GlorotUniform())
        self.out = Dense(units=g_dim, activation=tf.nn.tanh, kernel_initializer=tf.keras.initializers.GlorotUniform())

    def call(self, *ip):
        x = tf.concat([*ip], axis=1)
        h = self.fc1(x)
        h = self.fc2(h)
        h = self.fc3(h)
        op = self.out(h)
        return op


class GoalPred(tf.keras.Model):
    def __init__(self, g_dim):
        super(GoalPred, self).__init__()
        self.flatten = Flatten()
        self.fc1 = Dense(units=256, activation=tf.nn.relu)
        self.fc2 = Dense(units=256, activation=tf.nn.relu)
        self.fc3 = Dense(units=128, activation=tf.nn.relu)
        # self.fc4 = Dense(units=128, activation=tf.nn.relu)
        # self.fc5 = Dense(units=64, activation=tf.nn.relu)

        self.g_out = Dense(units=g_dim, activation=tf.nn.tanh)

    def call(self, *ip):
        x = tf.concat([*ip], axis=1)
        h = self.fc1(x)
        h = self.fc2(h)
        h = self.fc3(h)
        # h = self.fc4(h)
        # h = self.fc5(h)
        g = self.g_out(h)
        return g

    
class Policy(tf.keras.Model):
    def __init__(self, a_dim, actions_max):
        super(Policy, self).__init__()

        self.max_actions = actions_max
        self.fc1 = Dense(units=256, activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.GlorotUniform())
        self.fc2 = Dense(units=256, activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.GlorotUniform())
        self.fc3 = Dense(units=128, activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.GlorotUniform())
        # self.fc4 = Dense(units=128, activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.GlorotUniform())
        # self.fc5 = Dense(units=64, activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.GlorotUniform())
        self.a_out = Dense(units=a_dim, activation=tf.nn.tanh, kernel_initializer=tf.keras.initializers.GlorotUniform())

    def call(self, *ip):
        x = tf.concat([*ip], axis=1)
        h = self.fc1(x)
        h = self.fc2(h)
        h = self.fc3(h)
        # h = self.fc4(h)
        # h = self.fc5(h)
        actions = self.a_out(h) * self.max_actions
        return actions
    
    
class Attention(tf.keras.Model):
    def __init__(self, alpha_dim):
        super(Attention, self).__init__()
        self.fc1 = Dense(units=256, activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.GlorotUniform())
        self.fc2 = Dense(units=256, activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.GlorotUniform())
        self.fc3 = Dense(units=128, activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.GlorotUniform())
        self.a_out = Dense(units=alpha_dim, activation=tf.nn.softmax, kernel_initializer=tf.keras.initializers.GlorotUniform())
        
    def call(self, *ip):
        x = tf.concat([*ip], axis=1)
        h = self.fc1(x)
        h = self.fc2(h)
        h = self.fc3(h)
        alpha = self.a_out(h)
        return alpha


class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = Dense(units=256, activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.GlorotUniform())
        self.fc2 = Dense(units=256, activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.GlorotUniform())
        self.fc3 = Dense(units=128, activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.GlorotUniform())
        self.out = Dense(units=1, activation=tf.nn.sigmoid, kernel_initializer=tf.keras.initializers.GlorotUniform())

    def call(self, *ip):
        x = tf.concat([*ip], axis=1)
        h = self.fc1(x)
        h = self.fc2(h)
        h = self.fc3(h)
        op = self.out(h)
        return op
