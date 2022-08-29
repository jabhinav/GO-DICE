import tensorflow as tf
from keras.layers import Dense, Flatten
from tensorflow_probability.python.distributions import Normal


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
	def __init__(self, o_dim):
		super(Decoder, self).__init__()
		
		self.fc1 = Dense(units=256, activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.GlorotUniform())
		self.fc2 = Dense(units=256, activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.GlorotUniform())
		self.fc3 = Dense(units=128, activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.GlorotUniform())
		self.out = Dense(units=o_dim, activation=tf.nn.tanh, kernel_initializer=tf.keras.initializers.GlorotUniform())
	
	def call(self, *ip):
		x = tf.concat([*ip], axis=1)
		h = self.fc1(x)
		h = self.fc2(h)
		h = self.fc3(h)
		op = self.out(h)
		return op
	

class Classifier(tf.keras.Model):
	def __init__(self, y_dim):
		super(Classifier, self).__init__()
		self.out_prob_y = Dense(units=y_dim, activation=tf.nn.softmax)

	def call(self, z):
		prob_y = self.out_prob_y(z)
		return prob_y


class Conditional_Prior(tf.keras.Model):
	def __init__(self, z_dim):
		super(Conditional_Prior, self).__init__()
		self.locs_out = Dense(units=z_dim, activation=tf.nn.relu)
		self.std_out = Dense(units=z_dim, activation=tf.nn.softplus)
	
	def call(self, y, k=None):
		if not k:
			k = 1
		locs = self.locs_out(y)
		scale = self.std_out(y)
		scale = tf.clip_by_value(scale, clip_value_min=1e-3, clip_value_max=1e3)
		prior_z_y = Normal(loc=locs, scale=scale)
		return locs, scale, prior_z_y.sample(sample_shape=k)


class Policy(tf.keras.Model):
	def __init__(self, a_dim, actions_max):
		super(Policy, self).__init__()
		
		self.max_actions = actions_max
		self.fc1 = Dense(units=256, activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.GlorotUniform())
		self.fc2 = Dense(units=256, activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.GlorotUniform())
		self.fc3 = Dense(units=128, activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.GlorotUniform())
		self.a_out = Dense(units=a_dim, activation=tf.nn.tanh, kernel_initializer=tf.keras.initializers.GlorotUniform())
	
	def call(self, *ip):
		x = tf.concat([*ip], axis=1)
		h = self.fc1(x)
		h = self.fc2(h)
		h = self.fc3(h)
		actions = self.a_out(h) * self.max_actions
		return actions
