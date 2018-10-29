# Author	: Nihesh Anderson K
# File 		: DQN.py

"""
This module implements Deep Q network using keras, required for learning the Q function for atari games
"""


from keras.layers import Dense,Conv2D, Flatten
import numpy as np
from keras.models import Sequential

class DQN:

	# Class variables	
	actions = -1			# No of possible actions
	batch_size = -1			# Number of datapoints to push through in one iteration
	clf = -1				# DQN Model
	discount_factor	= -1	# Discount factor in bellman equation
	EPOCHS = 1				# training epochs

	def __init__(self, no_of_actions, input_dimension, batch_size, discount_factor):

		self.actions = no_of_actions
		self.batch_size = batch_size
		self.discount_factor = discount_factor

		self.clf = Sequential()

		# keras.layers.Conv2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)	
		# Kernel size - Convolution window size
		# Filters - Weights for convolving the window
		# Strides - The distance by which window has to be shifted in x and y
		# Data format - Denotes x-y representation of data

		layers = [
			# First convolutional layer with relu activation
			Conv2D(32, 8, strides = (4,4), padding="valid", activation="relu", input_shape = input_dimension, data_format="channels_first"),
			
			# Second convolutional layer with relu activation
			Conv2D(64, 4, strides = (2,2), padding="valid", activation="relu", input_shape = input_dimension, data_format="channels_first"),
			
			# Third convolutional layer with relu activation
			Conv2D(64, 3, strides = (1,1), padding="valid", activation="relu", input_shape = input_dimension, data_format="channels_first"),
			
			# Flatten the 2D convolved output to 1D
			Flatten(),

			# Add a fully connected dense layer with 512 neurons and relu activation
			Dense(512, activation = "relu"),

			# Output layer with self.actions number of nodes, each representing the magnitude of expected reward corresponding to an action
			Dense(self.actions)
		]
		
		for layer in layers:
			self.clf.add(layer)

		# RMSprop optimizer (Gradient descent with momentum). Mean squared loss is used 
		self.clf.compile(optimizer = "rmsprop", loss="mean_squared_error", metrics=["accuracy"])

	def update_network(self, data, predictor):

		"""
		Updates the weight of the network batch by batch. Predictor is another instance of DQN that is updated once in a while for faster convergence
		data = dict with cur_state, action, next_state, reward and done
		"""

		X = []
		Y = []

		for instance in data:

			# Append the current state to the training matrix 
			X.append(np.asarray(instance["cur_state"]))

			# Calculate expected reward
			expected_reward = predictor.predict(np.asarray([instance["next_state"]])).ravel()
			max_expected_reward = np.max(expected_reward)

			Y.append(np.asarray(list(self.predict(np.asarray([instance["cur_state"]])))[0]))

			if(instance["done"]):
				Y[-1][instance["action"]] = instance["reward"]												# The game has ended. 
			else:
				Y[-1][instance["action"]] = instance["reward"] + self.discount_factor*max_expected_reward	# Bellman update equation

		X = np.asarray(X).squeeze()
		Y = np.asarray(Y).squeeze()

		self.clf.fit(X,Y,batch_size = self.batch_size, nb_epoch=self.EPOCHS)

	def predict(self, state):

		"""
		Given the current state, this function returns the expected reward vector for all possible actions the agent can make. The argmax action is performed
		"""

		state = np.asarray(state)
		return self.clf.predict(state, batch_size=1)	# Batch size is one as there's just one data point

if(__name__=="__main__"):

	pass

