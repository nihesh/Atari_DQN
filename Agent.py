# Author	: Nihesh Anderson K
# File 		: Agent.py

from DQN import DQN
import numpy as np
import random
from copy import deepcopy
from PIL import Image

class AtariAgent:

	eps_greedy = -1
	state = []
	frame_dimension = ()
	replay_memory_size = -1
	replay_memory = []
	Q_Network = None
	stable_predictor = None
	action = -1
	input_shape = ()
	batch_size = -1
	update_frequency = -1
	cur_update = 1
	min_reqd_exp = -1
	discount_factor = -1

	def __init__(self, action_space_size, eps_greedy, frame_dimension, replay_memory_size, input_shape, batch_size, update_frequency, min_reqd_exp, discount_factor):

		"""
		Initialises the agent
		"""

		self.discount_factor = discount_factor
		self.min_reqd_exp = min_reqd_exp
		self.update_frequency = update_frequency
		self.batch_size = batch_size
		self.input_shape = input_shape
		self.action = action_space_size
		self.eps_greedy = eps_greedy
		self.frame_dimension = frame_dimension
		self.replay_memory_size = replay_memory_size
		self.Q_Network = DQN(self.action, self.input_shape, self.batch_size, self.discount_factor)
		self.stable_predictor = DQN(self.action, self.input_shape, self.batch_size, self.discount_factor)

		# Copy the weights of Q network into stable predictor. Stable predictor is used so that the Q function converges faster
		self.stable_predictor.clf.set_weights(self.Q_Network.clf.get_weights())

	def process_frame(self, frame):

		"""
		Rescales the frame to self.frame_dimension
		"""

		frame_img = Image.fromarray(frame, 'RGB').convert('L').resize(self.frame_dimension)

		return np.asarray(frame_img, dtype=np.uint8).reshape(frame_img.size[1], frame_img.size[0])

	def reset_state(self, init_state):
		
		"""
		Sets the 4 dimensional state matrix to [init_state]x4
		"""

		cur_state = np.asarray([self.process_frame(init_state)])
		self.state = deepcopy(cur_state)
		for i in range(3):
			self.state = np.append(self.state, deepcopy(cur_state), axis = 0)


	def update(self, action, next_state, reward, done):

		"""
		Updates the 4 dimensional state matrix with the next state
		"""

		# bounding reward to [-1,1]
		reward = min(reward, 1)
		reward = max(reward,-1)

		cur_state = deepcopy(self.state)
		self.state = np.append(self.state[1:], [self.process_frame(next_state)], axis=0)

		if(len(self.replay_memory) >= self.replay_memory_size):
			self.replay_memory.pop(0)

		event = {
			"cur_state"	 : cur_state,
			"action"	 : action,
			"next_state" : deepcopy(self.state),
			"reward"	 : reward,
			"done"		 : done
		}

		self.replay_memory.append(event)

		if(self.cur_update == 0):

			self.Q_Network.update_network(random.sample(self.replay_memory, self.batch_size), self.stable_predictor)
			self.stable_predictor.clf.set_weights(self.Q_Network.clf.get_weights())

		self.cur_update = (self.cur_update+1)%self.update_frequency

	def play(self, sample_space):

		rndfloat = random.uniform(0,1)
		if(rndfloat <= self.eps_greedy):
			return sample_space.sample()

		expected_rewards = self.stable_predictor.predict(np.asarray([self.state]))[0]

		action = -1
		reward = -1

		for i in range(len(expected_rewards)):
			if(expected_rewards[i] > reward):
				reward = expected_rewards[i]
				action = i

		return action

if(__name__ == "__main__"):

	pass
