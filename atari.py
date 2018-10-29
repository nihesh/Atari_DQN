# Author	: Nihesh Anderson K
# File 		: atari.py

import gym
import time
from Agent import AtariAgent

GAME = "Breakout-v4"
IMG_DIM = (84, 110)
MAX_EPISODES = 1000000
EPS_GREEDY = 0.05
REPLAY_MEMORY_SIZE = 10000		# Higher the better
INPUT_SHAPE = (4, 110, 84)
BATCH_SIZE = 32
UPDATE_FREQUENCY = 32			# Must be greater than BATCH_SIZE
MIN_EXP_TO_UPDATE = 50			# atmost REPLAY_MEMORY_SIZE
DISCOUNT_FACTOR = 0.99

if(__name__ == "__main__"):

	environment = gym.make(GAME)

	episodes = 0

	training_epoch = []
	cumulative_scores = []

	MIN_EXP_TO_UPDATE = min(MIN_EXP_TO_UPDATE, REPLAY_MEMORY_SIZE-1)
	agent = AtariAgent(environment.action_space.n, EPS_GREEDY, IMG_DIM, REPLAY_MEMORY_SIZE, INPUT_SHAPE, BATCH_SIZE, UPDATE_FREQUENCY, MIN_EXP_TO_UPDATE, DISCOUNT_FACTOR)

	while episodes < MAX_EPISODES:

		init_state = environment.reset()

		agent.reset_state(init_state)
		score = 0

		while(True):	# Play an episode

			# Randomly samples an action to perform
			action_to_perform = environment.action_space.sample()
			next_state,reward,done,_ = environment.step(action_to_perform)
			environment.render()
			agent.update(action_to_perform, next_state, reward, done)
			score += reward

			# Game ended?
			if(done):
				break

		episodes+=1
		training_epoch.append(episodes)
		cumulative_scores.append(score)

		print("Score obtained in Episode "+str(episodes)+" = "+str(score))

	environment.close()