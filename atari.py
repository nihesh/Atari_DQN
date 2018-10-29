# Author	: Nihesh Anderson K
# File 		: atari.py

import gym
import time
from Agent import AtariAgent

GAME = "Breakout-v4"
IMG_DIM = (84, 110)
MAX_EPISODES = 1000
EPS_GREEDY = 0.1
REPLAY_MEMORY_SIZE = 1000		# Higher the better
INPUT_SHAPE = (4, 110, 84)
BATCH_SIZE = 32

if(__name__ == "__main__"):

	environment = gym.make(GAME)

	episodes = 0

	training_epoch = []
	cumulative_scores = []

	agent = AtariAgent(environment.action_space.n, EPS_GREEDY, IMG_DIM, REPLAY_MEMORY_SIZE, INPUT_SHAPE, BATCH_SIZE)

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