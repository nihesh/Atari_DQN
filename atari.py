# Author	: Nihesh Anderson K
# File 		: atari.py

import gym
import time
from Agent import AtariAgent
import pickle

GAME = "Breakout-v4"
IMG_DIM = (84, 110)
MAX_EPISODES = 1000000000000
EPS_GREEDY = 0.1
EPS_DEC_RATE = 1e-6
REPLAY_MEMORY_SIZE = 100000		# Higher the better
REPLAY_START = 50000
INPUT_SHAPE = (4, 110, 84)
BATCH_SIZE = 32
TRAIN_FREQUENCY = 4				# freq of steps after which training has to be done
UPDATE_FREQUENCY = 10000		# updating predictor	
DISCOUNT_FACTOR = 0.99
MODEL_SAVE_RATE = 100

if(__name__ == "__main__"):

	environment = gym.make(GAME)

	try:
		print("Loading existing agent")
		file = open("./agent.obj", "rb")
		agent = pickle.load(file)
		file.close()
		print("No of items in replay memory: "+str(len(agent.replay_memory)))	# Replay memory size somehow gets reset to 0 here, after restarting
	except:	
		print("Setting up agent")
		agent = AtariAgent(environment.action_space.n, EPS_GREEDY, IMG_DIM, REPLAY_MEMORY_SIZE, INPUT_SHAPE, BATCH_SIZE, UPDATE_FREQUENCY, DISCOUNT_FACTOR, EPS_DEC_RATE, REPLAY_START, TRAIN_FREQUENCY)

	while agent.episode_id < MAX_EPISODES:

		init_state = environment.reset()

		sample_space = environment.action_space

		agent.reset_state(init_state)
		score = 0

		while(True):	# Play an episode

			# Randomly samples an action to perform
			action_to_perform = agent.play(sample_space)
			next_state,reward,done,_ = environment.step(action_to_perform)
			environment.render()
			agent.update(action_to_perform, next_state, reward, done)
			score += min(max(reward,-1),1)

			# Game ended?
			if(done):
				break

		if(agent.episode_id%MODEL_SAVE_RATE == 0):
			file = open("./agent.obj","wb")
			print("Serialising agent")
			pickle.dump(agent, file)
			file.close()
			file = open("./agent.obj","rb")
			agent = pickle.load(file)
			file.close()
			print(len(agent.replay_memory))		# Prints the actual size of replay memory 
		
		agent.finishEpisode(score)

		print("Score obtained in Episode "+str(agent.episode_id)+" = "+str(score))
		print("Max score so far = "+str(agent.max_reward))
		print("Avg score so far = "+str(agent.avg_reward))


	environment.close()