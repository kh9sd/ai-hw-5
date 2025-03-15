import time
import pickle
import numpy as np
from vis_gym import *

gui_flag = False # Set to True to enable the game state visualization
setup(GUI=gui_flag)
env = game # Gym environment already initialized within vis_gym.py

#env.render() # Uncomment to print game state info

def hash(obs):
	x,y = obs['player_position']
	h = obs['player_health']
	g = obs['guard_in_cell']
	if not g:
		g = 0
	else:
		g = int(g[-1])

	return x*(5*3*5) + y*(3*5) + h*5 + g

'''

Complete the function below to do the following:

	1. Run a specified number of episodes of the game (argument num_episodes). An episode refers to starting in some initial
	   configuration and taking actions until a terminal state is reached.
	2. Instead of saving all gameplay history, maintain and update Q-values for each state-action pair that your agent encounters in a dictionary.
	3. Use the Q-values to select actions in an epsilon-greedy manner. Refer to assignment instructions for a refresher on this.
	4. Update the Q-values using the Q-learning update rule. Refer to assignment instructions for a refresher on this.

	Some important notes:
		
		- The state space is defined by the player's position (x,y), the player's health (h), and the guard in the cell (g).
		
		- To simplify the representation of the state space, each state may be hashed into a unique integer value using the hash function provided above.
		  For instance, the observation {'player_position': (1, 2), 'player_health': 2, 'guard_in_cell='G4'} 
		  will be hashed to 1*5*3*5 + 2*3*5 + 2*5 + 4 = 119. There are 375 unique states.

		- Your Q-table should be a dictionary with the following format:

				- Each key is a number representing the state (hashed using the provided hash() function), and each value should be an np.array
				  of length equal to the number of actions (initialized to all zeros).

				- This will allow you to look up Q(s,a) as Q_table[state][action], as well as directly use efficient numpy operators
				  when considering all actions from a given state, such as np.argmax(Q_table[state]) within your Bellman equation updates.

				- The autograder also assumes this format, so please ensure you format your code accordingly.
  
		  Please do not change this representation of the Q-table.
		
		- The four actions are: 0 (UP), 1 (DOWN), 2 (LEFT), 3 (RIGHT), 4 (FIGHT), 5 (HIDE)

		- Don't forget to reset the environment to the initial configuration after each episode by calling:
		  obs, reward, done, info = env.reset()

		- The value of eta is unique for every (s,a) pair, and should be updated as 1/(1 + number of updates to Q_opt(s,a)).

		- The value of epsilon is initialized to 1. You are free to choose the decay rate.
		  No default value is specified for the decay rate, experiment with different values to find what works.

		- To refresh the game screen if using the GUI, use the refresh(obs, reward, done, info) function, with the 'if gui_flag:' condition.
		  Example usage below. This function should be called after every action.
		  if gui_flag:
		      refresh(obs, reward, done, info)  # Update the game screen [GUI only]

	Finally, return the dictionary containing the Q-values (called Q_table).

'''

"""
self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'FIGHT', 'HIDE']

obs = {
            'player_position': self.current_state['player_position'],
            'player_health': self.health_state_to_int[self.current_state['player_health']],
            'guard_in_cell': guard_in_cell if guard_in_cell else None,
        }

reward: single int
ex: self.rewards = {
            'goal': 10000,
            'combat_win': 10,
            'combat_loss': -1000,
            'defeat': -1000
        }
or could be 0

done: bool

info = {'result': result, 'action': action_name}
	where action_name is one of the 'UP', etc
	where result is some string like
		"Out of bounds!"
		"Guard {guards_in_room[0]} is in the room! You must fight or hide."

	return f"Fought {guard} and won!", self.rewards['combat_win']
	return f"Fought {guard} and lost!", self.rewards['combat_loss']
"""

# TODO: a better way to calc it?
NUMBER_OF_STATES = 375
# TODO: length of env.actions
NUMBER_OF_ACTIONS = 6


def epsilon_greedy(epsilon, Q_s):
	take_random_choice = random.random() < epsilon
	
	if (take_random_choice):
		return env.action_space.sample()
	else:
		return np.argmax(Q_s) 


def do_episode_Q_learning(Q_table: dict[int, np.array], gamma=0.9, epsilon=1):
	initial_state_observation, initial_reward, initial_done, initial_info = env.reset()
	assert(initial_reward == 0)
	assert(initial_done == False)

	state_hash = hash(initial_state_observation)
	updates = np.zeros((NUMBER_OF_STATES, NUMBER_OF_ACTIONS))

	while(True):
		action = epsilon_greedy(epsilon=epsilon, Q_s=Q_table[state_hash])

		state_succ_observation, reward, done, info, = env.step(action)

		# TODO: might need to move break to before loop?
		# or to very end?
		if done:
			break

		state_succ_hash = hash(state_succ_observation)

		learning_rate = 1 / (1 + updates[state_hash, action])

		Q_table[state_hash][action] = (1 - learning_rate) * Q_table[state_hash][action] + learning_rate * (reward + gamma * np.max(Q_table(state_succ_hash)) - Q_table[state_hash][action])

		updates[state_succ_hash, action] += 1
		state_hash = state_succ_hash
		

def Q_learning(num_episodes=10000, gamma=0.9, epsilon=1, decay_rate=0.999):
	"""
	Run Q-learning algorithm for a specified number of episodes.

    Parameters:
    - num_episodes (int): Number of episodes to run.
    - gamma (float): Discount factor.
    - epsilon (float): Exploration rate.
    - decay_rate (float): Rate at which epsilon decays. Epsilon is decayed as epsilon = epsilon * decay_rate after each episode.

    Returns:
    - Q_table (dict): Dictionary containing the Q-values for each state-action pair.
    """
	Q_table = {}

	# initial value for entry not in table is 0
	for _ in num_episodes:
		do_episode_Q_learning(Q_table=Q_table, gamma=gamma, epsilon=epsilon)
		epsilon *= decay_rate

	return Q_table

decay_rate = ''' YOUR DECAY RATE HERE '''

Q_table = Q_learning(num_episodes=1000000, gamma=0.9, epsilon=1, decay_rate=decay_rate) # Run Q-learning

# Save the Q-table dict to a file
with open('Q_table.pickle', 'wb') as handle:
    pickle.dump(Q_table, handle, protocol=pickle.HIGHEST_PROTOCOL)


'''
Uncomment the code below to play an episode using the saved Q-table. Useful for debugging/visualization.

Comment before final submission or autograder may fail.
'''

# Q_table = np.load('Q_table.pickle', allow_pickle=True)

# obs, reward, done, info = env.reset()
# total_reward = 0
# while not done:
# 	state = hash(obs)
# 	action = max(Q_table[state], key=Q_table[state].get)
# 	obs, reward, done, info = env.step(action)
# 	total_reward += reward
# 	if gui_flag:
# 		refresh(obs, reward, done, info)  # Update the game screen [GUI only]

# print("Total reward:", total_reward)

# # Close the
# env.close() # Close the environment


