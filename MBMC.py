import time
import numpy as np
from vis_gym import *
import re

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
	2. Keep track of (relevant) gameplay history in an appropriate format for each of the episodes.
	3. From gameplay history, estimate the probability of victory against each of the guards when taking the fight action.

	Some important notes:

		a. For this implementation, do not enforce a fight action upon encountering a guard, or hard-code movement actions for non-guard cells.
		   While, in practice, we often use external knowledge to inform our actions, this assignment is aimed at testing the ability to learn 
		   solely from uniformly random interactions with the environment.

		b. Upon taking the fight action, if the player defeats the guard, the player is moved to a random neighboring cell with 
		   UNCHANGED health. (2 = Full, 1 = Injured, 0 = Critical).

		c. If the player loses the fight, the player is still moved to a random neighboring cell, but the health decreases by 1.

		d. Your player might encounter the same guard in different cells in different episodes.

		e. A failed hide action results in a forced fight action by the environment; however, you do not need to account for this in your 
		   implementation. We make the simplifying assumption that that we did not 'choose' to fight the guard, rather the associated reward
		   or penalty based on the final outcome is simply a consequence of a success/failure for the hide action.

		f. All interaction with the environment must be done using the env.step() method, which returns the next
		   observation, reward, done (Bool indicating whether terminal state reached) and info. This method should be called as 
		   obs, reward, done, info = env.step(action), where action is an integer representing the action to be taken.

		g. The env.reset() method resets the environment to the initial configuration and returns the initial observation. 
		   Do not forget to also update obs with the initial configuration returned by env.reset().

		h. To simplify the representation of the state space, each state may be hashed into a unique integer value using the hash function provided above.
		   For instance, the observation {'player_position': (1, 2), 'player_health': 2, 'guard_in_cell='G4'} 
		   will be hashed to 1*5*3*5 + 2*3*5 + 2*5 + 4 = 119. There are 375 unique states.

		i. To refresh the game screen if using the GUI, use the refresh(obs, reward, done, info) function, with the 'if gui_flag:' condition.
		   Example usage below. This function should be called after every action.

		   if gui_flag:
		       refresh(obs, reward, done, info)  # Update the game screen [GUI only]

	Finally, return the np array, P which contains four float values, each representing the probability of defeating guards 1-4 respectively.

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

def guard_name_to_index(g: str):
	if g == "G1":
		return 0
	elif g == "G2":
		return 1
	elif g == "G3":
		return 2
	elif g == "G4":
		return 3
	else:
		assert False

def do_episode():
	env.reset()
	wins_guards_fights = np.zeros(len(env.guards))
	lost_guard_fights = np.zeros(len(env.guards))
	won_regex = re.compile(r'Fought (\S*) and won!')
	lost_regex = re.compile(r'Fought (\S*) and lost!')

	while(True):
		observation, reward, done, info, = env.step(env.action_space.sample())
		if done:
			break

		won_result = won_regex.match(info["result"])
		lost_result = lost_regex.match(info["result"])

		if won_result is not None:
			# print(won_result.group(1))
			wins_guards_fights[guard_name_to_index(won_result.group(1))] += 1
		if lost_result is not None:
			#print(lost_result.group(1))
			lost_guard_fights[guard_name_to_index(lost_result.group(1))] += 1

	return wins_guards_fights, wins_guards_fights + lost_guard_fights

def estimate_victory_probability(num_episodes=100000):
	"""
    Probability estimator

    Parameters:
    - num_episodes (int): Number of episodes to run.

    Returns:
    - P (numpy array): Empirically estimated probability of defeating guards 1-4.
    """
	P = np.zeros(len(env.guards))

	wins_guards_fights = np.zeros(len(env.guards))
	total_guard_fights = np.zeros(len(env.guards))
	for i in range(num_episodes):
		#print(i)
		episode_wins, episode_total = do_episode()

		wins_guards_fights += episode_wins
		total_guard_fights += episode_total

	#print(f"{wins_guards_fights=}")
	#print(f"{total_guard_fights=}")
	P = wins_guards_fights / total_guard_fights
	#print(f"{P=}")
	'''

	YOUR CODE HERE


	'''

	return P


#print(estimate_victory_probability())
