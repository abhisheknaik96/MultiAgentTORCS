import numpy as np
np.random.seed(1337)

from gym_torcs import TorcsEnv
import snakeoil3_gym as snakeoil3

import collections as col
import random
import argparse
import tensorflow as tf
import timeit
import math
import sys

import gc
gc.enable()

max_eps = 500
max_steps_eps = 3000
epsilon_start = 0.9


def playGame(f_diagnostics, train_indicator, port=3101):    #1 means Train, 0 means simply Run
	
	action_dim = 3  #Steering/Acceleration/Brake
	state_dim = 29  #Number of sensors input
	env_name = 'Torcs_Env'

	# Generate a Torcs environment
	print("I have been asked to use port: ", port)
	env = TorcsEnv(vision=False, throttle=True, gear_change=False) 
	
	client = snakeoil3.Client(p=port, vision=False)  # Open new UDP in vtorcs
	client.MAX_STEPS = np.inf

	client.get_servers_input(0)  # Get the initial input from torcs

	obs = client.S.d  # Get the current full-observation from torcs
	ob = env.make_observation(obs)

	# EXPLORE = total_explore
	episode_count = max_eps
	max_steps = max_steps_eps
	epsilon = epsilon_start
	done = False
	# epsilon_steady_state = 0.01 # This is used for early stopping.
 
	totalSteps = 0
	best_reward = -100000
	running_avg_reward = 0.

	print("TORCS Experiment Start.")
	for i in range(episode_count):

		save_indicator = 0 # 1 to save the learned weights, 0 otherwise
		early_stop = 1
		total_reward = 0.
		info = {'termination_cause':0}
		distance_traversed = 0.
		speed_array=[]
		trackPos_array=[]
		
		print('\n\nStarting new episode...\n')

		for step in range(max_steps):
			#Hard-coded steer=0, accel=1 and brake=0, define a_t as per any other algorithm
			a_t = np.asarray([0.0, 1.0, 0.0])		# [steer, accel, brake]

			ob, r_t, done, info = env.step(step, client, a_t, early_stop)
			if done:
				break
			analyse_info(info, printing=False)

			s_t1 = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY,  ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))
			distance_traversed += ob.speedX*np.cos(ob.angle) #Assuming 1 step = 1 second
			speed_array.append(ob.speedX*np.cos(ob.angle))
			trackPos_array.append(ob.trackPos)

			#Checking for nan rewards: TODO: This was actually below the following block
			if (math.isnan( r_t )):
				r_t = 0.0
				for bad_r in range( 50 ):
					print("Bad Reward Found")
				break #Introduced by Anirban

			total_reward += r_t
			s_t = s_t1

			# Displaying progress every 15 steps.
			if ( (np.mod(step,15)==0) ):        
			    print("Episode", i, "Step", step, "Epsilon", epsilon , "Action", a_t, "Reward", r_t )

			totalSteps += 1
			if done:
				break

		# Saving the best model.
		if ((save_indicator==1) and (train_indicator ==1 )):
			if (total_reward >= best_reward):
				print("Now we save model with reward " + str(total_reward) + " previous best reward was " + str(best_reward))
				best_reward = total_reward
				agent.saveNetwork()     
	
		running_avg_reward = running_average(running_avg_reward, i+1, total_reward)  


		print("TOTAL REWARD @ " + str(i) +"-th Episode  : Num_Steps= " + str(step) + "; Max_steps= " + str(max_steps) +"; Reward= " + str(total_reward) +"; Running average reward= " + str(running_avg_reward))
		print("Total Step: " + str(totalSteps))
		print("")

		print(info)
		if 'termination_cause' in info.keys() and info['termination_cause']=='hardReset':
			print('\n\n***Hard reset by some agent***\n\n')
			ob, client = env.reset(client=client) 
		else:
			ob, client = env.reset(client=client, relaunch=True) 

		s_t = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY,  ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))

		##uncomment this to get some statistics per episode like total distance traversed, average speed, distance from center of track, etc
		# document_episode(i, distance_traversed, speed_array, trackPos_array, info, running_avg_reward, f_diagnostics)

	env.end()  # Shut down TORCS
	print("Finish.")

def document_episode(episode_no, distance_traversed, speed_array, trackPos_array, info, running_avg_reward, f_diagnostics):
	"""
	Note down a tuple of diagnostic values for each episode
	(episode_no, distance_traversed, mean(speed_array), std(speed_array), mean(trackPos_array), std(trackPos_array), info[termination_cause], running_avg_reward)
	"""
	f_diagnostics.write(str(episode_no)+",")
	f_diagnostics.write(str(distance_traversed)+",")
	f_diagnostics.write(str(np.mean(speed_array))+",")
	f_diagnostics.write(str(np.std(speed_array))+",")
	f_diagnostics.write(str(np.mean(trackPos_array))+",")
	f_diagnostics.write(str(np.std(trackPos_array))+",")
	f_diagnostics.write(str(info['termination_cause'])+",")
	f_diagnostics.write(str(running_avg_reward)+"\n")


def running_average(prev_avg, num_episodes, new_val):
	total = prev_avg*(num_episodes-1) 
	total += new_val
	return np.float(total/num_episodes)

def analyse_info(info, printing=True):
	simulation_state = ['Normal', 'Terminated as car is OUT OF TRACK', 'Terminated as car has SMALL PROGRESS', 'Terminated as car has TURNED BACKWARDS']
	if printing and info['termination_cause']!=0:
		print(simulation_state[info['termination_cause']])

if __name__ == "__main__":
	
	try:
		port = int(sys.argv[1])
	except Exception as e:
		# raise e
		print("Usage : python %s <port>" % (sys.argv[0]))
		sys.exit()

	# f_diagnostics = open('output_logs/diagnostics', 'w') #Add date and time to file name
	f_diagnostics = ""
	playGame(f_diagnostics, train_indicator=1, port=port)
	# f_diagnostics.close()
