import sys
sys.path.append('./sample_DDPG_agent/') 

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
import os
from configurations import *
from ddpg import *

import gc
gc.enable()

# config parameters are printed in main

def playGame(f_diagnostics, train_indicator, port=3101):    # 1 means Train, 0 means simply Run
	
	action_dim = 3  #Steering/Acceleration/Brake
	state_dim = 65  #of sensors input
	env_name = 'Torcs_Env'
	save_location = "./weights/"+str(port)+"/"
	agent = DDPG(env_name, state_dim, action_dim, save_location)

	# Generate a Torcs environment
	print("I have been asked to use port: ", port)
	env = TorcsEnv(vision=False, throttle=True, gear_change=False) 
	ob = None
	while ob is None:
		try:
			client = snakeoil3.Client(p=port, vision=False)  # Open new UDP in vtorcs
			client.MAX_STEPS = np.inf

			client.get_servers_input(0)  # Get the initial input from torcs

			obs = client.S.d  # Get the current full-observation from torcs
			ob = env.make_observation(obs)

			s_t = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY,  ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm, ob.opponents))
		except:
			pass

	EXPLORE = total_explore
	episode_count = max_eps
	max_steps = max_steps_eps
	epsilon = epsilon_start
	done = False
	epsilon_steady_state = 0.01 # This is used for early stopping.
 
	totalSteps = 0
	best_reward = -100000
	running_avg_reward = 0.


	print("TORCS Experiment Start.")
	for i in range(episode_count):

		save_indicator = 0
		early_stop = 1
		total_reward = 0.
		info = {'termination_cause':0}
		distance_traversed = 0.
		speed_array=[]
		trackPos_array=[]
		
		print('\n\nStarting new episode...\n')

		for step in range(max_steps):

			# Take noisy actions during training
			if (train_indicator):
			    epsilon -= 1.0 / EXPLORE
			    epsilon = max(epsilon, epsilon_steady_state) 
			    a_t = agent.noise_action(s_t,epsilon) #Take noisy actions during training
			else:
			    a_t = agent.action(s_t)	# [steer, accel, brake]
			try:
				ob, r_t, done, info = env.step(step, client, a_t, early_stop)
				if done:
					break
				# print done
				# print 'Action taken'
				analyse_info(info, printing=False)

				s_t1 = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY,  ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm, ob.opponents))
				distance_traversed += ob.speedX*np.cos(ob.angle) #Assuming 1 step = 1 second
				speed_array.append(ob.speedX*np.cos(ob.angle))
				trackPos_array.append(ob.trackPos)

			#Checking for nan rewards: TODO: This was actually below the following block
				if (math.isnan( r_t )):
					r_t = 0.0
					for bad_r in range( 50 ):
						print( 'Bad Reward Found' )
					break #Introduced by Anirban

			# Add to replay buffer only if training
				if (train_indicator):
					agent.perceive(s_t,a_t,r_t,s_t1,done) # Add experience to replay buffer

			except Exception as e:
				print("Exception caught at port " + str(i) + str(e) )
				ob = None
				while ob is None:
					try:
						client = snakeoil3.Client(p=port, vision=False)  # Open new UDP in vtorcs
						client.MAX_STEPS = np.inf
						client.get_servers_input(0)  # Get the initial input from torcs
						obs = client.S.d  # Get the current full-observation from torcs
						ob = env.make_observation(obs)
					except:
						pass  
					continue
			total_reward += r_t
			s_t = s_t1

			# Displaying progress every 15 steps.
			if ( (np.mod(step,15)==0) ):        
			    print("Episode", i, "Step", step, "Epsilon", epsilon , "Action", a_t, "Reward", r_t )

			totalSteps += 1
			if done:
				break


		if ((save_indicator==1) and (train_indicator ==1 )):
                        # Uncomment this for saving only the best model.
			#if (total_reward >= best_reward): 
			#	print("Now we save model with reward " + str(total_reward) + " previous best reward was " + str(best_reward))
			best_reward = total_reward
			if(i%300 == 0):
				agent.saveNetwork(i)     
	
		running_avg_reward = running_average(running_avg_reward, i+1, total_reward)  


		print("TOTAL REWARD @ " + str(i) +"-th Episode  : Num_Steps= " + str(step) + "; Max_steps= " + str(max_steps) +"; Reward= " + str(total_reward) +"; Running average reward= " + str(running_avg_reward))
		print("Total Step: " + str(totalSteps))
		print("")

		print(info)
		try:
			if 'termination_cause' in info.keys() and info['termination_cause']=='hardReset':
				print('Hard reset by some agent')
				ob, client = env.reset(client=client, relaunch=True) 
			else:
				ob, client = env.reset(client=client, relaunch=True) 
		except Exception as e:
			print("Exception caught at point B at port " + str(i) + str(e) )
			ob = None
			while ob is None:
				try:
					client = snakeoil3.Client(p=port, vision=False)  # Open new UDP in vtorcs
					client.MAX_STEPS = np.inf
					client.get_servers_input(0)  # Get the initial input from torcs
					obs = client.S.d  # Get the current full-observation from torcs
					ob = env.make_observation(obs)
				except:
					print("Exception caught at at point C at port " + str(i) + str(e) )

		  
		s_t = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY,  ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm, ob.opponents))

		# document_episode(i, distance_traversed, speed_array, trackPos_array, info, running_avg_reward, f_diagnostics)

	env.end()  # This is for shutting down TORCS
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

	print('is_training : ' + str(is_training))
	print('Starting best_reward : ' + str(start_reward))
	print( total_explore )
	print( max_eps )
	print( max_steps_eps )
	print( epsilon_start )
	print('config_file : ' + str(configFile))

	# f_diagnostics = open('output_logs/diagnostics_for_window_' + sys.argv[1]+'_with_fixed_episode_length', 'w') #Add date and time to file name
	f_diagnostics = ""
	playGame(f_diagnostics, train_indicator=0, port=port)
	# f_diagnostics.close()
