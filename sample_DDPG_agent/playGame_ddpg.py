
import numpy as np
np.random.seed(1337)

from gym_torcs import TorcsEnv
import random
import argparse
import tensorflow as tf
from configurations import *

from ddpg import *
import gc
gc.enable()

import timeit
import math

print 'is_training : ' + str(is_training)
print 'Starting best_reward : ' + str(start_reward)
print( total_explore )
print( max_eps )
print( max_steps_eps )
print( epsilon_start )
print 'config_file : ' + str(configFile)


def playGame(train_indicator=is_training):    # 1 means Train, 0 means simply Run

    action_dim = 3  # Steering/Acceleration/Brake
    # state_dim = 29  # Number of sensory inputs
    state_dim = 52  # Number of sensory inputs
    env_name = 'Torcs_Env' + str(state_dim)
    agent = DDPG(env_name, state_dim, action_dim)

    # Generate a TORCS environment
    vision = False
    env = TorcsEnv(vision=vision, throttle=True, gear_change=False)
    
    EXPLORE = total_explore
    episode_count = max_eps
    max_steps = max_steps_eps
    if wait_at_beginning==1:
        max_steps += initial_wait_period        # for initial_wait_period, set all actions to zero.
    epsilon = epsilon_start
    done = False
    
    step = 0
    best_reward = start_reward

    print("TORCS Experiment Start.")
    for i in range(episode_count):
        # Occasional Testing
        # if (( np.mod(i, 10) == 0 ) and (i>20)):
        #     train_indicator= 0
        # else:
        #     train_indicator=is_training

        # visualize the policy every 10 episodes
        if np.mod(i, visualize_after) == 0:
            ob = env.reset(relaunch=True)   
        else:
            ob = env.reset()
            
        # Early episode annealing for out of track driving and small progress
        # During early training phases - out of track and slow driving is allowed as humans do ( Margin of error )
        # As one learns to drive the constraints become stricter
        
        random_number = random.random()
        eps_early = max(epsilon,0.10)
        # if (random_number < (1.0-eps_early)) and (train_indicator == 1):
        #     early_stop = 1
        # else: 
        #     early_stop = 0
        early_stop = 1
        print("Episode : " + str(i) + " Replay Buffer " + str(agent.replay_buffer.count()) + ' Early Stopping: ' + str(early_stop) +  ' Epsilon: ' + str(eps_early) +  ' RN: ' + str(random_number)  )

        # Initializing the first state
        # s_t = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY,  ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))
        s_t = np.hstack((                      
                        ob.angle, 
                        ob.trackPos, 
                        ob.speed, 
                        ob.speedY, 
                        ob.speedZ, 
                        ob.dist_L, 
                        ob.dist_R, 
                        ob.dist_L_rear, 
                        ob.dist_R_rear, 
                        ob.toMarking_L, 
                        ob.toMarking_M, 
                        ob.toMarking_R, 
                        ob.dist_LL, 
                        ob.dist_MM, 
                        ob.dist_RR, 
                        ob.dist_LL_rear, 
                        ob.dist_MM_rear, 
                        ob.dist_RR_rear, 
                        ob.toMarking_LL, 
                        ob.toMarking_ML, 
                        ob.toMarking_MR, 
                        ob.toMarking_RR, 
                        ob.distToVisibleTurn, 
                        ob.visibleTurnRadius, 
                        ob.visibleTypeRoad, 
                        ob.currRadius, 
                        ob.typeRoad, 
                        ob.track,                  # 19
                        ob.rpm,
                        ob.damage, 
                        ob.wheelSpinVel            # 4
            ))                                     # 52 in total

        # For counting the total reward and total steps in the current episode
        total_reward = 0.
        step_eps = 0.
        
        restart = False                 # to penalize going offTrack
        for j in range(max_steps):
            
            # Take noisy actions during training
            if (train_indicator):
                epsilon -= 1.0 / EXPLORE
                epsilon = max(epsilon, 0.1)             # doesn't let epsilon get driven to 0
                a_t = agent.noise_action(s_t,epsilon)
            else:
                a_t = agent.action(s_t)
                
            if wait_at_beginning==1 and j < initial_wait_period:
                a_t = np.zeros(3)

            # ob, r_t, done, info = env.step(a_t[0],early_stop)
            # ob, r_t, done, info = env.step(a_t,early_stop)
            ob, r_t, done, info = env.step(a_t,early_stop)


            # s_t1 = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))
            s_t1 = np.hstack((                      
                        ob.angle, 
                        ob.trackPos, 
                        ob.speed, 
                        ob.speedY, 
                        ob.speedZ, 
                        ob.dist_L, 
                        ob.dist_R, 
                        ob.dist_L_rear, 
                        ob.dist_R_rear, 
                        ob.toMarking_L, 
                        ob.toMarking_M, 
                        ob.toMarking_R, 
                        ob.dist_LL, 
                        ob.dist_MM, 
                        ob.dist_RR, 
                        ob.dist_LL_rear, 
                        ob.dist_MM_rear, 
                        ob.dist_RR_rear, 
                        ob.toMarking_LL, 
                        ob.toMarking_ML, 
                        ob.toMarking_MR, 
                        ob.toMarking_RR, 
                        ob.distToVisibleTurn, 
                        ob.visibleTurnRadius, 
                        ob.visibleTypeRoad, 
                        ob.currRadius, 
                        ob.typeRoad, 
                        ob.track,                  # 19
                        ob.rpm,
                        ob.damage, 
                        ob.wheelSpinVel            # 4
            ))                                      # 52 in total

            # Add to replay buffer and train the network(s)
            if wait_at_beginning==0 or (wait_at_beginning==1 and j >= initial_wait_period):
                if (train_indicator):
                    agent.perceive(s_t,a_t,r_t,s_t1,done)
                
            # Checking for NaN rewards
            if ( math.isnan( r_t )):
                r_t = 0.0
                for bad_r in range( 50 ):
                    print( 'Bad Reward Found' )

            total_reward += r_t
            s_t = s_t1

            # Displaying progress every 15 steps.
            if ( (np.mod(step,15)==0) or r_t<=-5000):        
                print("Episode", i, "Step", step_eps,"Epsilon", epsilon , "Action", a_t, "Reward", r_t )

            step += 1
            step_eps += 1
            if done:
                break
                
        # Saving the best model.
        if total_reward >= best_reward :
            if (train_indicator==1):
                print("Now we save model with reward " + str( total_reward) + " previous best reward was " + str(best_reward))
                best_reward = total_reward
                agent.saveNetwork()       
                
        print("TOTAL REWARD @ " + str(i) +"-th Episode  : Reward " + str(total_reward))
        print("Total Step: " + str(step))
        print("")

    env.end()  # This is for shutting down TORCS
    print("Finish.")

if __name__ == "__main__":
    playGame()

