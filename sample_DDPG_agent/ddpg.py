
# -----------------------------------
# Deep Deterministic Policy Gradient
# Author: Flood Sung
# Date: 2016.5.4
# -----------------------------------

import gym
import tensorflow as tf
import numpy as np
from OU import OU
import math, random
from critic_network import CriticNetwork 
from actor_network import ActorNetwork
from ReplayBuffer import ReplayBuffer
from configurations import save_location


# Hyper Parameters:

REPLAY_BUFFER_SIZE = 100000
REPLAY_START_SIZE = 100
BATCH_SIZE = 32
GAMMA = 0.99


class DDPG:
    """docstring for DDPG"""
    def __init__(self, env_name, state_dim,action_dim):
        self.name = 'DDPG' # name for uploading results
        self.env_name = env_name
        # Randomly initialize actor network and critic network
        # with both their target networks
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Ensure action bound is symmetric
        self.time_step = 0 
        self.sess = tf.InteractiveSession()

        self.actor_network = ActorNetwork(self.sess,self.state_dim,self.action_dim)
        self.critic_network = CriticNetwork(self.sess,self.state_dim,self.action_dim)
        
        # initialize replay buffer
        self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)

        # Initialize a random process the Ornstein-Uhlenbeck process for action exploration
        self.OU = OU()
        
        # loading networks
        self.saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state(save_location)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")

    def train(self):
        #print "train step",self.time_step
        # Sample a random minibatch of N transitions from replay buffer
        minibatch = self.replay_buffer.getBatch(BATCH_SIZE)
        state_batch = np.asarray([data[0] for data in minibatch])
        action_batch = np.asarray([data[1] for data in minibatch])
        reward_batch = np.asarray([data[2] for data in minibatch])
        next_state_batch = np.asarray([data[3] for data in minibatch])
        done_batch = np.asarray([data[4] for data in minibatch])

        # for action_dim = 1
        action_batch = np.resize(action_batch,[BATCH_SIZE,self.action_dim])

        # Calculate y_batch
        
        next_action_batch = self.actor_network.target_actions(next_state_batch)
        q_value_batch = self.critic_network.target_q(next_state_batch,next_action_batch)
        y_batch = []  
        for i in range(len(minibatch)): 
            if done_batch[i]:
                y_batch.append(reward_batch[i])
            else :
                y_batch.append(reward_batch[i] + GAMMA * q_value_batch[i])
        y_batch = np.resize(y_batch,[BATCH_SIZE,1])
        # Update critic by minimizing the loss L
        self.critic_network.train(y_batch,state_batch,action_batch)

        # Update the actor policy using the sampled gradient:
        action_batch_for_gradients = self.actor_network.actions(state_batch)
        q_gradient_batch = self.critic_network.gradients(state_batch,action_batch_for_gradients)

        self.actor_network.train(q_gradient_batch,state_batch)

        # Update the target networks
        self.actor_network.update_target()
        self.critic_network.update_target()
            
    def saveNetwork(self):
        self.saver.save(self.sess, save_location + self.env_name + 'network' + '-ddpg', global_step = self.time_step)


    def action(self,state):
        action = self.actor_network.action(state)
        action[0] = np.clip( action[0], -1 , 1 )
        action[1] = np.clip( action[1], 0 , 1 )
        action[2] = np.clip( action[2], 0 , 1 )
        #print "Action:", action
        return action

    def noise_action(self,state,epsilon):
        # Select action a_t according to the current policy and exploration noise
        action = self.actor_network.action(state)
        #print action.shape
        #print "Action_No_Noise:", action
        noise_t = np.zeros(self.action_dim)
        noise_t[0] = epsilon * self.OU.function(action[0],  0.0 , 0.60, 0.80)
        noise_t[1] = epsilon * self.OU.function(action[1],  0.5 , 1.00, 0.10)
        noise_t[2] = epsilon * self.OU.function(action[2], -0.1 , 1.00, 0.05)
        
        if random.random() <= 0.1:
           # print("********Stochastic brake***********")
           noise_t[2] = epsilon * self.OU.function(action[2],  0.2 , 1.00, 0.10)

        action = action + noise_t
        action[0] = np.clip(action[0], -1, 1)
        action[1] = np.clip(action[1], 0 , 1)
        action[2] = np.clip(action[2], 0 , 1)
        
        #print "Action_Noise:", action
        return action
    
    def perceive(self,state,action,reward,next_state,done):
        # Store transition (s_t,a_t,r_t,s_{t+1}) in replay buffer
        
        if ( not (math.isnan( reward ))):
            self.replay_buffer.add(state,action,reward,next_state,done)
        
        self.time_step =  self.time_step + 1 
        # Store transitions to replay start size then start training
        if self.replay_buffer.count() >  REPLAY_START_SIZE:
            self.train()


