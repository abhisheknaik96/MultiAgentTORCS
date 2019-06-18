import gym
from gym import spaces
import snakeoil3_gym as snakeoil3
import numpy as np
import numpy as np
import copy
import collections as col
import os
import time
import random

class TorcsEnv:
    terminal_judge_start = 100      # If after 100 timestep still no progress, terminated
    termination_limit_progress = 1  # [km/h], episode terminates if car is running slower than this limit
    default_speed = 50

    initial_reset = False

    obs_dim = 65
    act_dim = 3

    def __init__(self, vision=False, throttle=False, gear_change=False):
        self.vision = vision
        self.throttle = throttle
        self.gear_change = gear_change

        self.initial_run = True
        self.time_step = 0

        self.currState = None 

        # Now the action_space and observation_space are actually being used, just like in OpenAI's gym 
        if throttle is False:                           # Throttle is generally True
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,))
        else:
            high = np.array([1., 1., 1.])
            low = np.array([-1., 0., 0.])
            self.action_space = spaces.Box(low=low, high=high)              # steer, accel, brake (according to agent_to_torcs() (check the function definition))

        if vision is False:                             # Vision is True when you need the images from the simulator 
            high = np.inf*np.ones(self.obs_dim)
            low = -high
            self.observation_space = spaces.Box(low, high)
        	# just like in https://github.com/openai/gym/blob/master/gym/envs/mujoco/mujoco_env.py#L50 (as of 30/5/17)
        else:
            high = np.array([1., np.inf, np.inf, np.inf, 1., np.inf, 1., np.inf, 255], dtype=theano.config.floatX)
            low = np.array([0., -np.inf, -np.inf, -np.inf, 0., -np.inf, 0., -np.inf, 0], dtype=theano.config.floatX)
            self.observation_space = spaces.Box(low=low, high=high)			

    def terminate(self):
        episode_terminate = True
        client.R.d['meta'] = True
        print('Terminating because bad episode')


    def step(self, step, client, u, early_stop):
        # client = self.client

        this_action = self.agent_to_torcs(u)

        # Apply Action
        action_torcs = client.R.d
        # Steering
        action_torcs['steer'] = this_action['steer']  # in [-1, 1]

        #  Simple Automatic Throttle Control by Snakeoil
        if self.throttle is False:
            target_speed = self.default_speed
            if client.S.d['speedX'] < target_speed - (client.R.d['steer']*50):
                client.R.d['accel'] += .01
            else:
                client.R.d['accel'] -= .01

            if client.R.d['accel'] > 0.2:
                client.R.d['accel'] = 0.2

            if client.S.d['speedX'] < 10:
                client.R.d['accel'] += 1/(client.S.d['speedX']+.1)

            # Traction Control System
            if ((client.S.d['wheelSpinVel'][2]+client.S.d['wheelSpinVel'][3]) -
               (client.S.d['wheelSpinVel'][0]+client.S.d['wheelSpinVel'][1]) > 5):
                action_torcs['accel'] -= .2
        else:
            action_torcs['accel'] = this_action['accel']
            action_torcs['brake'] = this_action['brake']

        #  Automatic Gear Change by Snakeoil
        if self.gear_change is True:
            action_torcs['gear'] = this_action['gear']
        else:
            #  Automatic Gear Change by Snakeoil is possible
            action_torcs['gear'] = 1
            if self.throttle:
                if client.S.d['speedX'] > 50:
                    action_torcs['gear'] = 2
                if client.S.d['speedX'] > 80:
                    action_torcs['gear'] = 3
                if client.S.d['speedX'] > 110:
                    action_torcs['gear'] = 4
                if client.S.d['speedX'] > 140:
                    action_torcs['gear'] = 5
                if client.S.d['speedX'] > 170:
                    action_torcs['gear'] = 6
        # Save the previous full-obs from torcs for the reward calculation
        obs_pre = copy.deepcopy(client.S.d)

        # One-Step Dynamics Update #################################
        # Apply the Agent's action into torcs
        client.respond_to_server()
        # Get the response of TORCS
        code = client.get_servers_input(step)

        if code==-1:
            client.R.d['meta'] = True
            print('Terminating because server stopped responding')
            return None, 0, client.R.d['meta'], {'termination_cause':'hardReset'}

        # Get the current full-observation from torcs
        obs = client.S.d

        # Make an obsevation from a raw observation vector from TORCS
        self.observation = self.make_observation(obs)
        self.currState = np.hstack((self.observation.angle, self.observation.track, self.observation.trackPos, 
                                    self.observation.speedX, self.observation.speedY,  self.observation.speedZ, 
                                    self.observation.wheelSpinVel/100.0, self.observation.rpm))

        # Reward setting Here #######################################
        # direction-dependent positive reward
        track = np.array(obs['track'])
        trackPos = np.array(obs['trackPos'])
        sp = np.array(obs['speedX'])
        damage = np.array(obs['damage'])
        rpm = np.array(obs['rpm'])

        progress = sp*np.cos(obs['angle']) - np.abs(sp*np.sin(obs['angle'])) - sp * np.abs(obs['trackPos'])
        reward = progress

        # collision detection
        if obs['damage'] - obs_pre['damage'] > 0:
            reward = -1

        # Termination judgement #########################
        episode_terminate = False
        if ( (abs(track.any()) > 1 or abs(trackPos) > 1) and early_stop ):  # Episode is terminated if the car is out of track
            reward = -200
            episode_terminate = True
            client.R.d['meta'] = True
            print('Terminating because Out of Track')

        # if self.terminal_judge_start < self.time_step: # Episode terminates if the progress of agent is small
        #     if ( (progress < self.termination_limit_progress) and early_stop ):
        #         print("No progress")
        #         #episode_terminate = True
        #         client.R.d['meta'] = True
        #         #print('Terminating because Small Progress')

        if np.cos(obs['angle']) < 0: # Episode is terminated if the agent runs backward
            episode_terminate = True
            client.R.d['meta'] = True
            print('Terminating because Turned Back')


        if client.R.d['meta'] is True: # Send a reset signal
            self.initial_run = False
            client.respond_to_server()

        self.time_step += 1

        return self.observation, reward, client.R.d['meta'], {}
        # return reward

    def reset(self, client, relaunch=False):
        #print("Reset")

        port = client.port
        self.time_step = 0
        # print '111'
        if self.initial_reset is not True:
            client.R.d['meta'] = True
            client.respond_to_server()

            ## TENTATIVE. Restarting TORCS every episode suffers the memory leak bug!
            if relaunch is True:
                self.reset_torcs()
                print("### TORCS is RELAUNCHED ###")

        # Modify here if you use multiple tracks in the environment
        client = snakeoil3.Client(p=port, vision=self.vision)  # Open new UDP in vtorcs
        client.MAX_STEPS = np.inf

        # client = self.client
        client.get_servers_input(-1)  # Get the initial input from torcs

        obs = client.S.d  # Get the current full-observation from torcs
        self.observation = self.make_observation(obs)
        self.currState = np.hstack((self.observation.angle, self.observation.track, self.observation.trackPos, 
                                    self.observation.speedX, self.observation.speedY,  self.observation.speedZ, 
                                    self.observation.wheelSpinVel/100.0, self.observation.rpm))

        self.last_u = None

        self.initial_reset = False
        return self.get_obs(), client

    def end(self):
        os.system('pkill torcs')

    def get_obs(self):
        return self.observation

    def reset_torcs(self):
        print("relaunch torcs")
        os.system('pkill torcs')

    def agent_to_torcs(self, u):
        torcs_action = {'steer': u[0]}

        if self.throttle is True:  # throttle action is enabled             # This is generally true
            torcs_action.update({'accel': u[1]})
            torcs_action.update({'brake': u[2]})

        if self.gear_change is True: # gear change action is enabled        # This is generally false
            torcs_action.update({'gear': int(u[3])})

        return torcs_action


    def obs_vision_to_image_rgb(self, obs_image_vec):
        image_vec =  obs_image_vec
        r = image_vec[0:len(image_vec):3]
        g = image_vec[1:len(image_vec):3]
        b = image_vec[2:len(image_vec):3]

        sz = (64, 64)
        r = np.array(r).reshape(sz)
        g = np.array(g).reshape(sz)
        b = np.array(b).reshape(sz)
        return np.array([r, g, b], dtype=np.uint8)

    def make_observation(self, raw_obs):
        if self.vision is False:
            names = ['focus',
                     'speedX', 'speedY', 'speedZ', 'angle', 'damage',
                     'opponents',
                     'rpm',
                     'track', 
                     'trackPos',
                     'wheelSpinVel']
            Observation = col.namedtuple('Observaion', names)
            return Observation(focus=np.array(raw_obs['focus'], dtype=np.float32)/200.,
                               speedX=np.array(raw_obs['speedX'], dtype=np.float32)/300.0,
                               speedY=np.array(raw_obs['speedY'], dtype=np.float32)/300.0,
                               speedZ=np.array(raw_obs['speedZ'], dtype=np.float32)/300.0,
                               angle=np.array(raw_obs['angle'], dtype=np.float32)/3.1416,
                               damage=np.array(raw_obs['damage'], dtype=np.float32),
                               opponents=np.array(raw_obs['opponents'], dtype=np.float32)/200.,
                               rpm=np.array(raw_obs['rpm'], dtype=np.float32)/10000,
                               track=np.array(raw_obs['track'], dtype=np.float32)/200.,
                               trackPos=np.array(raw_obs['trackPos'], dtype=np.float32)/1.,
                               wheelSpinVel=np.array(raw_obs['wheelSpinVel'], dtype=np.float32))
        else:
            names = ['focus',
                     'speedX', 'speedY', 'speedZ', 'angle',
                     'opponents',
                     'rpm',
                     'track',
                     'trackPos',
                     'wheelSpinVel',
                     'img']
            Observation = col.namedtuple('Observaion', names)

            # Get RGB from observation
            image_rgb = self.obs_vision_to_image_rgb(raw_obs[names[8]])

            return Observation(focus=np.array(raw_obs['focus'], dtype=np.float32)/200.,
                               speedX=np.array(raw_obs['speedX'], dtype=np.float32)/self.default_speed,
                               speedY=np.array(raw_obs['speedY'], dtype=np.float32)/self.default_speed,
                               speedZ=np.array(raw_obs['speedZ'], dtype=np.float32)/self.default_speed,
                               opponents=np.array(raw_obs['opponents'], dtype=np.float32)/200.,
                               rpm=np.array(raw_obs['rpm'], dtype=np.float32),
                               track=np.array(raw_obs['track'], dtype=np.float32)/200.,
                               trackPos=np.array(raw_obs['trackPos'], dtype=np.float32)/1.,
                               wheelSpinVel=np.array(raw_obs['wheelSpinVel'], dtype=np.float32),
                               img=image_rgb)
