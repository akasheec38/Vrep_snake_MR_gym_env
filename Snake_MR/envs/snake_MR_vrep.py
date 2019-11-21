# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 19:47:15 2019

@author: MOC817
"""
import math
import matplotlib.pyplot as plt
import VREP_com
import vrep
import numpy as np

import gym
from gym import error, spaces, utils
from gym.utils import seeding


clientID = VREP_com.clientID
Handle = VREP_com.Handle

class snake_MR(gym.Env):    
    
    def __init__(self):
        self.min_action = -1.0
        self.max_action = 1.0
        self.Target_vel = 0.50
        self.tau = 0.005  # seconds between state updates
        self.action_space = spaces.Box(low=self.min_action, high=self.max_action,
                                       shape=(8,), dtype=np.float32)
        self.observation_space = spaces.Discrete(26)
        self.seed()
        self.reset()
        self.a1 = 0.2
        self.a2 = 0.2
        self.b1 = 0.6
        self.torque_max = 1.0000e+02
        self.angular_vel_max = 6.28319
        
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def step(self, action):
        done =False
        VREP_com.vrep_moves_robot(clientID, Handle, action)
        vrep.simxSynchronousTrigger(clientID)
        vrep.simxGetPingTime(clientID) 
        
        
        ## observe the state after taking movement
        Observation_array = []
        Observation_array = np.array(Observation_array)
        ### get joint position, angular_vel, head_vel, torque of each joints
        joints_pos = VREP_com.vrep_get_joint_pos(clientID, Handle)
        joints_angular_vel = VREP_com.vrep_get_angular_vel(clientID, Handle)
        joints_torque = VREP_com.vrep_get_torque(clientID, Handle)    
        Observation_array = np.concatenate((joints_pos,joints_angular_vel, joints_torque))    
        err_code ,linearVelocity_cam_each_step, angularVelocity_cam = vrep.simxGetObjectVelocity(clientID,Handle[0, 9],vrep.simx_opmode_blocking)
        if err_code !=0 : raise Exception()
        robot_vel = linearVelocity_cam_each_step[1]
        Observation_array = np.append(Observation_array, round(robot_vel, 4))
        Observation_array = np.append(Observation_array, self.Target_vel)
        
        ## get reward
        power = np.sum(np.abs(joints_torque*joints_angular_vel))
        #nor_power = (np.sum((np.abs(joints_torque*joints_angular_vel)))/(self.torque_max*self.angular_vel_max))/8
        #print('power: ', power)
        #reward = ((1 - (np.abs(self.Target_vel - head_vel))/self.a1)**(1/self.a2))*((np.abs(1-nor_power))**(self.b1**-2))
        #reward = np.round(reward, 5)
        #print('head_velocity',head_vel ) 
        
        err_code, MyRobotPos_Current = vrep.simxGetObjectPosition(clientID, Handle[0, 9], -1, vrep.simx_opmode_blocking)
        MyRobotPos_Current = np.array(MyRobotPos_Current)
        COT_inv   = (5.000e-01 * 9.8 * MyRobotPos_Current[1])/power
        self.state = np.array(Observation_array)
        if robot_vel >= self.Target_vel:
            done = True
        return self.state, COT_inv,  done, {}
    
    def reset(self):
        VREP_com.vrep_sim_check(clientID)
        vrep.simxStartSimulation(clientID, vrep.simx_opmode_oneshot)
        vrep.simxSynchronousTrigger(clientID) 
        VREP_com.vrep_reset_robot_joints(clientID, Handle)
        vrep.simxSynchronousTrigger(clientID) 
        vrep.simxGetPingTime(clientID)
        
        Observation_array = []
        Observation_array = np.array(Observation_array)
        ### get joint position, angular_vel, head_vel, torque of each joints
        joints_pos = VREP_com.vrep_get_joint_pos(clientID, Handle)
        joints_angular_vel = VREP_com.vrep_get_angular_vel(clientID, Handle)
        joints_torque = VREP_com.vrep_get_torque(clientID, Handle)    
        Observation_array = np.concatenate((joints_pos,joints_angular_vel, joints_torque))    
        err_code ,linearVelocity_cam_each_step, angularVelocity_cam = vrep.simxGetObjectVelocity(clientID,Handle[0, 9],vrep.simx_opmode_blocking)
        if err_code !=0 : raise Exception()
        #head_vel = (linearVelocity_cam_each_step[0]**2 + linearVelocity_cam_each_step[1]**2 + linearVelocity_cam_each_step[2]**2)**0.5 
        robot_vel = linearVelocity_cam_each_step[1]  ## to direct in Y direction only 
        Observation_array = np.append(Observation_array, round(robot_vel, 4))
        Observation_array = np.append(Observation_array, self.Target_vel) 
        #vrep.simxStopSimulation(clientID, vrep.simx_opmode_oneshot)
        self.state = Observation_array
        return self.state
    
    def close(self):
        vrep.simxStopSimulation(clientID, vrep.simx_opmode_oneshot)

     
        
        