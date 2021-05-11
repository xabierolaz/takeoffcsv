#!/usr/bin/env python
# ROS python API
import rospy
from numpy import inf
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import numpy as np
import csv
from gym import spaces
from torch.distributions import Normal
# 3D point & Stamped Pose msgs
# it is a Laserscan not a point cloude so we may not need the point msg here

# import all mavros messages and services
from collections import namedtuple
import torch 
import torch.nn as nn
from torch.optim import Adam
from torchsummary import summary
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.distributions import MultivariateNormal
import thread
import time
import math
import random


from sensor_msgs.msg import LaserScan
from gazebo_msgs.msg import ModelStates
from scipy.spatial import distance
from geometry_msgs.msg import Pose
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState
from datetime import datetime
from std_srvs.srv import Empty
from std_msgs.msg import Float64MultiArray, Float32

pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)


rospy.wait_for_service("/gazebo/set_model_state")
m = rospy.ServiceProxy("/gazebo/set_model_state",SetModelState)



command = ModelState()
command.model_name = "Kwad"

velPub = rospy.Publisher('/Kwad/joint_motor_controller/command', Float64MultiArray, queue_size=4)
reset_world = rospy.ServiceProxy('/gazebo/reset_world', Empty)
fieldnames = ['reward','motor_er','distance_er','pitch_er','roll_er', 'yaw_er', 'motor values','distance' , 'lidar_value' , 'global_x' , 'global_y' , 'pitch' ,'roll', 'yaw']
fieldnames2 = ['epsodic reward'  ,'closest_dist' ,'max_height' ,'end_distance' ,'max_distance' , 'highest_x' , 'highest_y' , 'highest_pitch' , 'highest_roll' , 'epsodic motor_er' , 'episodic distance_er','episodic pitch_er','episodic roll_er','episodic yaw_er' ]
m1 = 50
m2 = 50
m3 = 50
m4 = 50



global roll, pitch, yaw , lidar_distance ,velocity , global_x , global_y  , episode , total_motor_er, total_distance_er ,total_pitch_er ,total_roll_er , total_yaw_er ,  closest_dist , distance_goal , name_data , end_distance , max_distance , episode_number , change 

change = int(10)
episode_number = 0
velocity = Float64MultiArray()

name =  'TRAININGS/'+ str(datetime.now())
with open(name,'w') as csv_file_2:
    csv_writer_2 = csv.DictWriter(csv_file_2 , fieldnames = fieldnames2)
    csv_writer_2.writeheader()
#reward_file_name ='rewards_new_2'+str(datetime.now())
my_folder = '/home/cuda/Desktop/TRAININGS/EPISODES/' +str(datetime.now()) + '/'
ac_weights_folder = '/home/cuda/Desktop/TRAININGS/AC_WEIGHTS/' + str(datetime.now()) + '/'
cr_weights_folder = '/home/cuda/Desktop/TRAININGS/CR_WEIGHTS/' + str(datetime.now()) + '/'
    
if not os.path.exists(my_folder):
    os.makedirs(my_folder)

if not os.path.exists(ac_weights_folder):
    os.makedirs(ac_weights_folder)

if not os.path.exists(cr_weights_folder):
    os.makedirs(cr_weights_folder)



class Network(nn.Module):
    def __init__(self , in_dim , out_dim):
        super(Network,self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim , 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64,64),
            nn.Dropout(0.5),
            nn.Linear(64,out_dim),
            nn.Tanh()
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.log_std = nn.Parameter(torch.ones(out_dim)*-4)
    
    def forward(self,obs):
        if isinstance(obs ,np.ndarray):
            obs = torch.tensor(obs ,dtype= torch.float).to(self.device)
        
        out = self.model(obs)

        return out
'''
class Network(nn.Module):
    def __init__(self , in_dim , out_dim):
        super(Network,self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim , 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64,128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16,out_dim),
            nn.Tanh()
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.log_std = nn.Parameter(torch.ones(out_dim)*-4)
    
    def forward(self,obs):
        if isinstance(obs ,np.ndarray):
            obs = torch.tensor(obs ,dtype= torch.float).to(self.device)
        
        out = self.model(obs)

        return out

'''


class Environment():
    global m1 , m2 , m3 , m4 ,lidar_distance ,global_y , global_x ,roll, pitch, episode , total_motor_er, total_distance_er , total_pitch_er, total_roll_er, total_yaw_er  , closest_dist , distance_goal , name_data , end_distance , max_distance ,episode_number
    def __init__(self):
        
        observation_high = np.array([np.finfo(np.float32).max,np.finfo(np.float32).max,np.finfo(np.float32).max,np.finfo(np.float32).max,np.finfo(np.float32).max,np.finfo(np.float32).max,np.finfo(np.float32).max,np.finfo(np.float32).max,np.finfo(np.float32).max,np.finfo(np.float32).max,np.finfo(np.float32).max,np.finfo(np.float32).max,np.finfo(np.float32).max,np.finfo(np.float32).max,np.finfo(np.float32).max,np.finfo(np.float32).max,np.finfo(np.float32).max,np.finfo(np.float32).max,np.finfo(np.float32).max,np.finfo(np.float32).max,np.finfo(np.float32).max,np.finfo(np.float32).max,np.finfo(np.float32).max,np.finfo(np.float32).max,np.finfo(np.float32).max,np.finfo(np.float32).max,np.finfo(np.float32).max,np.finfo(np.float32).max,np.finfo(np.float32).max,np.finfo(np.float32).max,np.finfo(np.float32).max,np.finfo(np.float32).max,np.finfo(np.float32).max])
        self.action_space = spaces.Box(np.array([-1.0,-1.0,-1.0,-1.0]),np.array([1.0,1.0,1.0,1.0]), dtype=np.float64)
        self.observation_space = spaces.Box(-observation_high, observation_high)


    def reset(self):
        global m1,m2,m3,m4
        obs = np.array([0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0, 0.0, 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0])
        reset_world()
        velocity.data = [0 , 0 , 0 , 0]
        velPub.publish(velocity)
        rospy.sleep(1)
        m1 = 50
        m2 = 50
        m3 = 50
        m4 = 50

        return obs 


    def step(self,action,state):
        global m1 , m2 , m3 , m4 , global_y , global_x , lidar_distance , yaw , roll , pitch


        motor_0 = 1500 + action[0]

        motor_1 = 1500 + action[1]

        motor_2 = 1500 + action[2]

        motor_3 = 1500 + action[3]
        #Limit esc pulses
        if(motor_0 > 2000): motor_0 = 2000
        if(motor_1 > 2000): motor_1 = 2000
        if(motor_2 > 2000): motor_2 = 2000
        if(motor_3 > 2000): motor_3 = 2000

        if(motor_0 < 1100): motor_0 = 1100
        if(motor_1 < 1100): motor_1 = 1100
        if(motor_2 < 1100): motor_2 = 1100
        if(motor_3 < 1100): motor_3 = 1100
        #Map esc values
        m4 = ((motor_0 -1500)/25) + 50
        m2 = ((motor_1 -1500)/25) + 50
        m3 = ((motor_2 -1500)/25) + 50
        m4 = ((motor_3 -1500)/25) + 50

        velocity.data = [m1,-m2,m3,-m4]
        velPub.publish(velocity)


        reward = self.get_reward(state,action)
        next_state = self.get_state(state)
    
        return next_state , reward 
    
    #REWARDS

    def get_reward(self ,state, action):        
        global m1 , m2 , m3 , m4 ,lidar_distance ,global_y , global_x ,roll, pitch, episode , total_motor_er, total_distance_er ,total_pitch_er, total_roll_er, total_yaw_er , closest_dist , distance_goal , name_data , end_distance , max_distance
        
        #print(m1,m2,m3,m4)
        drone_position = [global_x , global_y , lidar_distance]
        goal_position = [0 , 0 , 4]
        distance_goal = distance.euclidean(goal_position , drone_position)  
        #print(distance_goal , drone_position ) 
        if closest_dist > distance_goal :
            closest_dist = distance_goal

        reward = 0
        distance_er = 0              
        pitch_er = 0
        roll_er = 0
        yaw_er = 0
        motor_er =  0


        end_distance =  distance_goal
        
        if max_distance < distance_goal:
            max_distance = distance_goal

       
        
        distance_er = -distance_goal/4 #4/4=1
        pitch_er = -abs(pitch)/180 #180/180=1
        roll_er = -abs(roll)/180 #180/180=1
        yaw_er = (-abs(abs(yaw)-abs(state[25])))/180 #180/180=1
        
        
       
        '''
        
        if lidar_distance <= 4:
                
            if (lidar_distance - state[24]) > 0.001:
                distance_er = 0.1
                print("UP ")
                if abs(abs(yaw)-abs(state[25])) > 1 and abs(abs(state[25])-abs(state[14])) > 1:
                    print("UP+YAW")
                    distance_er = 0.2
            else:
                distance_er = 0
                print("NO UP")



        elif lidar_distance > 4:
            if m1<state[29] and  m2 < state[30] and  m3 < state[31] and  m4 < state[32]:
            #if abs(lidar_distance - state[24]) < abs((state[24] - state[13])) or lidar_distance < state[24]:    
                print("DECREASE")
                distance_er = 0.2
            else:
                distance_er = -0.2
                print("NO DECREASE")

            
           
        #YAW OPTION A
        #if the yaw has been increasing over abs 1 in two consecutive steps
        if abs(abs(yaw)-abs(state[25])) > 1 and abs(abs(state[25])-abs(state[14])) > 1:
            #print("yaw ERROR")
            yaw_er = -1
        else:
            yaw_er = 0
            #print("yaw OK")
                #print("yaw error")
            
        ''' 
       


        
        total_yaw_er += yaw_er
        total_distance_er  += distance_er
        total_pitch_er  += pitch_er
        total_roll_er  += roll_er
        
        reward = 1 + total_distance_er + total_pitch_er + total_roll_er + total_yaw_er + total_motor_er

        #print (reward)
        with open(name_data + '.csv','a') as csv_file:
            csv_writer = csv.DictWriter(csv_file , fieldnames = fieldnames)
            information = { "reward" : reward , "distance_er" : distance_er , "distance" : distance_goal, "lidar_value" : lidar_distance, "motor values" : [m1,m2,m3,m4], "global_x" : global_x , "global_y": global_y , "pitch" : pitch ,"roll" : roll, "yaw" : yaw }
            csv_writer.writerow(information)
 
        return reward
    
    def get_state(self,state):
        global global_x, global_y, distance_goal, lidar_distance, yaw, pitch, roll
        #obs = np.array([global_x, global_y , lidar_distance , yaw, pitch, roll, distance_goal])
        
        state[0] = state[11]
        state[1] = state[12]
        state[2] = state[13]
        state[3] = state[14]
        state[4] = state[15]
        state[5] = state[16]
        state[6] = state[17]
        state[7] = state[18]
        state[8] = state[19]
        state[9] = state[20]
        state[10] = state[21]

        state[11] = state[22]
        state[12] = state[23]
        state[13] = state[24]
        state[14] = state[25]
        state[15] = state[26]
        state[16] = state[27]
        state[17] = state[28]
        state[18] = state[29]
        state[19] = state[30]
        state[20] = state[31]
        state[21] = state[32]
        
        state[22] = global_x
        state[23] = global_y
        state[24] = lidar_distance
        state[25] = yaw/18
        state[26] = pitch/18
        state[27] = roll/18
        state[28] = distance_goal
        state[29] = m1/10
        state[30] = m2/10
        state[31] = m3/10
        state[32] = m4/10
        
        return state



class PPO():
    def __init__(self):
        self.env = Environment()
        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.shape[0]
        self._init_hyperparameters()
        
        self.actor = Network(self.obs_dim , self.act_dim)
        self.critic = Network(self.obs_dim , 1)
        model_parameters = filter(lambda p: p.requires_grad, self.actor.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        #print(params)
        self.actor_optim = Adam(self.actor.parameters(), lr=2.5e-4 ,weight_decay=0.2)
        self.critic_optim = Adam(self.critic.parameters(), lr=2.5e-4 ,weight_decay=0.2)

        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.002)
        self.cov_mat = torch.diag(self.cov_var).to(self.actor.device)

    def learn(self,total_timesteps):
        global m1 , m2 , m3 , m4 ,lidar_distance ,global_y , global_x ,roll, pitch, episode , total_motor_er, total_distance_er ,total_yaw_er , closest_dist , distance_goal , name_data , end_distance , max_distance
        
        t_till_now = 0
        i_till_now = 0

        while t_till_now < total_timesteps:
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens ,V = self.rollout()
            pause()

            t_till_now += np.sum(batch_lens)
            # Increment the number of iterations
            i_till_now += 1
            

            # Calculate advantage at k-th iteration
            #V, _ = self.evaluate(batch_obs, batch_acts)
                        
            A_k = batch_rtgs - V.detach()                                                                       
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            A_k = A_k.to(self.actor.device)
            A_k = torch.reshape(A_k ,(len(A_k),1))
            # This is the loop where we update our network for some n epochs
            for _ in range(self.n_updates_per_iteration):                                                       
                # Calculate V_phi and pi_theta(a_t | s_t)
                V, curr_log_probs = self.evaluate(batch_obs, batch_acts)
                #print(batch_log_probs,"batch_log_probs")
                #print(curr_log_probs,"curr_log_probs")
                ratios = torch.exp(curr_log_probs - batch_log_probs)
                
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k
                
                actor_loss = (-torch.min(surr1, surr2)).mean()
                critic_loss = nn.MSELoss()(V, batch_rtgs)

                # Calculate gradients and perform backward propagation for actor network
                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optim.step()
                # Calculate gradients and perform backward propagation for critic network
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()
                
                #print(actor_loss , "actor loss" , critic_loss , "critic loss")

            unpause()

            

    def rollout(self):
        global m1 , m2 , m3 , m4 ,lidar_distance ,global_y , global_x ,roll, pitch, episode , total_motor_er, total_distance_er ,total_yaw_er ,total_pitch_er,total_roll_er, closest_dist , distance_goal , name_data , end_distance , max_distance  ,highest_point ,highest_x , highest_y , highest_pitch , highest_roll ,episode_number 
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_rtgs = []
        batch_lens = []

        ep_rews = []
        t = 0

        while t < self.timesteps_per_batch:
            ep_rews = [] # rewards collected per episode
            # Reset the environment. sNote that obs is short for observation. 
            obs = self.env.reset()
            done = False
            #episode_number = episode_count/self.max_timesteps_per_episode
            name_data = my_folder + str(episode_number)
            closest_dist = 100
            end_distance = 0 
            max_distance = 0

            highest_point = 0
            highest_x = 0
            highest_y = 0
            highest_pitch = 0 
            highest_roll = 0
            total_motor_er = 0
            total_yaw_er = 0
            total_pitch_er = 0
            total_roll_er = 0
            total_distance_er = 0
           

            with open(name_data + '.csv','w') as csv_file:
                csv_writer = csv.DictWriter(csv_file , fieldnames = fieldnames)
                csv_writer.writeheader()
            # Run an episode for a maximum of max_timesteps_per_episode timesteps
            for ep_t in range(self.max_timesteps_per_episode):

                t += 1 # Increment timesteps ran this batch so far
                # print(t)
                # Track observations in this batch
                batch_obs.append(obs.copy())

                # Calculate action and make a step in the env. 
                # Note that rew is short for reward.
                action, log_prob = self.get_action(obs)
                obs, rew = self.env.step(action,obs)
                #print(obs)

                if highest_point < lidar_distance :
                    highest_point = lidar_distance
            
                if highest_x < global_x:
                    highest_x = global_x
                
                if highest_y < global_y:
                    highest_y = global_y

                if highest_pitch < pitch:
                    highest_pitch = pitch

                if highest_roll < roll:
                    highest_roll = roll

                

                # Track recent reward, action, and action log probability
                ep_rews.append(rew)  # we are collecting all the rewards in ep_rews

                batch_acts.append(action)
                batch_log_probs.append(log_prob)
                if abs(pitch) >60 or abs(roll) >60 :
                    break
                    
                if abs(global_x) >4 or abs(global_y)> 4 :
                    break

                if (lidar_distance < 1) and (abs(global_x)>2 or abs(global_y)>2):
                    break

            with open(name,'a') as csv_file_2:
                csv_writer_2 = csv.DictWriter(csv_file_2 , fieldnames = fieldnames2)
                information2 = {"epsodic reward" : sum(ep_rews), "closest_dist" : closest_dist, "max_height" : highest_point,"end_distance": end_distance,"max_distance": max_distance, "highest_x" : highest_x, "highest_y" : highest_y , "highest_pitch" : highest_pitch, "highest_roll" : highest_roll,"epsodic motor_er" : total_motor_er, "episodic distance_er" : total_distance_er, "episodic pitch_er": total_pitch_er, "episodic roll_er": total_roll_er, "episodic yaw_er" :total_yaw_er, }
                #'epsodic reward' ,'closest_dist' ,'max_height' ,'end_distance' ,'max_distance' , 'highest_x' , 'highest_y' , 'highest_pitch' , 'highest_roll' , 'epsodic motor_er' , 'episodic distance_er','episodic pitch_er','episodic roll_er','episodic yaw_er'
                csv_writer_2.writerow(information2)
            #PRINT TOTAL EPISODIC REWARD
            #print(total_yaw_er)
            
            print(sum(ep_rews) , "in episode", episode_number, "distance", total_distance_er, "pitch", total_pitch_er, "roll", total_roll_er, "yaw", total_yaw_er ) # sum will add all the rewards
            episode_number = episode_number +1
            batch_lens.append(ep_t + 1)
            batch_rews.append(ep_rews)

        batch_obs = torch.tensor(batch_obs, dtype=torch.float).to(self.actor.device)
        batch_acts = np.array(batch_acts)
        

        batch_acts = torch.tensor(batch_acts, dtype=torch.float).to(self.actor.device)
        V, _ = self.evaluate(batch_obs, batch_acts)
        batch_log_probs = np.array(batch_log_probs)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float).to(self.actor.device)
        batch_rtgs = self.compute_rtgs(batch_rews ,V) 
        #print(sum(batch_acts == 0),"len(batch_acts == 0)")

        if (episode_number%100 == 0):
            torch.save(self.actor.state_dict(), ac_weights_folder + 'ppo_actor'+ str(episode_number)+ '.pth')
            torch.save(self.critic.state_dict(), cr_weights_folder + 'ppo_critic' +str(episode_number)+ '.pth')        
        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens , V

    def compute_rtgs(self, batch_rews , V):
        batch_rtgs = []
        gae = 0
        values_list = list(V)
        values_list.append(torch.tensor(0.0))
        rewards_list = sum(batch_rews ,[])
        
        delta = 0
        # Iterate through each episode
        for i in reversed(range(len(rewards_list))):
            delta = rewards_list[i] + self.gamma * values_list[i + 1]  - values_list[i]
            gae = delta + self.gamma * self.lam* gae
            batch_rtgs.insert(0, gae)

        # Convert the rewards-to-go into a tensor
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float).to(self.actor.device)
        
        return batch_rtgs

    def get_action(self, obs):
        global change
        # Query the actor network for a mean action
        out = self.actor(obs).to(self.actor.device)
        std = self.actor.log_std.exp().expand_as(out).to(self.actor.device)
        #print(std)
        dist = Normal(out, std)
        #dist = dist.to(self.actor.device)
        # Sample an action from the distribution
        action = dist.sample()
        #print(change)
        #print(action)
        # Calculate the log probability for that action
        log_prob = dist.log_prob(action)
        log_prob = log_prob.cpu().detach().numpy()
        #print(obs)
        #print(log_prob)
        # Return the sampled action and the log probability of that action in our distribution
        return action.cpu().detach().numpy(), log_prob

    def evaluate(self, batch_obs, batch_acts):
        V = self.critic(batch_obs).squeeze()

        # Calculate the log probabilities of batch actions using most recent actor network.
        # This segment of code is similar to that in get_action()
        out = self.actor(batch_obs).to(self.actor.device)

        std = self.actor.log_std.exp().expand_as(out).to(self.actor.device)
        dist = Normal(out, std)
        log_probs = dist.log_prob(batch_acts)
        #print(batch_obs)
        #print(log_probs ,"evaluate")
		# Return the value vector V of each observation in the batch
		# and log probabilities log_probs of each action in the batch
        return V, log_probs

    def _init_hyperparameters(self):
        # Initialize default values for hyperparameters
        # Algorithm hyperparameters
        
        self.max_timesteps_per_episode = 2048   # Max number of timesteps per episode
        self.timesteps_per_batch = 3*self.max_timesteps_per_episode          # Number of timesteps to run per batch
        self.n_updates_per_iteration = 5             # Number of times to update actor/critic per iteration BATCH SIZE
        
        self.gamma = 0.99                              # Discount factor to be applied when calculating Rewards-To-Go
        self.lam = 0.95                                 # GAE lambda
        self.clip = 0.2                                 # Recommended 0.2, helps define the threshold to clip the ratio during SGA



def euler_from_quaternion(x, y, z, w):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
     
        return roll_x*57.2958 , pitch_y*57.2958 , yaw_z*57.2958





def service():
    global roll, pitch, yaw , lidar_distance , m1 , m2 , m3 , m4 ,episode , total_motor_er, total_distance_er , total_pitch_er, total_roll_er, total_yaw_er , closest_dist , distance_goal , name_data , end_distance , max_distance ,highest_point ,highest_x , highest_y , highest_pitch , highest_roll 
    
    #print(lidar_distance,"lidar_distance")
    velocity.data = [0 , 0 , 0 , 0]
    velPub.publish(velocity)
    reset_world()
    closest_dist = 100
    end_distance = 0 
    max_distance = 0

    
    
    time.sleep(1)
    
    model = PPO()
    model.learn(100000000)

    


def lasercall_back(msg):
    global roll, pitch, yaw , lidar_distance , m1 ,m2 ,m3 , m4
    #print("step 2") 
    distance = msg.ranges
    lidar_distance = min(distance)
    if (lidar_distance == inf or lidar_distance == -inf) :
        lidar_distance = 0
        


def location_callback(msg):
    global roll, pitch, yaw , lidar_distance , global_x , global_y , global_z
    ind = msg.name.index('Kwad')
    #print("step 3")
    #print(ind)
    orientationObj = msg.pose[ind].orientation
    positionObj = msg.pose[ind].position
    global_x = positionObj.x
    global_y = positionObj.y
    lidar_distance = positionObj.z
    roll, pitch, yaw = euler_from_quaternion(orientationObj.x, orientationObj.y, orientationObj.z, orientationObj.w)

def main():
    rospy.init_node('Drone_control_ppo')
    #rospy.Subscriber("/Kwad/scan" , LaserScan , lasercall_back)
    rospy.Subscriber("/gazebo/model_states" , ModelStates , location_callback)
    rospy.wait_for_service('/gazebo/reset_world')
    #print("kk")
    
    velocity.data = [0 , 0 , 0 , 0]
    velPub.publish(velocity)
    
    thread.start_new_thread(service,())
    #thread.start_new_thread(wind_,())

    rospy.spin()


if __name__ == main():
    main()