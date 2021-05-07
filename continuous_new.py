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
print("step 1")

velPub = rospy.Publisher('/Kwad/joint_motor_controller/command', Float64MultiArray, queue_size=4)
reset_world = rospy.ServiceProxy('/gazebo/reset_world', Empty)
fieldnames = ['reward','reward_m','reward_d','reward_x','reward_y','reward_pr' , 'motor values','distance' , 'lidar_value' , 'global_x' , 'global_y' , 'pitch' ,'roll', 'yaw']
fieldnames2 = ['epsodic reward' , 'total_positive_reward' , 'total_negative_reward' ,'closest_dist' ,'max_height' ,'end_distance' ,'max_distance' , 'up_actions(100)' , 'down_actions(0)', 'highest_x' , 'highest_y' , 'highest_pitch' , 'highest_roll' , 'epsodic reward_m' , 'episodic reward_d','episodic reward_xy' ,'episodic reward_pr' ]
m1 = 0
m2 = 0
m3 = 0
m4 = 0


#print("step define global")
global roll, pitch, yaw , lidar_distance ,velocity , global_x , global_y  , episode , total_reward_m, total_reward_d ,total_reward_xy , total_reward_pr , closest_dist , distance_goal , name_data , total_positive_reward , total_negative_reward  , end_distance , max_distance , episode_count , change 

change = int(10)
episode_count = 0


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
            nn.Linear(in_dim , 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32,32),
            nn.Dropout(0.5),
            nn.Linear(32,32),
            nn.Dropout(0.5),
            nn.Linear(32,out_dim),
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




class Environment():
    global m1 , m2 , m3 , m4 ,lidar_distance ,global_y , global_x ,roll, pitch, episode , total_reward_m, total_reward_d ,total_reward_xy , total_reward_pr , closest_dist , distance_goal , name_data , total_positive_reward , total_negative_reward , end_distance , max_distance
    def __init__(self):
        
        observation_high = np.array([np.finfo(np.float32).max,np.finfo(np.float32).max,np.finfo(np.float32).max,np.finfo(np.float32).max,np.finfo(np.float32).max,np.finfo(np.float32).max,np.finfo(np.float32).max,np.finfo(np.float32).max,np.finfo(np.float32).max,np.finfo(np.float32).max,np.finfo(np.float32).max,np.finfo(np.float32).max,np.finfo(np.float32).max,np.finfo(np.float32).max])
        self.action_space = spaces.Box(np.array([-1.0,-1.0,-1.0,-1.0]),np.array([1.0,1.0,1.0,1.0]), dtype=np.float64)
        self.observation_space = spaces.Box(-observation_high, observation_high)


    def reset(self):
        global m1,m2,m3,m4
        obs = np.array([0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0])
        #print(obs.shape ,"obs.shape()")
        reset_world()
        velocity.data = [0 , 0 , 0 , 0]
        velPub.publish(velocity)
        rospy.sleep(1)
        m1 = 0
        m2 = 0
        m3 = 0
        m4 = 0

        return obs 


    def step(self,action,state):
        global m1 , m2 , m3 , m4 , global_y , global_x , lidar_distance , yaw , roll , pitch
        
        
    
        #CONVERT TO REWARD
        #print(abs(action[0]) ,-abs(action[1]) , abs(action[2]), -abs(action[3]))
        #print(action)

        #print(action[0]*change,action[1]*change,action[2]*change,action[3]*change)
        #action = action*change
        m1 = m1 + action[0]*change
        m2 = m2 + action[1]*change
        m3 = m3 + action[2]*change
        m4 = m4 + action[3]*change

        if m1 > 100:
            m1 = 100
        if m1 < 0:
            m1 = 0
        if m2 > 100 :
            m2 = 100
        if m2 < 0:
            m2 = 0
        if m3 > 100 :
            m3 = 100
        if m3 < 0:
            m3 = 0
        if m4 > 100 :
            m4 = 100
        if m4 < 0:
            m4 = 0
        #print(m1,m2,m3,m4)
        #print(action)
        velocity.data = [abs(m1) ,-abs(m2) , abs(m3), -abs(m4)]
        #velocity.data = [0,0 , 0, 0]
        velPub.publish(velocity)


        reward = self.get_reward(state,action)
        next_state = self.get_state(state)
    
        return next_state , reward 
    
    #REWARDS

    def get_reward(self ,state, action):        
        global m1 , m2 , m3 , m4 ,lidar_distance ,global_y , global_x ,roll, pitch, episode , total_reward_m, total_reward_d ,total_reward_xy , total_reward_pr , closest_dist , distance_goal , name_data , total_positive_reward , total_negative_reward , end_distance , max_distance
        
        #print(m1,m2,m3,m4)
        #print(global_x)
        drone_position = [global_x , global_y , lidar_distance]
        goal_position = [0 , 0 , 4]
        distance_goal = distance.euclidean(goal_position , drone_position)  
        #print(distance_goal , drone_position ) 
        if closest_dist > distance_goal :
            closest_dist = distance_goal

        reward = 0
        reward_d = 0              
        reward_pr = 0
        reward_x = 0
        reward_y = 0
        reward_m =  0


        end_distance =  distance_goal
        
        if max_distance < distance_goal:
            max_distance = distance_goal

        #print(state[13] - distance_goal > 0.0001 )
        
        
       
        if (global_x <1 and global_y < 1) and ((lidar_distance > 0.1) and (lidar_distance < 4)) and (abs(pitch) < 0.05) and (abs(roll) < 0.05) and  (abs((abs(yaw)-abs(state[10]))) < 0.0001):

            reward_d = 0.01

            if lidar_distance > 3.8 and lidar_distance < 4.0:

                reward_d = 1
        else:
            reward_d = 0

        
        
        
        total_reward_d  += reward_d
        total_reward_pr  += reward_pr

        

        reward = reward_d + reward_pr
        
        with open(name_data + '.csv','a') as csv_file:
            csv_writer = csv.DictWriter(csv_file , fieldnames = fieldnames)
            information = { "reward" : reward , "reward_d" : reward_d , "distance" : distance_goal, "lidar_value" : lidar_distance, "motor values" : [m1,m2,m3,m4], "global_x" : global_x , "global_y": global_y , "pitch" : pitch ,"roll" : roll, "yaw" : yaw }
            csv_writer.writerow(information)

        
       

        return reward
    
    def get_state(self,state):
        global global_x, global_y, distance_goal, lidar_distance, yaw, pitch, roll
        #obs = np.array([global_x, global_y , lidar_distance , yaw, pitch, roll, distance_goal])
        
        state[0] = state[7]
        state[1] = state[8]
        state[2] = state[9]
        state[3] = state[10]
        state[4] = state[11]
        state[5] = state[12]
        state[6] = state[13]

        
        state[7] = global_x
        state[8] = global_y
        state[9] = lidar_distance
        state[10] = yaw
        state[11] = pitch
        state[12] = roll
        state[13] = distance_goal
        
        #print(global_x, global_y, distance_goal, lidar_distance, yaw, pitch, roll)
        #print(state)
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
        self.actor_optim = Adam(self.actor.parameters(), lr= 3e-4 ,weight_decay=0.2)
        self.critic_optim = Adam(self.critic.parameters(), lr= 1e-3 ,weight_decay=0.2)

        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.002)
        self.cov_mat = torch.diag(self.cov_var).to(self.actor.device)

    def learn(self,total_timesteps):
        global m1 , m2 , m3 , m4 ,lidar_distance ,global_y , global_x ,roll, pitch, episode , total_reward_m, total_reward_d ,total_reward_xy , total_reward_pr , closest_dist , distance_goal , name_data , total_positive_reward , total_negative_reward , end_distance , max_distance
        
        t_till_now = 0
        i_till_now = 0

        while t_till_now < total_timesteps:
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()
            pause()

            t_till_now += np.sum(batch_lens)
            # Increment the number of iterations
            i_till_now += 1
            

            # Calculate advantage at k-th iteration
            V, _ = self.evaluate(batch_obs, batch_acts)
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
        global m1 , m2 , m3 , m4 ,lidar_distance ,global_y , global_x ,roll, pitch, episode , total_reward_m, total_reward_d ,total_reward_xy , total_reward_pr , closest_dist , distance_goal , name_data , total_positive_reward , total_negative_reward , end_distance , max_distance  ,highest_point ,highest_x , highest_y , highest_pitch , highest_roll , up_actions , down_actions ,episode_count 
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
            episode_number = episode_count/self.max_timesteps_per_episode
            name_data = my_folder + str(episode_number)
            closest_dist = 100
            end_distance = 0 
            max_distance = 0

            highest_point = 0
            highest_x = 0
            highest_y = 0
            highest_pitch = 0 
            highest_roll = 0
            up_actions = 0
            down_actions = 0
            total_reward_m = 0
            total_reward_xy = 0
            total_reward_d = 0
            total_reward_pr = 0
            total_positive_reward = 0
            total_negative_reward = 0
            

            with open(name_data + '.csv','w') as csv_file:
                csv_writer = csv.DictWriter(csv_file , fieldnames = fieldnames)
                csv_writer.writeheader()
            # Run an episode for a maximum of max_timesteps_per_episode timesteps
            for ep_t in range(self.max_timesteps_per_episode):

                t += 1 # Increment timesteps ran this batch so far
                # print(t)
                # Track observations in this batch
                episode_count += 1
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
            with open(name,'a') as csv_file_2:
                csv_writer_2 = csv.DictWriter(csv_file_2 , fieldnames = fieldnames2)
                information2 = {"epsodic reward" : sum(ep_rews) ,"total_positive_reward" : total_positive_reward, "total_negative_reward" : total_negative_reward, "closest_dist" : closest_dist, "max_height" : highest_point,"end_distance": end_distance,"max_distance": max_distance, "up_actions(100)":up_actions , "down_actions(0)":down_actions, "highest_x" : highest_x, "highest_y" : highest_y , "highest_pitch" : highest_pitch, "highest_roll" : highest_roll,"epsodic reward_m" : total_reward_m, "episodic reward_d" : total_reward_d, "episodic reward_xy" :total_reward_xy, "episodic reward_pr": total_reward_pr}
                #'epsodic reward' , 'total_positive_reward' , 'total_negative_reward' ,'closest_dist' ,'max_height' ,'end_distance' ,'max_distance' , 'up_actions(100)' , 'down_actions(0)', 'highest_x' , 'highest_y' , 'highest_pitch' , 'highest_roll' , 'epsodic reward_m' , 'episodic reward_d','episodic reward_y' ,'episodic reward_pr'
                csv_writer_2.writerow(information2)

            print(sum(ep_rews) , "in episode" ,episode_number ) # sum will add all the rewards
            #print("pitch/roll reward" ,total_reward_pr, "distance reward" ,total_reward_d )
            batch_lens.append(ep_t + 1)
            batch_rews.append(ep_rews)

        batch_obs = torch.tensor(batch_obs, dtype=torch.float).to(self.actor.device)
        batch_acts = np.array(batch_acts)
        
        batch_acts = torch.tensor(batch_acts, dtype=torch.float).to(self.actor.device)
        batch_log_probs = np.array(batch_log_probs)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float).to(self.actor.device)
        batch_rtgs = self.compute_rtgs(batch_rews)  
        #print(sum(batch_acts == 0),"len(batch_acts == 0)")

        if (episode_number%100 == 0):
            torch.save(self.actor.state_dict(), ac_weights_folder + 'ppo_actor'+ str(episode_number)+ '.pth')
            torch.save(self.critic.state_dict(), cr_weights_folder + 'ppo_critic' +str(episode_number)+ '.pth')        
        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens

    def compute_rtgs(self, batch_rews):
        batch_rtgs = []

        # Iterate through each episode
        for ep_rews in reversed(batch_rews):

            discounted_reward = 0 # The discounted reward so far

            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)

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
        self.n_updates_per_iteration = 4             # Number of times to update actor/critic per iteration
        self.lr = 2.5e-4                               # Learning rate of actor optimizer
        self.gamma = 0.99                               # Discount factor to be applied when calculating Rewards-To-Go
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
     
        return roll_x, pitch_y, yaw_z




velocity = Float64MultiArray()

def service():
    global roll, pitch, yaw , lidar_distance , m1 , m2 , m3 , m4 ,episode , total_reward_m, total_reward_d ,total_reward_xy , total_reward_pr , closest_dist , distance_goal , name_data , total_positive_reward , total_negative_reward , end_distance , max_distance ,highest_point ,highest_x , highest_y , highest_pitch , highest_roll , up_actions , down_actions 
    
    #print(lidar_distance,"lidar_distance")
    velocity.data = [0 , 0 , 0 , 0]
    velPub.publish(velocity)
    reset_world()
    closest_dist = 100
    end_distance = 0 
    max_distance = 0

    highest_point = 0
    highest_x = 0
    highest_y = 0
    highest_pitch = 0 
    highest_roll = 0
    up_actions = 0
    down_actions = 0
    total_reward_m = 0
    total_reward_xy = 0
    total_reward_d = 0
    total_reward_pr = 0
    total_positive_reward = 0
    total_negative_reward = 0
    
    time.sleep(1)
    
    model = PPO()
    model.learn(100000000)

    


def lasercall_back(msg):
    global roll, pitch, yaw , lidar_distance , m1 ,m2 ,m3 , m4
    #print("step 2") 
    distance = msg.ranges
    lidar_distance = min(distance)
    #print(lidar_distance)
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