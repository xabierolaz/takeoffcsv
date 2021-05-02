#!/usr/bin/env python
# ROS python API
import rospy
from numpy import inf
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import numpy as np
import csv
from gym import spaces

# 3D point & Stamped Pose msgs
# it is a Laserscan not a point cloude so we may not need the point msg here

# import all mavros messages and services
from collections import namedtuple
import torch 
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
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
fieldnames = ['reward','reward_m','reward_d','reward_yaw','reward_pr' , 'motor values','distance' , 'lidar_value' , 'x_coordinate' , 'y_coordiante' , 'pitch' ,'roll' ,'wrong action']
fieldnames2 = ['epsodic reward' , 'total_positive_reward' , 'total_negative_reward' ,'closest_dist' ,'max_height' ,'end_distance' ,'max_distance' , 'up_actions' , 'down_actions', 'highest_x' , 'highest_y' , 'highest_pitch' , 'highest_roll' , 'epsodic reward_m' , 'episodic reward_d','episodic reward_yaw' ,'episodic reward_pr','wrong action' ]

m1 = 50
m2 = 50
m3 = 50
m4 = 50


#print("step define global")
global roll, pitch, yaw , lidar_distance ,velocity , global_x , global_y  , episode , total_reward_m, total_reward_d ,total_reward_yaw , total_reward_pr , closest_dist , distance_goal , name_data , total_positive_reward , total_negative_reward  , end_distance , max_distance , episode_number , up_actions , down_actions
episode_number = 0




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
            nn.Linear(in_dim , 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512,out_dim),
            nn.Softmax(dim = -1)
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self,obs):
        if isinstance(obs ,np.ndarray):
            obs = torch.tensor(obs ,dtype= torch.float).to(self.device)
        
        out = self.model(obs)

        return out



class Environment():
    global m1, m2 , m3 , m4 ,lidar_distance ,global_y , global_x ,roll, pitch, episode , total_reward_m, total_reward_d ,total_reward_yaw , total_reward_pr , closest_dist , distance_goal , name_data , total_positive_reward , total_negative_reward , end_distance , max_distance
    def __init__(self):
        
        observation_high = np.array([np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max,np.finfo(np.float32).max,np.finfo(np.float32).max, np.finfo(np.float32).max , np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max , np.finfo(np.float32).max])
        
        self.observation_space = spaces.Box(-observation_high, observation_high)


    def reset(self):
        obs = np.array([0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0])
        #print(obs.shape ,"obs.shape()")
        wrong_action = 0
        reset_world()
        velocity.data = [0 , 0 , 0 , 0]
        velPub.publish(velocity)
        rospy.sleep(1)
        return obs ,wrong_action


    def step(self,action,state ,w_a):
        global m1, m2 , m3 , m4 , global_y , global_x , lidar_distance , yaw , roll , pitch 
        change_error = 0
        
        #action = previous motor state +/- change
        
        
        
        change = 20            
        max = 70
        min = 30
        
        
        
        '''
        #auto
        if lidar_distance > 5 :
            action = 1         
        elif pitch < -0.05:
            action = 4
        elif pitch > 0.05:
            action = 5  
        elif roll < -0.05:
            action = 2
        elif roll > 0.05:
            action = 3 
        elif lidar_distance < 5 :
            action = 0
        '''
        #ROLL--
               
         

        if action == 0 : #up
            #print("action 0")
            m1 = 50 + change
            m2 = 50 + change
            m3 = 50 + change
            m4 = 50 + change 
        

        elif action == 1 : #down
            #print("action 1")
            m1 = 50 - change
            m2 = 50 - change
            m3 = 50 - change
            m4 = 50 - change     
        
        elif action == 2 : #roll + 
            #print("action 2")
            m1 = 50 + change
            m2 = 50 + change
            m3 = 50 - change
            m4 = 50 - change
        elif action == 3 : #roll - #CONFIRMED
            #print("action 3")
            m1 = 50 - change
            m2 = 50 - change
            m3 = 50 + change
            m4 = 50 + change
        elif action == 4 : #pitch +
            #print("action 4")
            m1 = 50 - change
            m2 = 50 + change
            m3 = 50 - change
            m4 = 50 + change
            
        elif action == 5 : #pitch -
            #print("action 5")
            m1 = 50 + change
            m2 = 50 - change
            m3 = 50 + change
            m4 = 50 - change
        
        
        
        '''
        elif action == 6 : #yaw +
            m1= state[7] + change/5.0
            m2 = state[8] + change/5.0
            m3 = state[9] + change/5.0
            m4 = state[10] + change/5.0
        elif action == 7 : #yaw -
            m1= state[7] - change/5.0
            m2 = state[8] - change/5.0
            m3 = state[9] - change/5.0
            m4 = state[10] - change/5.0
        
        if action == 0 : #up
            #print("action 0")
            m1= state[18] + change
            m2 = state[19] - change
            m3 = state[20] + change
            m4 = state[21] - change
        ''' 
        
         
        
        
        
      
        #IF WE GET AWAY blOM X OR Y ITS OK TO USE PITCH AND ROLL (TO A MAXIMUM) IF THE DISTANCE TO THE POINT IS BEING REDUCED

        if m1 > max:
            print("mayor")
            m1 = max
        if m1 < min:
            m1 = min
        if m2 > max:
            m2 = max
        if m2 < min:
            m2 = min
        if m3 > max:
            m3 = max
        if m3 < min:
            m3 = min
        if m4 > max:
            m4 = max
        if m4 < min:
            m4 = min
        # motors 3 and 4 are swaped in sdf
        velocity.data = [abs(m1) ,-abs(m2) , abs(m4), -abs(m3)]
        velPub.publish(velocity)

        reward , wrong_action = self.get_reward(state ,action , w_a)
        next_state = self.get_state(state)
       
        #FIX GROUND DISTANCE DIFFERENCE OF 0.10000000149 BECAUSE OF SDF MODEL
        

        return next_state , reward , wrong_action

    #REWARDS

    def get_reward(self , state, action , wrong_action):        
        global m1, m2 , m3 , m4 ,lidar_distance ,global_y , global_x ,roll, pitch, yaw, episode , total_reward_m, total_reward_d ,total_reward_yaw , total_reward_pr , closest_dist , distance_goal , name_data , total_positive_reward , total_negative_reward , end_distance , max_distance  , up_actions , down_actions
    
        drone_position = [global_x , global_y , lidar_distance]
        goal_position = [0 , 0 , 5]
        distance_goal = distance.euclidean(goal_position , drone_position)
        reward = 0
        reward_d = 0
        reward_pr = 0
        reward_yaw = 0
        reward_m =  0
        
        
        #closest  drone got to desired location is calculated here
        if closest_dist > distance_goal :
            closest_dist = distance_goal

        end_distance =  distance_goal

        # max distance by drone is getting calculated here
        if max_distance < distance_goal:
            max_distance = distance_goal
        
        
        
        distance_error = -distance_goal
        pr_error = -(abs(pitch) + abs(roll))
        #yaw_error = -(abs(state[3])-abs(state[14]))#WIP
        motor_error = 0
        
        
        #REWARD FUNCTION 2.0
        #reward function below this line
       
       
        #TESTS

       
        
        reward_d = lidar_distance*0.01
            
      
        
        #REWARD_D FOR GOING UP REWARD_M FOR GOING
       
        total_reward_d  += reward_d
        total_reward_pr  += reward_pr
        total_reward_m += reward_m
        total_reward_yaw += reward_yaw
        

        reward = reward_d + reward_pr + reward_m + reward_yaw
        
        with open(name_data + '.csv','a') as csv_file:
            csv_writer = csv.DictWriter(csv_file , fieldnames = fieldnames)
            information = { "reward" : reward , "reward_d" : reward_d , "distance" : distance_goal, "lidar_value" : lidar_distance,"reward_yaw" : reward_yaw ,"reward_pr" : reward_pr  ,"reward_m" : reward_m , "motor values" : [m1,m2,m3,m4], "x_coordinate" : global_x , "y_coordiante": global_y , "pitch" : pitch ,"roll" : roll ,"wrong action" : wrong_action }
            csv_writer.writerow(information)

        
       

        return reward ,wrong_action
    
    def get_state(self,state):
        global global_x, global_y, distance_goal, lidar_distance, yaw, pitch, roll, m1, m2, m3, m4
        
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
        
        state[11] = global_x
        state[12] = global_y
        state[13] = lidar_distance
        state[14] = yaw
        state[15] = pitch
        state[16] = roll
        state[17] = distance_goal
        state[18] = m1
        state[19] = m2
        state[20] = m3
        state[21] = m4
        
        
        return state



class PPO():
    def __init__(self):

        self.env = Environment()
        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = 5 #self.env.action_space.shape
        self._init_hyperparameters()
        
        self.actor = Network(self.obs_dim , self.act_dim)
        self.critic = Network(self.obs_dim , 1)
        

        self.actor_optim = Adam(self.actor.parameters(), lr= 1e-4 ,weight_decay=0.2)
        self.critic_optim = Adam(self.critic.parameters(), lr= 3e-4 , weight_decay=0.2)

    def learn(self,total_timesteps):
        global m1, m2 , m3 , m4 ,lidar_distance ,global_y , global_x ,roll, pitch, episode , total_reward_m, total_reward_d ,total_reward_yaw , total_reward_pr , closest_dist , distance_goal , name_data , total_positive_reward , total_negative_reward , end_distance , max_distance 
        
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

            # This is the loop where we update our network for some n epochs
            for _ in range(self.n_updates_per_iteration):                                                       
                # Calculate V_phi and pi_theta(a_t | s_t)
                V, curr_log_probs = self.evaluate(batch_obs, batch_acts)
               
                ratios = torch.exp(curr_log_probs - batch_log_probs)
                # Calculate surrogate losses.
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
            unpause()

            


    def rollout(self):
        global m1, m2 , m3 , m4 ,lidar_distance ,global_y , global_x ,roll, pitch, episode , total_reward_m, total_reward_d ,total_reward_yaw , total_reward_pr , closest_dist , distance_goal , name_data , total_positive_reward , total_negative_reward , end_distance , max_distance  ,highest_point ,highest_x , highest_y , highest_pitch , highest_roll , up_actions , down_actions ,episode_number 
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
            obs ,wrong_action = self.env.reset()
            
            done = False
            
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
            total_reward_yaw = 0
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
                
                batch_obs.append(obs)

                # Calculate action and make a step in the env. 
                # Note that rew is short for reward.
                action, log_prob = self.get_action(obs)
                #action =0
                obs, rew , wrong_action = self.env.step(action, obs ,wrong_action)


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

                if action ==0 :
                    up_actions += 1
                if action ==1 :
                    down_actions += 1


                # Track recent reward, action, and action log probability
                ep_rews.append(rew)  # we are collecting all the rewards in ep_rews

                batch_acts.append(action)
                batch_log_probs.append(log_prob)
                if distance_goal > 10:
                    break

            with open(name,'a') as csv_file_2:
                csv_writer_2 = csv.DictWriter(csv_file_2 , fieldnames = fieldnames2)
                information2 = {"epsodic reward" : sum(ep_rews) ,"total_positive_reward" : total_positive_reward, "total_negative_reward" : total_negative_reward, "closest_dist" : closest_dist, "max_height" : highest_point,"end_distance": end_distance,"max_distance": max_distance, "up_actions": up_actions , "down_actions": down_actions, "highest_x" : highest_x, "highest_y" : highest_y , "highest_pitch" : highest_pitch, "highest_roll" : highest_roll,"epsodic reward_m" : total_reward_m, "episodic reward_d" : total_reward_d, "episodic reward_yaw" :total_reward_yaw, "episodic reward_pr": total_reward_pr , "wrong action" : wrong_action}
                #'epsodic reward' , 'total_positive_reward' , 'total_negative_reward' ,'closest_dist' ,'max_height' ,'end_distance' ,'max_distance' , 'up_actions(100)' , 'down_actions(0)', 'highest_x' , 'highest_y' , 'highest_pitch' , 'highest_roll' , 'epsodic reward_m' , 'episodic reward_d','episodic reward_yaw' ,'episodic reward_pr'
                csv_writer_2.writerow(information2)

            print(sum(ep_rews) , "in episode" ,episode_number, "up actions", up_actions, "down actions", down_actions , "wrong actions" , wrong_action ) # sum will add all the rewards
            #print("pitch/roll reward" ,total_reward_pr, "distance reward" ,total_reward_d )
            episode_number += 1 
            batch_lens.append(ep_t + 1)
            batch_rews.append(ep_rews)

        batch_obs = torch.tensor(batch_obs, dtype=torch.float).to(self.actor.device)
        batch_acts = np.array(batch_acts)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float).to(self.actor.device)
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
        # Query the actor network for a mean action
        out = self.actor(obs)
        #print(sum(out))
        dist = Categorical(logits=out)
        # Sample an action blom the distribution
        action = dist.sample()
        # Calculate the log probability for that action
        log_prob = dist.log_prob(action)
        # Return the sampled action and the log probability of that action in our distribution
        return action.cpu().detach().numpy(), log_prob.detach()

    def evaluate(self, batch_obs, batch_acts):
        V = self.critic(batch_obs).squeeze()

        # Calculate the log probabilities of batch actions using most recent actor network.
        # This segment of code is similar to that in get_action()
        out = self.actor(batch_obs)
        
        dist = Categorical(logits=out)
        log_probs = dist.log_prob(batch_acts)

		# Return the value vector V of each observation in the batch
		# and log probabilities log_probs of each action in the batch
        return V, log_probs

    def _init_hyperparameters(self):
        # Initialize default values for hyperparameters
        # Algorithm hyperparameters
        
        self.max_timesteps_per_episode = 4096   # Max number of timesteps per episode
        self.timesteps_per_batch = 3*self.max_timesteps_per_episode          # Number of timesteps to run per batch
        self.n_updates_per_iteration = 5                # Number of times to update actor/critic per iteration                                                        
        self.gamma = 0.95                               # Discount factor to be applied when calculating Rewards-To-Go
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
    global roll, pitch, yaw , lidar_distance , m1, m2 , m3 , m4 ,episode , total_reward_m, total_reward_d ,total_reward_yaw , total_reward_pr , closest_dist , distance_goal , name_data , total_positive_reward , total_negative_reward , end_distance , max_distance ,highest_point ,highest_x , highest_y , highest_pitch , highest_roll , up_actions , down_actions 
    
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
    change_error = 0
    down_actions = 0
    total_reward_m = 0
    total_reward_yaw = 0
    total_reward_d = 0
    total_reward_pr = 0
    total_positive_reward = 0
    total_negative_reward = 0
    
    time.sleep(1)
    
    model = PPO()
    model.learn(50000000)

    


def lasercall_back(msg):
    global roll, pitch, yaw , lidar_distance , m1,m2 ,m3 , m4
    #print("step 2") 
    distance = msg.ranges
    lidar_distance = min(distance)
    #print(lidar_distance)
    if (lidar_distance == inf or lidar_distance == -inf) :
        lidar_distance = 50
        


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