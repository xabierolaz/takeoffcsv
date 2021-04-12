#!/usr/bin/env python
# ROS python API
import rospy
from numpy import inf

# 3D point & Stamped Pose msgs
# it is a Laserscan not a point cloude so we may not need the point msg here
from std_msgs.msg import Float64MultiArray, Float32

# import all mavros messages and services
from collections import namedtuple
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import math
import random
from sensor_msgs.msg import LaserScan
from gazebo_msgs.msg import ModelStates
import thread
import time
from scipy.spatial import distance

from geometry_msgs.msg import Pose
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState
from datetime import datetime
from std_srvs.srv import Empty

pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)


rospy.wait_for_service("/gazebo/set_model_state")
m = rospy.ServiceProxy("/gazebo/set_model_state",SetModelState)

command = ModelState()
command.model_name = "Kwad"

from std_srvs.srv import Empty
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

velPub = rospy.Publisher('/Kwad/joint_motor_controller/command', Float64MultiArray, queue_size=4)
reset_world = rospy.ServiceProxy('/gazebo/reset_world', Empty)
fieldnames = ['reward','reward_m','reward_d','reward_y','reward_rp' , 'motor values','distance' , 'lidar_value' , 'x_coordinate' , 'y_coordiante' , 'pitch' ,'roll']
fieldnames2 = ['epsodic reward' , 'epsodic reward_m' , 'episodic reward_d','episodic reward_y' ,'episodic reward_rp','closest_dist' ,'highest_point', 'highest_x' , 'highest_y' , 'highest_pitch' , 'highest_roll']
m1 = 0
m2 = 0
m3 = 0
m4 = 0

class Critic(nn.Module):
    def __init__(self , input_dims):
        super(Critic , self).__init__()

        self.critic = nn.Sequential(
            nn.Linear(in_features= input_dims , out_features= 100),
            nn.Sigmoid(),
            nn.Linear(in_features= 100 , out_features= 200),
            nn.Sigmoid(),
            nn.Linear(in_features= 200 , out_features= 100),
            nn.Sigmoid(),
            nn.Linear(in_features= 100 , out_features= 1),
            nn.Tanh()
            
        )
        self.optimizer = optim.Adam(self.parameters(),lr = 0.0003)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        #self.device = T.device('cpu')
        self.to(self.device)
        print(self.device)


    def forward(self , state):
        state = T.tensor(state , dtype= T.float).to(self.device)
        #state = state.flatten()
        #print(state)
        value = self.critic(state)
        #print(value ,"Value")
        return value


class Actor(nn.Module):
    def __init__(self , input_dims , n_actions ):
        super(Actor , self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(in_features= input_dims , out_features= 100),
            nn.Sigmoid(),
            nn.Linear(in_features= 100 , out_features= 200),
            nn.Sigmoid(),
            nn.Linear(in_features= 200 , out_features= 100),
            nn.Sigmoid(),
            nn.Linear(in_features= 100 , out_features= n_actions),
            nn.Softmax(dim=-1)
        )
        self.optimizer = optim.Adam(self.parameters(), lr= 0.00003)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        #self.device = T.device('cpu')
        self.to(self.device)


    def forward(self , state):
        state = T.tensor(state , dtype= T.float).to(self.device)
        #state = state.flatten()
        #print(state)
        dist = self.actor(state)
        #print(dist , "dist")
        dist  = Categorical(dist)
        return dist



class agent(object):
    def __init__( self , batch_size  , n_epochs , n_actions = 9   , input_dims = 44):
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.input_dims = input_dims
        # choose between using loaded weight or fresh start (una de las dos lineas, solo, actor o t.load)
        self.actor = Actor(self.input_dims , self.n_actions) #fresh start
        #self. actor = T.load('/home/cuda/Desktop/ac_weights_load_this/actor_weights_final.pth') loaded weight   
        self.critic = Critic(self.input_dims)
        self.gamma = 0.99 #increased horizon
        self.gae_lambda = 0.95
        self.policy_clip = 0.2

    def get_action(self, dist):
        ac = dist.sample()
        #print(ac , "action")
        return ac

    def learn(self, states , actions , values , rewards , probs):
        
        for i in range(self.n_epochs):
            #print("epoch" , i)
            
            st_arr ,ac_arr , old_prob_arr ,val_arr , rewards_arr , batches = self.get_batch(states , actions , values , rewards , probs)
            vals = val_arr
            adv = np.zeros(len(rewards_arr),dtype=np.float32)
            for j in range(len(rewards_arr) -1):
                discount = 1
                a_t = 0 
                for k in range(j , len(rewards_arr) -1):
                    a_t += discount*(rewards_arr[k] + self.gamma*vals[k+1] - vals[k])
                    discount *= self.gamma*self.gae_lambda
                adv[j] = a_t
            
            #print(rewards_arr , "reward array")
            #print(adv , "advantage")
            adv = T.tensor(adv).to(self.actor.device)
            vals = T.tensor(vals).to(self.actor.device)

            for batch in batches :
                
                sts = T.tensor(st_arr[batch]).to(self.actor.device)
                old_probs = T.tensor(old_prob_arr[batch]).to(self.actor.device)
                acs = T.tensor(ac_arr[batch]).to(self.actor.device)

                dist = self.actor.forward(sts)
                #print(dist)
                cr_val = self.critic.forward(sts)

                cr_val = T.squeeze(cr_val)
                new_probs = dist.log_prob(acs)
                
                prob_ratio = new_probs.exp() / old_probs.exp()
                
                weighted_probs = adv[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1-self.policy_clip,
                        1+self.policy_clip)*adv[batch]

                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                returns = adv[batch] + vals[batch]
                critic_loss = (returns-cr_val)**2
                critic_loss = critic_loss.mean()
                total_loss = actor_loss + 0.5*critic_loss
                #print(actor_loss , critic_loss , total_loss , "actor_loss , critic_loss , total_loss ")
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                T.nn.utils.clip_grad_norm_(self.actor.parameters() ,5)
                T.nn.utils.clip_grad_norm_(self.critic.parameters() , 5)
                self.actor.optimizer.step()
                self.critic.optimizer.step()
    
    def get_batch(self , states , actions , values , rewards , probs):
        num = len(states)
        start_of_batch = np.arange(0, num, self.batch_size)
        indices = np.arange(num, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in start_of_batch]
        
        return np.array(states) , np.array(actions , dtype= np.int) , np.array(probs) , np.array(values ,dtype=np.float) , np.array(rewards) , batches


class Environment():
    def __init__(self):
        self.state = []
        self.reward = []
    
    def step(self,state,action):
        global m1 , m2 , m3 , m4 , global_y , global_x , lidar_distance , yaw , roll , pitch
        
        #print(action, "Neural Network action")
        #8 ACTIONS
        #print(lidar_distance)
        
        #action =  0
        if action == 0 : #up
            m1 = 100
            m2 = 100
            m3 = 100
            m4 = 100
        elif action == 1: #pitch+
            m1 = 45
            m2 = 55
            m3 = 55
            m4 = 45
        elif action == 2: #pitch-
            m1 = 55
            m2 = 45
            m3 = 45
            m4 = 55
        elif action == 3: #roll+
            m1 = 45
            m2 = 45
            m3 = 55
            m4 = 55
        elif action == 4: #roll+-
            m1 = 55
            m2 = 55
            m3 = 45
            m4 = 45
        elif action == 5: #yaw+
            m1 = 55
            m2 = 45
            m3 = 55
            m4 = 45
        elif action == 6: #yaw-
            m1 = 45
            m2 = 55
            m3 = 45
            m4 = 55
        elif action == 7: #mantain
            m1 = 50
            m2 = 50
            m3 = 50
            m4 = 50
        
        elif action == 8: #down
            m1 = 0
            m2 = 0
            m3 = 0
            m4 = 0

    
        #CONVERT TO REWARD
        velocity.data = [abs(m1) ,-abs(m2) , abs(m3), -abs(m4)]
        velPub.publish(velocity)

        reward = self.get_reward(state ,action )
        next_state = self.get_state(state)
       
        #FIX GROUND DISTANCE DIFFERENCE OF 0.10000000149 BECAUSE OF SDF MODEL
        '''if lidar_distance <= 0.11 :
            lidar_distance = 0'''
        return next_state , reward 
    
    #REWARDS

    def get_reward(self , state , action):        
        global m1 , m2 , m3 , m4 ,lidar_distance ,global_y , global_x ,roll, pitch, episode , total_reward_m, total_reward_d ,total_reward_y , total_reward_rp , closest_dist , distance_goal , name_data
        
        #print(m1,m2,m3,m4)

        drone_position = [global_x , global_y , lidar_distance]
        goal_position = [0 , 0 , 7]
        distance_goal = distance.euclidean(goal_position , drone_position)      
        reward_d = 0
        reward = 0
        reward_m = 0
        if closest_dist > distance_goal :
            closest_dist = distance_goal
        
        
        
        #DISTANCE
        
        if distance_goal < 6 and lidar_distance < 8 :
            reward_d = (math.exp(-0.166*(distance_goal**2)))*0.0001
       
        else:
            reward_d = -(distance_goal * 0.000005)
        
        total_reward_d  += reward_d
        '''
        #extra adjustment
        if lidar_distance > 6 and < 7.2 and distance_goal < 0.5 #extra requisite here:
            reward = reward_d + reward_d * 2
                if global_x < 1 and global_y < 1 and pitch < 0.1 and roll < 0.1
                     reward = reward_d + reward_d * 3
        else
        reward = +0
        '''
        #punishments
        #print(distance_goal,"distance_goal" , lidar_distance , "lidar_distance" , reward_d , "distance reward")
        

        #reward_d = 0
        reward_y = 0
        reward_rp = 0
        
        #print(reward_d)
        reward = reward_d

        with open(name_data + '.csv','a') as csv_file:
            csv_writer = csv.DictWriter(csv_file , fieldnames = fieldnames)
            information = { "reward" : reward ,"reward_m" : reward_m , "reward_d" : reward_d , "reward_y" : reward_y ,"reward_rp" : reward_rp  , "motor values" : [m1,m2,m3,m4] , "distance" : distance_goal, "lidar_value" : lidar_distance, "x_coordinate" : global_x , "y_coordiante": global_y , "pitch" : pitch ,"roll" : roll }
            csv_writer.writerow(information)
        
        
        return reward
    
    def get_state(self,state):
        global global_x, global_y, lidar_distance, yaw, pitch, roll, m1, m2, m3 ,m4 , distance_goal
        state[0] = state[1]
        state[1] = state[2]
        state[2] = state[3]
        state[3] = state[4]
        state[4] = lidar_distance
        return state


# Graphs
import numpy as np
import csv

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


global roll, pitch, yaw , lidar_distance ,velocity , global_x , global_y  , episode , total_reward_m, total_reward_d ,total_reward_y , total_reward_rp , closest_dist , distance_goal , name_data


velocity = Float64MultiArray()

def service():
    global roll, pitch, yaw , lidar_distance , m1 , m2 , m3 , m4 ,episode , total_reward_m, total_reward_d ,total_reward_y , total_reward_rp , closest_dist , distance_goal , name_data
    env = Environment()
    N =2048
    batch_size = 128
    n_epochs = 4
    
    drone = agent(n_actions = 9 ,batch_size = batch_size , n_epochs = n_epochs ,input_dims = 5)
    reward_history = []
    Number_of_episodes = 10000
    #print(lidar_distance,"lidar_distance")
    name =  'EP_data/data.csv'+ str(datetime.now())
    with open(name,'w') as csv_file_2:
        csv_writer_2 = csv.DictWriter(csv_file_2 , fieldnames = fieldnames2)
        csv_writer_2.writeheader()
    reward_file_name ='rewards_new_2'+str(datetime.now())
    
    for i in range(Number_of_episodes):
        print(i)
        m1 = 0
        m2 = 0
        m3 = 0
        m4 = 0
        velocity.data = [0 , 0 , 0 , 0]
        velPub.publish(velocity)
        reset_world()
        rospy.sleep(2)
        done = False
        drone_position = [global_x , global_y , lidar_distance]
        goal_position = [0 , 0 , 7]
        distance_goal = distance.euclidean(goal_position , drone_position)
        
        state = [0]*5
        j = 0
        
        ep_reward = 0
        closest_dist = 100
        highest_point = 0
        highest_x = 0
        highest_y = 0
        highest_pitch = 0 
        highest_roll = 0
        
        states=[]
        values = []
        actions = []
        rewards = []
        probs = []
        total_reward_m = 0
        total_reward_y = 0
        total_reward_d = 0
        total_reward_rp = 0
        file_ = open(reward_file_name, 'a')
        
        episode = i
        name_data = 'data/data'+str(datetime.now())+ str(episode) 
        with open(name_data +'.csv','w') as csv_file:
            csv_writer = csv.DictWriter(csv_file , fieldnames = fieldnames)
            csv_writer.writeheader()
        k =  0
        for k in range(N):
                
            dist = drone.actor.forward(np.array(state))
            value = drone.critic.forward(np.array(state))
            action = drone.get_action(dist)
            prob = T.squeeze(dist.log_prob(action)).item()
            #print(action)
            next_state , reward = env.step(state,action)

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

                           

            actions.append(action)
            values.append(value)
            #print(reward)
            states.append(state[:])
            rewards.append(reward)
            #dones.append(done)
            probs.append(prob)
            state = next_state
            ep_reward = ep_reward+reward
            k = k+1
            if k % 512 == 0:
                pause()
                drone.learn(states , actions , values , rewards , probs)
                unpause()
                states=[]
                values = []
                actions = []
                rewards = []
                probs = []
        reward_history.append(ep_reward)
        print("episode reward", ep_reward)


        with open(name,'a') as csv_file_2:
            csv_writer_2 = csv.DictWriter(csv_file_2 , fieldnames = fieldnames2)
            information2 = { "epsodic reward" : ep_reward , "epsodic reward_m" : total_reward_m , "episodic reward_d" : total_reward_d ,"episodic reward_y" :total_reward_y ,"episodic reward_rp": total_reward_rp , "closest_dist" : closest_dist , "highest_point" : highest_point, "highest_x" : highest_x , "highest_y" : highest_y  , "highest_pitch" : highest_pitch , "highest_roll" : highest_roll}
            csv_writer_2.writerow(information2)

        file_.write(str(ep_reward) + str("\n"))
        file_.close()

        if ep_reward >= np.mean(reward_history):
            T.save(drone.actor,'/home/cuda/Desktop/ac_weights/actor_weights' +str(i)+'.pth')
    T.save(drone.actor,'/home/cuda/Desktop/ac_weights/actor_weights_final.pth')


'''def wind_():
    global roll, pitch, yaw , lidar_distance , m1 , m2 , m3 , m4 ,episode , total_reward_m, total_reward_d ,total_reward_y , total_reward_rp , closest_dist , distance_goal , name_data
    for i in range(100):

        print(global_x , "wind can be")'''

def lasercall_back(msg):
    global roll, pitch, yaw , lidar_distance , m1 ,m2 ,m3 , m4 
    distance = msg.ranges
    lidar_distance = min(distance)
    if (lidar_distance == inf or lidar_distance == -inf) :
        lidar_distance = 0
        m1 = 0
        m2 = 0
        m3 = 0
        m4 = 0
        velocity.data = [0 , 0 , 0 , 0]
        velPub.publish(velocity)
        reset_world()



def location_callback(msg):
    global roll, pitch, yaw , lidar_distance , global_x , global_y , global_z
    ind = msg.name.index('Kwad')
    #print(ind)
    orientationObj = msg.pose[ind].orientation
    positionObj = msg.pose[ind].position
    global_x = positionObj.x
    global_y = positionObj.y
    roll, pitch, yaw = euler_from_quaternion(orientationObj.x, orientationObj.y, orientationObj.z, orientationObj.w)

def main():
    rospy.init_node('Drone_control')
    rospy.Subscriber("/Kwad/scan" , LaserScan , lasercall_back)
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
