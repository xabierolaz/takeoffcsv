#!/usr/bin/env python
# ROS python API
import rospy
from numpy import inf
from scipy.spatial.transform import Rotation as Rotation
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

torch.set_printoptions(precision=None, threshold=10000000000, edgeitems=None, linewidth=None, profile=None,
                       sci_mode=None)
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
unpause()

rospy.wait_for_service("/gazebo/set_model_state")
m = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)

command = ModelState()
command.model_name = "Kwad"

velPub = rospy.Publisher('/Kwad/joint_motor_controller/command', Float64MultiArray, queue_size=1)

fieldnames = ['reward', 'motor values', 'distance', 'distance_xy', 'lidar_value', 'global_x', 'global_y', 'pitch',
              'roll', 'yaw']
fieldnames2 = ['episode reward', 'closest_dist', 'max_height', 'end_distance', 'max_distance', 'highest_x', 'highest_y',
               'highest_pitch', 'highest_roll']
m1 = 0
m2 = 0
m3 = 0
m4 = 0

global roll, pitch, yaw, lidar_distance, distance_xy, velocity, global_x, global_y, episode, closest_dist, distance_goal, name_data, end_distance, max_distance, episode_number, change, last_position
last_position = np.array([0, 0, 0])

change = 35
episode_number = 0
velocity = Float64MultiArray()

name = '/home/cuda/Desktop/TRAININGS/' + str(datetime.now())
with open(name, 'w') as csv_file_2:
    csv_writer_2 = csv.DictWriter(csv_file_2, fieldnames=fieldnames2)
    csv_writer_2.writeheader()
# reward_file_name ='rewards_new_2'+str(datetime.now())
my_folder = '/home/cuda/Desktop/TRAININGS/EPISODES/' + str(datetime.now()) + '/'
ac_weights_folder = '/home/cuda/Desktop/TRAININGS/AC_WEIGHTS/' + str(datetime.now()) + '/'
cr_weights_folder = '/home/cuda/Desktop/TRAININGS/CR_WEIGHTS/' + str(datetime.now()) + '/'

if not os.path.exists(my_folder):
    os.makedirs(my_folder)

if not os.path.exists(ac_weights_folder):
    os.makedirs(ac_weights_folder)

if not os.path.exists(cr_weights_folder):
    os.makedirs(cr_weights_folder)


# Running stats code - Used to normalize the returns and advantages over a large batch of episodes
class running_stats:
    """
    Running stats (average and standard deviation).
    """
    from pathlib import Path
    import torch
    root_path = Path(__file__).parent

    def __init__(self, name, device):
        self.name = name
        self.device = device
        self.mean = None
        # Tries to load the file on the machine
        try:
            name = 'policy/' + self.name + '_running_stats.npz'
            file = np.load(self.root_path / name)
            self.length = torch.tensor(file['arr_0']).to(device)
            self.average = torch.tensor(file['arr_1']).to(device)
            self.M2 = torch.tensor(file['arr_2']).to(device)
            self.std = torch.sqrt(self.M2 / (self.length - 1))
            print(self.name + ' running stats loaded' + ' Average: {:.2f} Std: {:.2f}'.format(self.average, torch.sqrt(
                self.M2 / (self.length - 1))))
        # else create new files
        except:
            self.length = torch.tensor(0.).to(device)
            self.average = torch.tensor(0.).to(device)
            self.M2 = torch.tensor(0.).to(device)
            self.std = 1
            print(self.name + ' running stats created')

    def reset(self):
        # Resets the running stats
        self.length = torch.tensor(0.).to(self.device)
        self.average = torch.tensor(0.).to(self.device)
        self.M2 = torch.tensor(0.).to(self.device)

    def new_sample(self, sample):
        # Uses a new sample to update the running stats
        self.length += sample.shape[0]
        delta = sample - self.average
        self.average += torch.sum(delta / self.length)
        delta_2 = sample - self.average
        self.M2 += torch.sum(delta * delta_2)
        self.save()
        self.std = torch.sqrt(self.M2 / (self.length - 1))
        return self.average, self.std

    def save(self):
        # Save the stats on file
        name = 'policy/' + self.name + '_running_stats.npz'
        np.savez(self.root_path / name,
                 np.array(self.length.cpu()), np.array(self.average.cpu()), np.array(self.M2.cpu()))


reset_command = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)


def reset_service():
    command = ModelState()
    command.model_name = "Kwad"
    location = Pose()
    location.orientation.x = 0
    location.orientation.y = 0
    location.orientation.z = 0
    location.orientation.w = 1
    location.position.x = 0
    location.position.y = 0
    location.position.z = 0.05
    command.pose = location
    reset_command(command)
    velocity.data = [20, -20, 20, -20]
    velPub.publish(velocity)
    rospy.sleep(1.5)


reset_world = reset_service


class Network(nn.Module):
    def __init__(self, in_dim, out_dim, critic=False):
        super(Network, self).__init__()
        if critic:
            self.model = nn.Sequential(
                nn.Linear(in_dim, 256),
                nn.Tanh(),
                nn.Linear(256, 512),
                nn.Tanh(),
                nn.Linear(512, out_dim),
            )
        else:
            self.model = nn.Sequential(
                nn.Linear(in_dim, 256),
                nn.Tanh(),
                nn.Linear(256, 512),
                nn.Tanh(),
                nn.Linear(512, out_dim),
                nn.Tanh()
            )

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.log_std = nn.Parameter(torch.ones(out_dim) * -2.5)

    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float).to(self.device)

        out = self.model(obs)

        return out


class Environment():
    global m1, m2, m3, m4, lidar_distance, global_y, global_x, roll, pitch, episode, closest_dist, distance_goal, name_data, end_distance, max_distance, episode_number
    # adding reward shaping
    reward_shaping = None
    previous_reward_shaping = None

    # Weights for the reward function
    C_control = 0.001
    C_distance = 10
    C_velocity = 1
    C_attitude = 0.5

    R_solved = 1200
    R_drift = -20

    def __init__(self):
        self.rate = rospy.Rate(40)

        observation_high = np.array(
            [100, 100, 100, 100,
             100, 100, 100, 100,
             100, 100, 100, 100,
             100, 100, 100, 100,
             100, 100, 100, 100,
             100, 100, 100, 100,
             100, 100, 100, 100,
             100, 100, 100, 100,
             100, 100, 100, 100,
             100, 100, 100, 100,
             100, 100, 100, 100,
             100, 100, 100, 100,
             100, 100, 100, 100,
             100, 100])
        self.action_space = spaces.Box(np.array([-1.0, -1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0, 1.0]),
                                       dtype=np.float64)
        self.observation_space = spaces.Box(-observation_high, observation_high)

    def reset(self):
        global m1, m2, m3, m4, last_time, last_position
        obs = np.zeros(3 * 18)
        unpause()
        reset_world()
        self.rate.sleep()
        velocity.data = [20, -20, 20, -20]
        velPub.publish(velocity)
        self.rate.sleep()
        last_time = None
        last_position = np.array([0, 0, 0])
        m1 = 20
        m2 = -20
        m3 = 20
        m4 = -20

        return obs

    def step(self, action, state, ep_t):
        global m1, m2, m3, m4, global_y, global_x, lidar_distance, yaw, roll, pitch

        ''''
        motor_1 = 1500 + action[0]

        motor_2 = 1500 + action[1]

        motor_3 = 1500 + action[2]

        motor_4 = 1500 + action[3]
        #Limit esc pulses
        if(motor_1 > 2000): motor_1 = 2000
        if(motor_2 > 2000): motor_2 = 2000
        if(motor_3 > 2000): motor_3 = 2000
        if(motor_4 > 2000): motor_4 = 2000

        if(motor_1 < 1100): motor_1 = 1100
        if(motor_2 < 1100): motor_2 = 1100
        if(motor_3 < 1100): motor_3 = 1100
        if(motor_4 < 1100): motor_4 = 1100
        #Map esc values
        m1 = ((motor_1 -1500)/25) + 50
        m2 = ((motor_2 -1500)/25) + 50
        m3 = ((motor_3 -1500)/25) + 50
        m4 = ((motor_4 -1500)/25) + 50
        '''
        # print(action)
        m1 = 51 + action[0] * change
        m2 = -51 - action[1] * change
        m3 = 51 + action[2] * change
        m4 = -51 - action[3] * change

        if m1 > 100:
            m1 = 100
        if m1 < 0:
            m1 = 0
        if m2 < -100:
            m2 = -100
        if m2 > -0:
            m2 = -0
        if m3 > 100:
            m3 = 100
        if m3 < 0:
            m3 = 0
        if m4 < -100:
            m4 = -100
        if m4 > -0:
            m4 = -0

        velocity.data = np.array([m1, m2, m3, m4])
        velPub.publish(velocity)
        self.rate.sleep()
        next_state = self.get_state(state)
        done, reward = self.get_reward(next_state, action, ep_t)
        return done, next_state, reward

        # REWARDS

    def get_reward(self, state, action, ep_t):
        global m1, m2, m3, m4, lidar_distance, roll, pitch, episode, closest_dist, distance_goal, name_data, \
            end_distance, max_distance, pose_velocity, att_velocity

        # organizing the states
        goal_position = np.array([0, 0, 4])
        drone_position = np.array([global_x, global_y, lidar_distance])
        controls = np.array([m1, m2, m3, m4]) / 100
        attitude = np.array([pitch, roll, yaw])
        # drone_velocity = velocity.copy()

        # calculating the negative rewards - based on the control effort
        effort = np.sqrt(np.sum(np.square(controls)))
        distance = np.linalg.norm(drone_position - goal_position)
        distance_xy = np.linalg.norm(drone_position[0:2] - goal_position[0:2])
        distance_z = drone_position[2] - goal_position[2]
        velocity_error = np.linalg.norm(pose_velocity)
        # velocity_error = np.linalg.norm(drone_velocity)
        attitude_error = np.linalg.norm(attitude[0:2])
        # reward shaping
        self.reward_shaping = - self.C_distance * distance / 4 - self.C_attitude * attitude_error / 180 - self.C_velocity * velocity_error
        if self.previous_reward_shaping is not None:
            reward = self.reward_shaping - self.previous_reward_shaping - self.C_control * effort
        else:
            reward = - self.C_control * effort
        self.previous_reward_shaping = self.reward_shaping

        lidar_error = (-0.0112*lidar_distance**4 + 0.06*lidar_distance**3)*(1/max(0.1, distance_xy))
        ground_penalty = 0 if lidar_distance > 0.1 else -0.2

        reward = lidar_error - ((distance_xy/2)**(0.5))/10 - ((velocity_error/4)**(0.5))/20 - \
        ((attitude_error/360)**(0.5))/5 - ((abs(att_velocity[2])/np.pi)**(0.5))/5 - ((effort)**(0.5))/150 + ground_penalty

        # print(m1,m2,m3,m4)

        distance_goal = distance
        # print(distance_goal , drone_position )

        if closest_dist > distance_goal:
            closest_dist = distance_goal

        end_distance = distance_goal

        if max_distance < distance_goal:
            max_distance = distance_goal

        done = False
        # if distance less than 10 centimeters, end the episode and reward positively
        if distance < 0.2 and attitude_error < 1 and velocity_error < 0.05:
            self.count_to_solved += 1
            # print('\rSolving {:.2%}'.format(self.count_to_solved/(5*40)), end='')
            if self.count_to_solved > 5*40:
                self.reward_shaping = None
                self.previous_reward_shaping = None
                done = True
                reward = self.R_solved
                print('Solved')
        # if the distance in xy larger than 2 meters, end the episode with negative reward
        else:
            self.count_to_solved = 0
        if distance_xy > 2:
            self.reward_shaping = None
            self.previous_reward_shaping = None
            done = True
            reward = self.R_drift
            print('xy_distance')
        # if the distance in z larger than 4.5 meters, end the episode with negative reward
        elif distance_z > 0.5:
            self.reward_shaping = None
            self.previous_reward_shaping = None
            done = True
            reward = self.R_drift
            print('z_distance')
        # if the attitude error is larger than 90 degrees, end the episode with negative reward
        elif attitude_error > 30:
            self.reward_shaping = None
            self.previous_reward_shaping = None
            done = True
            reward = self.R_drift
            print('attitude_error')
        elif velocity_error > 6:
            self.reward_shaping = None
            self.previous_reward_shaping = None
            done = True
            reward = self.R_drift
            print('velocity_error')
        # if the episode timestep is over, end the episode with negative reward
        elif ep_t <= 1:
            if distance < 0.5:
                self.reward_shaping = None
                self.previous_reward_shaping = None
                done = True
                reward = self.R_solved
                print('solved')
            else:
                self.reward_shaping = None
                self.previous_reward_shaping = None
                done = True
                reward = self.R_drift
                print('timestep_max')

        with open(name_data + '.csv', 'a') as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            information = {"reward": reward, "distance": distance_goal, "distance_xy": distance_xy,
                           "lidar_value": lidar_distance, "motor values": [m1, m2, m3, m4], "global_x": global_x,
                           "global_y": global_y, "pitch": pitch, "roll": roll, "yaw": yaw}
            csv_writer.writerow(information)

        return done, reward

    def get_state(self, state):
        global global_x, global_y, distance_goal, lidar_distance, yaw, pitch, roll, pose_velocity, att_velocity, attitude
        roll, pitch, yaw = euler_from_quaternion(attitude)
        # obs = np.array([global_x, global_y , lidar_distance , yaw, pitch, roll, distance_goal])
        # changed how the states roll
        STATE_SIZE = 18
        new_state = np.array([global_x / 4, global_y / 4, lidar_distance / 4,
                              attitude[0], attitude[1], attitude[2], attitude[3],
                              np.linalg.norm([global_x, global_y, lidar_distance - 4]) / 4,
                              m1 / 100, m2 / 100, m3 / 100, m4 / 100,
                              pose_velocity[0], pose_velocity[1], pose_velocity[2],
                              att_velocity[0], att_velocity[1], att_velocity[2]])

        state = np.concatenate((state[STATE_SIZE:], new_state))

        # state[0] = state[11]
        # state[1] = state[12]
        # state[2] = state[13]
        # state[3] = state[14]
        # state[4] = state[15]
        # state[5] = state[16]
        # state[6] = state[17]
        # state[7] = state[18]
        # state[8] = state[19]
        # state[9] = state[20]
        # # added new states (position and angular velocity)
        # state[10] = state[21]
        # state[11] = state[22]
        # state[12] = state[23]
        # state[13] = state[24]
        # state[14] = state[25]
        # state[15] = state[26]
        # state[16] = state[27]
        #
        # state[17] = state[22]
        # state[18] = state[23]
        # state[19] = state[24]
        # state[20] = state[25]
        # state[21] = state[26]
        # state[22] = state[27]
        # state[23] = state[28]
        # state[24] = state[29]
        # state[25] = state[30]
        # state[26] = state[31]
        # state[27] = state[32]
        #
        # state[22] = global_x / 4
        # state[23] = global_y / 4
        # state[24] = lidar_distance / 4
        # state[25] = yaw / 180
        # state[26] = pitch / 180
        # state[27] = roll / 180
        # state[28] = np.linalg.norm([global_x, global_y, lidar_distance - 4]) / 4
        # state[29] = m1 / 100
        # state[30] = m2 / 100
        # state[31] = m3 / 100
        # state[32] = m4 / 100

        return state


class PPO():
    def __init__(self):
        self.env = Environment()
        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.shape[0]
        self._init_hyperparameters()

        self.actor = Network(self.obs_dim, self.act_dim)
        self.critic = Network(self.obs_dim, 1, critic=True)
        self.actor.load_state_dict(torch.load('/home/cuda/Desktop/TRAININGS/AC_WEIGHTS/2021-05-19 13:02:34.597074/ppo_actor200.pth'))
        self.critic.load_state_dict(torch.load('/home/cuda/Desktop/TRAININGS/CR_WEIGHTS/2021-05-19 13:02:34.597077/ppo_critic200.pth'))
        model_parameters = filter(lambda p: p.requires_grad, self.actor.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        # print(params)
        self.actor_optim = Adam(self.actor.parameters(), lr=5e-5, weight_decay=0)
        self.critic_optim = Adam(self.critic.parameters(), lr=4e-4, weight_decay=0)

        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.002)
        self.cov_mat = torch.diag(self.cov_var).to(self.actor.device)

        self.ret_stats = running_stats('ret', self.actor.device)
        self.adv_stats = running_stats('adv', self.actor.device)

    def shuffle_tensor(self, tensor_list):
        """
        @param tensor_list: list of tensors to be shuffled on batch dimension
        @return: same list with the tensors shuffled
        """
        random_index = torch.randperm(len(tensor_list[0]))
        tensor_list_shuffle = []
        for tensor in tensor_list:
            tensor_list_shuffle.append(tensor[random_index])
        return tensor_list_shuffle

    def learn(self, total_timesteps):
        global m1, m2, m3, m4, lidar_distance, global_y, global_x, roll, pitch, episode, closest_dist, distance_goal, name_data, end_distance, max_distance

        t_till_now = 0
        i_till_now = 0

        while t_till_now < total_timesteps:
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens, V = self.rollout()
            pause()

            t_till_now += np.sum(batch_lens)
            # Increment the number of iterations
            i_till_now += 1

            # Calculate advantage at k-th iteration
            # V, _ = self.evaluate(batch_obs, batch_acts)

            A_k = batch_rtgs - V.detach()
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            A_k = A_k.to(self.actor.device)
            # A_k = torch.reshape(A_k, (len(A_k), 1))
            # This is the loop where we update our network for some n epochs
            for _ in range(self.n_updates_per_iteration):
                list_tensor = batch_obs, batch_acts, batch_log_probs, batch_rtgs, A_k
                batch_obs_s, batch_acts_s, batch_log_probs_s, batch_rtgs_s, A_k_s = self.shuffle_tensor(list_tensor)
                # Calculate V_phi and pi_theta(a_t | s_t)
                V, curr_log_probs = self.evaluate(batch_obs_s, batch_acts_s)
                # print(batch_log_probs,"batch_log_probs")
                # print(curr_log_probs,"curr_log_probs")
                ratios = torch.exp(curr_log_probs - batch_log_probs_s)

                surr1 = ratios * A_k_s
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k_s

                actor_loss = (-torch.min(surr1, surr2)).mean()
                # print(actor_loss.size(), V.size(), curr_log_probs.size(), batch_obs_s.size(), batch_acts_s.size(), A_k_s.size(), ratios.size(), surr1.size(), surr2.size())
                # Calculate gradients and perform backward propagation for actor network
                self.actor_optim.zero_grad()
                actor_loss.backward()
                self.actor_optim.step()
                # Calculate gradients and perform backward propagation for critic network

            for _ in range(10 * self.n_updates_per_iteration):
                list_tensor = batch_obs, batch_acts, batch_log_probs, batch_rtgs, A_k
                batch_obs_s, batch_acts_s, batch_log_probs_s, batch_rtgs_s, A_k_s = self.shuffle_tensor(list_tensor)
                # Calculate V_phi and pi_theta(a_t | s_t)
                V, curr_log_probs = self.evaluate(batch_obs_s, batch_acts_s)
                critic_loss = nn.MSELoss()(V, batch_rtgs_s)
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

                # print(actor_loss , "actor loss" , critic_loss , "critic loss")

            unpause()

    def rollout(self):
        global m1, m2, m3, m4, lidar_distance, global_y, global_x, roll, pitch, episode, closest_dist, distance_goal, name_data, end_distance, max_distance, highest_point, highest_x, highest_y, highest_pitch, highest_roll, episode_number
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_mask = []
        batch_rtgs = []
        batch_lens = []

        ep_rews = []
        t = 0

        while t < self.timesteps_per_batch:
            ep_rews = []  # rewards collected per episode
            # Reset the environment. sNote that obs is short for observation.
            obs = self.env.reset()

            done = False
            # episode_number = episode_count/self.max_timesteps_per_episode
            name_data = my_folder + str(episode_number)
            closest_dist = 100
            end_distance = 0
            max_distance = 0

            highest_point = 0
            highest_x = 0
            highest_y = 0
            highest_pitch = 0
            highest_roll = 0

            with open(name_data + '.csv', 'w') as csv_file:
               csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
               csv_writer.writeheader()
            # Run an episode for a maximum of max_timesteps_per_episode timesteps

            for ep_t in range(self.max_timesteps_per_episode):

                t += 1  # Increment timesteps ran this batch so far
                # print(t)
                # Track observations in this batch
                batch_obs.append(obs.copy())

                # Calculate action and make a step in the env.
                # Note that rew is short for reward.
                action, log_prob = self.get_action(obs)

                done, obs, rew = self.env.step(action, obs, self.max_timesteps_per_episode - ep_t)
                # print(obs)

                if highest_point < lidar_distance:
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

                if done:
                    pause()
                    break

            with open(name, 'a') as csv_file_2:
                csv_writer_2 = csv.DictWriter(csv_file_2, fieldnames=fieldnames2)
                information2 = {"episode reward": sum(ep_rews), "closest_dist": closest_dist,
                                "max_height": highest_point, "end_distance": end_distance, "max_distance": max_distance,
                                "highest_x": highest_x, "highest_y": highest_y, "highest_pitch": highest_pitch,
                                "highest_roll": highest_roll}
                csv_writer_2.writerow(information2)
            # PRINT TOTAL EPISODIC REWARD

            print(sum(ep_rews), "in episode", episode_number, "distance")  # sum will add all the rewards
            episode_number = episode_number + 1
            batch_lens.append(ep_t + 1)
            ep_dones = np.ones(len(ep_rews))
            ep_dones[-1] = 0
            batch_rews += ep_rews
            batch_mask += list(ep_dones)
            if (episode_number % 100 == 0):
                torch.save(self.actor.state_dict(), ac_weights_folder + 'ppo_actor' + str(episode_number) + '.pth')
                torch.save(self.critic.state_dict(), cr_weights_folder + 'ppo_critic' + str(episode_number) + '.pth')

        pause()
        batch_obs = torch.tensor(batch_obs, dtype=torch.float).to(self.actor.device)
        batch_acts = np.array(batch_acts)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float).to(self.actor.device)
        self.env.reset()
        V, _ = self.evaluate(batch_obs, batch_acts)
        batch_log_probs = np.array(batch_log_probs)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float).to(self.actor.device)
        batch_rtgs = self.compute_rtgs(batch_rews, V, batch_mask)
        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens, V

    def compute_rtgs(self, batch_rews, V, masks):
        batch_rtgs = []
        gae = 0
        #ret_mean = self.ret_stats.average
        #ret_std = self.ret_stats.std
        #V = (V * (ret_std + 1e-8)) + ret_mean
        values_list = list(V)
        values_list.append(torch.tensor(0.0))
        rewards_list = batch_rews
        # print(V, batch_rews, rewards_list)
        delta = 0
        # Iterate through each episode
        for i in reversed(range(len(rewards_list))):
            delta = rewards_list[i] + self.gamma * values_list[i + 1] * masks[i] - values_list[i]
            gae = delta + self.gamma * self.lam * masks[i] * gae
            batch_rtgs.insert(0, gae + values_list[i])

        # Convert the rewards-to-go into a tensor
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float).to(self.actor.device)
        # Normalizing the returns
        # mean, std = self.ret_stats.new_sample(batch_rtgs)
        # batch_rtgs = (batch_rtgs - mean) / (std + 1e-8)

        return batch_rtgs

    def get_action(self, obs, evaluation=False):
        global change
        # Query the actor network for a mean action
        out = self.actor(obs).to(self.actor.device)
        std = self.actor.log_std.exp().expand_as(out).to(self.actor.device)
        # print(std)
        dist = Normal(out, std)
        # dist = dist.to(self.actor.device)
        # Sample an action from the distribution
        if evaluation:
            action = out
        else:
            action = dist.sample()
        # print(change)
        # print(action)
        # Calculate the log probability for that action
        log_prob = torch.sum(dist.log_prob(action))
        log_prob = log_prob.cpu().detach().numpy()
        # print(obs)
        # print(log_prob)
        # Return the sampled action and the log probability of that action in our distribution
        return action.cpu().detach().numpy(), log_prob

    def evaluate(self, batch_obs, batch_acts):
        V = self.critic(batch_obs).squeeze()

        # Calculate the log probabilities of batch actions using most recent actor network.
        # This segment of code is similar to that in get_action()
        out = self.actor(batch_obs).to(self.actor.device)

        std = self.actor.log_std.exp().expand_as(out).to(self.actor.device)
        dist = Normal(out, std)
        log_probs = torch.sum(dist.log_prob(batch_acts), axis=1)
        # print(batch_obs)
        # print(log_probs ,"evaluate")
        # Return the value vector V of each observation in the batch
        # and log probabilities log_probs of each action in the batch
        return V, log_probs

    def _init_hyperparameters(self):
        # Initialize default values for hyperparameters
        # Algorithm hyperparameters

        self.max_timesteps_per_episode = 800  # Max number of timesteps per episode
        self.timesteps_per_batch = 4 * self.max_timesteps_per_episode  # Number of timesteps to run per batch
        self.n_updates_per_iteration = 5  # Number of times to update actor/critic per iteration BATCH SIZE

        self.gamma = 0.99  # Discount factor to be applied when calculating Rewards-To-Go
        self.lam = 0.99  # GAE lambda
        self.clip = 0.2  # Recommended 0.2, helps define the threshold to clip the ratio during SGA


def euler_from_quaternion(attitude):
    """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
    # t0 = +2.0 * (w * x + y * z)
    # t1 = +1.0 - 2.0 * (x * x + y * y)
    # roll_x = math.atan2(t0, t1)
    #
    # t2 = +2.0 * (w * y - z * x)
    # t2 = +1.0 if t2 > +1.0 else t2
    # t2 = -1.0 if t2 < -1.0 else t2
    # pitch_y = math.asin(t2)
    #
    # t3 = +2.0 * (w * z + x * y)
    # t4 = +1.0 - 2.0 * (y * y + z * z)
    # yaw_z = math.atan2(t3, t4)
    rotation = Rotation.from_quat(attitude)
    attitude = rotation.as_euler('xyz', degrees=True)
    roll_x = attitude[0]
    pitch_y = attitude[1]
    yaw_z = attitude[2]

    return roll_x, pitch_y, yaw_z


def service():
    global roll, pitch, yaw, lidar_distance, m1, m2, m3, m4, episode, closest_dist, distance_goal, name_data, end_distance, max_distance, highest_point, highest_x, highest_y, highest_pitch, highest_roll

    # print(lidar_distance,"lidar_distance")
    # velocity.data = [0, 0, 0, 0]
    # velPub.publish(velocity)
    # reset_world()
    closest_dist = 100
    end_distance = 0
    max_distance = 0

    rospy.sleep(3)

    model = PPO()
    model.learn(100000000)


def lasercall_back(msg):
    global roll, pitch, yaw, lidar_distance, m1, m2, m3, m4
    # print("step 2")
    distance = msg.ranges
    lidar_distance = min(distance)
    if (lidar_distance == inf or lidar_distance == -inf):
        lidar_distance = 0


def location_callback(msg):
    global lidar_distance, global_x, global_y, global_z, velocity, last_position, last_time, pose_velocity, att_velocity, attitude
    ind = msg.name.index('Kwad')
    # print("step 3")
    # print(ind)
    # velocityObj = msg.pose[ind].velocity
    orientationObj = msg.pose[ind].orientation
    positionObj = msg.pose[ind].position
    velocityObj = msg.twist[ind].linear
    angvelocityObj = msg.twist[ind].angular

    global_x = positionObj.x
    global_y = positionObj.y
    lidar_distance = positionObj.z

    pose_velocity = np.array([velocityObj.x, velocityObj.y, velocityObj.z])
    att_velocity = np.array([angvelocityObj.x, angvelocityObj.y, angvelocityObj.z])

    attitude = np.array([orientationObj.x, orientationObj.y, orientationObj.z, orientationObj.w])
    # roll, pitch, yaw = euler_from_quaternion(orientationObj.x, orientationObj.y, orientationObj.z, orientationObj.w)


def main():
    rospy.init_node('Drone_control_ppo')
    global last_time, pose_velocity
    last_time = None
    pose_velocity = np.array([0, 0, 0])
    # rospy.Subscriber("/Kwad/scan" , LaserScan , lasercall_back)
    rospy.Subscriber("/gazebo/model_states", ModelStates, location_callback)
    rospy.wait_for_service('/gazebo/reset_world')
    # print("kk")

    velocity.data = [20, -20, 20, -20]
    velPub.publish(velocity)

    thread.start_new_thread(service, ())
    # thread.start_new_thread(wind_,())

    rospy.spin()


if __name__ == main():
    main()
