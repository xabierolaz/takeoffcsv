#!/usr/bin/env python
# ROS python API
import rospy

# 3D point & Stamped Pose msgs
from geometry_msgs.msg import Point, PoseStamped
# import all mavros messages and services
from mavros_msgs.msg import *
from mavros_msgs.srv import *
from collections import namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import random
import numpy as np
# Flight modes class
# Flight modes are activated using ROS services
from std_srvs.srv import Empty
import csv

info = namedtuple('info' ,
                  ('state' , 'action' , 'next_state' ,'reward')) # Defining the tuple for the saving the state , action , reward , nxt_state sequence

x0 = 0.0
x1 = 0.0
x2 = 0.0
x3 = 0.0
y0 = 0.0
y1 = 0.0
y2 = 0.0
y3 = 0.0
z0 = 0.0
z1 = 0.0
z2 = 0.0
z3 = 0.0
Ori_X0 = 0.0
Ori_X1 = 0.0
Ori_X2 = 0.0
Ori_X3 = 0.0
Ori_Y0 = 0.0
Ori_Y1 = 0.0
Ori_Y2 = 0.0
Ori_Y3 = 0.0
Ori_Z0 = 0.0
Ori_Z1 = 0.0
Ori_Z2 = 0.0
Ori_Z3 = 0.0
Ori_W0 = 0.0
Ori_W1 = 0.0
Ori_W2 = 0.0
Ori_W3 = 0.0




#defining replay memory and it's functions 

class replay_memory :
    def __init__(self , size):
        self.size = size
        self.memory = []
        self.position = 0

    def push(self, data ): # to push the data into the namedtuple
        if len(self.memory)< self.size :
            self.memory.append(data)
        else:
            self.memory[self.position  % self.size] = data

        self.position += 1

    def sample (self , batch_size): # random sample selected for training with batch size what we select  
        if len(self.memory) > batch_size:
            return random.sample(self.memory , batch_size )
        else:
            print("Batch size is bigger than the memory size")

    def can_provide_sample(self,batch_size): # to check if we can take the sample out from memory or not
        return len(self.memory) >  batch_size + 10

class epsilon_greedy : # exploration/exploitation selection   
    def __init__(self,start , end , decay):
        self.start = start
        self.end = end
        self.decay = decay

    def get_exploration_coe(self , current_step ):
        return self.end + (self.start - self.end)*math.exp(-1.* current_step* self.decay)


class DQN(nn.Module): # Network class 
    def __init__(self):
        super(DQN,self).__init__()

        self.fc1 = nn.Linear(in_features= 28 , out_features= 100) # 16 Number of inputs we are providing 4 set of height ,yaw ,pitch, roll
        self.fc2 = nn.Linear(in_features= 100 , out_features= 200)
        self.fc3 = nn.Linear(in_features= 200 , out_features= 200)
        self.fc4 = nn.Linear(in_features= 200 , out_features= 50)
        self.out = nn.Linear(in_features= 50, out_features= 8) # 6 outputs 

    def forward(self , state1): 
        
        state = state1.flatten() #flattening the tensor received to send it as a input 
        #print(len(state))
        a1 = F.sigmoid(self.fc1(state))
        a2 = F.sigmoid(self.fc2(a1))
        a3 = F.sigmoid(self.fc3(a2))
        a4 = F.sigmoid(self.fc4(a3))
        q_vals = self.out(a4) #returning  action-values 
        
        return q_vals

class Drone():
    def __init__(self,strategy,n_action,current_step):
        self.n_action = n_action
        self.current_step = current_step
        self.strategy = strategy

    
    def get_action(self,state,policy_net,cnt,sp_pub,rate):
    	global exploration_count , exploitation_count 
	#print("get action")
        
        r = self.strategy.get_exploration_coe(self.current_step) # selecting exploration and exploitation
        self.current_step += 1
        if r > random.random(): # exploration
            exploration_count += 1
            qv = float(random.randint(1, 8))

        else:
            exploitation_count += 1 # explotation
            qvs = policy_net.forward(state)
            qvs = qvs.detach().numpy()
            qv = np.argmax(qvs) + 1
            

        self.take_action(qv,cnt,sp_pub,rate)
        return qv


    def take_action(self , action,cnt,sp_pub,rate , x = 0.1): # taking action according to the DQN
        
	#print("take action")
    	if action == 1:
		cnt.z_dir(x)
	    	sp_pub.publish(cnt.sp)
	    	rate.sleep()
	
    	if action == 2:
		cnt.z_dir(-x)
	    	sp_pub.publish(cnt.sp)
	    	rate.sleep()
    		
    	if action == 3:
		cnt.yaw(x)
	    	sp_pub.publish(cnt.sp)
	    	rate.sleep()

	if action == 4:
		cnt.yaw(-x)
	    	sp_pub.publish(cnt.sp)
	    	rate.sleep()

    	if action == 5:
		cnt.pitch(x)
	    	sp_pub.publish(cnt.sp)
	    	rate.sleep()

    	if action == 6:
		cnt.pitch(-x)
	    	sp_pub.publish(cnt.sp)
	    	rate.sleep()

    	if action == 7:
		cnt.roll(x)
	    	sp_pub.publish(cnt.sp)
	    	rate.sleep()

    	if action == 8:
		cnt.roll(-x)
		sp_pub.publish(cnt.sp)
	    	rate.sleep()

    	if action == 9:
		cnt.x_dir(x)
		sp_pub.publish(cnt.sp)
	    	rate.sleep()

    	if action == 10:
		cnt.x_dir(-x)
		sp_pub.publish(cnt.sp)
	    	rate.sleep()

    	if action == 11:
		cnt.y_dir(x)
		sp_pub.publish(cnt.sp)
	    	rate.sleep()

    	if action == 12:
		cnt.y_dir(-x)
		sp_pub.publish(cnt.sp)
	    	rate.sleep()
    	

class Environment():
    def __init__(self):
        pass

    def get_state(self,cnt): # gives us state by storing that sensors data into initially defined variables
        global  x0 , x1 , x2 , x3 , y0 , y1 , y2 , y3 , z0 , z1 , z2 , z3 , Ori_W0 , Ori_W1 , Ori_W2 , Ori_W3 , Ori_X0 , Ori_X1 , Ori_X2 , Ori_X3 , Ori_Y0 , Ori_Y1 , Ori_Y2 , Ori_Y3 , Ori_Z0 , Ori_Z1 , Ori_Z2 , Ori_Z3  
	#print("get state")
        x0 = x1
        x1 = x2
        x2 = x3
        x3 = cnt.local_pos_x
        y0 = y1
        y1 = y2
        y2 = y3
        y3 = cnt.local_pos_y
        z0 = z1
        z1 = z2
        z2 = z3
        z3 = cnt.local_pos_z
	Ori_W0 = Ori_W1
	Ori_W1 = Ori_W2
	Ori_W2 = Ori_W3
	Ori_W3 = cnt.local_pos_ori_w 
        Ori_X0 = Ori_X1
	Ori_X1 = Ori_X2
	Ori_X2 = Ori_X3
	Ori_X3 = cnt.local_pos_ori_x 
        Ori_Y0 = Ori_Y1
	Ori_Y1 = Ori_Y2
	Ori_Y2 = Ori_Y3
	Ori_Y3 = cnt.local_pos_ori_y 
        Ori_Z0 = Ori_Z1
	Ori_Z1 = Ori_Z2
	Ori_Z2 = Ori_Z3
	Ori_Z3 = cnt.local_pos_ori_z 
        
        return torch.FloatTensor([[x0,y0,z0,Ori_X0,Ori_Y0,Ori_Z0,Ori_W0],[x1,y1,z1,Ori_X1,Ori_Y1,Ori_Z1,Ori_W1],[x2,y2,z2,Ori_X2,Ori_Y2,Ori_Z2,Ori_W2],[x3,y3,z3,Ori_X3,Ori_Y3,Ori_Z3,Ori_W3]])


    def get_reward(self,state): # reward for action
	current_x = float(state[3,0])	
	current_y = float(state[3,1])
	current_z = float(state[3,2])
	current_ori_x = float(state[3,3])
	current_ori_y = float(state[3,4])
	current_ori_z = float(state[3,5])
	current_ori_w = float(state[3,6])
	previous_x = float(state[2,0])	
	previous_y = float(state[2,1])
	previous_z = float(state[2,2])
	previous_ori_x = float(state[2,3])
	previous_ori_y = float(state[2,4])
	previous_ori_z = float(state[2,5])
	previous_ori_w = float(state[2,6])
	
	reward = 0.0
	if (previous_x - current_x) != 0:
		reward = reward - 0.5

	if (previous_y - current_y) != 0:
		reward = reward - 0.5

	if current_z < 6.0 :	
		if (previous_z - current_z) > 0:
			reward = reward + 0.5

	if current_z > 6.0 :
		reward = reward - 0.5

	if current_z > 5.8 and current_z < 6.2:
		reward = reward + 2.0

	if (previous_z - current_z) < 0:
		reward = reward - 0.9	 	
	
	if (previous_ori_x - current_ori_x) != 0:
		reward =  reward - (abs(previous_ori_x - current_ori_x))

	if (previous_ori_y - current_ori_y) != 0:
		reward = reward - (abs(previous_ori_y - current_ori_y))
	
	if (previous_ori_z - current_ori_z) != 0:
		reward = reward - (abs(previous_ori_z - current_ori_z))

	if (previous_ori_w - current_ori_w) != 0:
		reward = reward - (abs(previous_ori_w - current_ori_w))

    	return reward

    def random_wind(self,drone,cnt,sp_pub,rate): # random wind generaion
	qv = float(random.randint(3, 12))
	#print("random action")
	drone.take_action(qv,cnt,sp_pub,rate, x=0.3)	        


class Qval:

    @staticmethod
    def get_current(policy_network,states,actions):
        return policy_network.forward(states)

    @staticmethod
    def get_next(target_network,next_states):
        return target_network.forward(next_states)



def extract_tensors(experiences):
    batch = info(*zip(*experiences))
    
    t1 = torch.cat(batch.state)
    
    t2 = torch.cat(batch.action)
    
    t3 = torch.cat(batch.reward)
    
    t4 = torch.cat(batch.next_state)
    

    return ( t1, t2, t3, t4)


'''

some standard parameters which are required for the DQN 
start , end , decay us needed for greedy approach
env object instanciated from environment class
other adjustable parameters 
instanciating both the networks policy as well as target
Adam optimiser selected

'''

start = 1
end = 0.1
decay = 0.003
env = Environment()
num_episodes = 10
num_steps = 50

lr = 0.025
gamma = 0.9
batch_size = 5
memory_size = 100000
policy_network = torch.load('/home/alumno/Escritorio/NN_weights/Netwrok_weights.pth')
target_network = DQN()
target_network.load_state_dict(policy_network.state_dict())
target_network.eval()
optimiser = optim.Adam(params=policy_network.parameters(), lr = lr)
memory = replay_memory(size=memory_size)
exploitation_count = 0
exploration_count = 0
















class fcuModes:
    def __init__(self):
        pass

    def setTakeoff(self):
    	rospy.wait_for_service('mavros/cmd/takeoff')
    	try:
    		takeoffService = rospy.ServiceProxy('mavros/cmd/takeoff', mavros_msgs.srv.CommandTOL)
    		takeoffService(altitude = 2.0)
    	except rospy.ServiceException, e:
    		print "Service takeoff call failed: %s"%e

    def setArm(self):
        rospy.wait_for_service('mavros/cmd/arming')
        try:
            armService = rospy.ServiceProxy('mavros/cmd/arming', mavros_msgs.srv.CommandBool)
            armService(True)
        except rospy.ServiceException, e:
            print "Service arming call failed: %s"%e

    def setDisarm(self):
        rospy.wait_for_service('mavros/cmd/arming')
        try:
            armService = rospy.ServiceProxy('mavros/cmd/arming', mavros_msgs.srv.CommandBool)
            armService(False)
        except rospy.ServiceException, e:
            print "Service disarming call failed: %s"%e

    def setStabilizedMode(self):
        rospy.wait_for_service('mavros/set_mode')
        try:
            flightModeService = rospy.ServiceProxy('mavros/set_mode', mavros_msgs.srv.SetMode)
            flightModeService(custom_mode='STABILIZED')
        except rospy.ServiceException, e:
            print "service set_mode call failed: %s. Stabilized Mode could not be set."%e

    def setOffboardMode(self):
        rospy.wait_for_service('mavros/set_mode')
        try:
            flightModeService = rospy.ServiceProxy('mavros/set_mode', mavros_msgs.srv.SetMode)
            flightModeService(custom_mode='OFFBOARD')
        except rospy.ServiceException, e:
            print "service set_mode call failed: %s. Offboard Mode could not be set."%e

    def setAltitudeMode(self):
        rospy.wait_for_service('mavros/set_mode')
        try:
            flightModeService = rospy.ServiceProxy('mavros/set_mode', mavros_msgs.srv.SetMode)
            flightModeService(custom_mode='ALTCTL')
        except rospy.ServiceException, e:
            print "service set_mode call failed: %s. Altitude Mode could not be set."%e

    def setPositionMode(self):
        rospy.wait_for_service('mavros/set_mode')
        try:
            flightModeService = rospy.ServiceProxy('mavros/set_mode', mavros_msgs.srv.SetMode)
            flightModeService(custom_mode='POSCTL')
        except rospy.ServiceException, e:
            print "service set_mode call failed: %s. Position Mode could not be set."%e

    def setAutoLandMode(self):
        rospy.wait_for_service('mavros/set_mode')
        try:
            flightModeService = rospy.ServiceProxy('mavros/set_mode', mavros_msgs.srv.SetMode)
            flightModeService(custom_mode='AUTO.LAND')
        except rospy.ServiceException, e:
               print "service set_mode call failed: %s. Autoland Mode could not be set."%e

class Controller:
    # initialization method
    def __init__(self):
        # Drone state
        self.state = State()
        # Instantiate a setpoints message
        self.sp = PoseStamped()
        # set the flag to use position setpoints and yaw angle
        #self.sp.type_mask = int('010111111000', 2)
        # LOCAL_NED
        #self.sp.coordinate_frame = 1

        # We will fly at a fixed altitude for now
        # Altitude setpoint, [meters]
        self.ALT_SP = 2.0
        # update the setpoint message with the required altitude
        self.sp.pose.position.z = self.ALT_SP
        # Step size for position update
        self.STEP_SIZE = 2.0
		# Fence. We will assume a square fence for now
        self.FENCE_LIMIT = 10.0

        # A Message for the current local position of the drone
        self.local_pos_x = 0.0
	self.local_pos_y = 0.0
	self.local_pos_z = 0.0
	self.local_pos_ori_x = 0.0
	self.local_pos_ori_y = 0.0
	self.local_pos_ori_z = 0.0
	self.local_pos_ori_w = 0.0
        # initial values for setpoints
        self.sp.pose.position.x = 0.0
        self.sp.pose.position.y = 0.0

        # speed of the drone is set using MPC_XY_CRUISE parameter in MAVLink
        # using QGroundControl. By default it is 5 m/s.

	# Callbacks

    ## local position callback
    def posCb(self, msg):
        self.local_pos_x = msg.pose.position.x
        self.local_pos_y = msg.pose.position.y
        self.local_pos_z = msg.pose.position.z
        self.local_pos_ori_x = msg.pose.orientation.x
        self.local_pos_ori_y = msg.pose.orientation.y
        self.local_pos_ori_z = msg.pose.orientation.z
        self.local_pos_ori_w = msg.pose.orientation.w

    ## Drone State callback
    def stateCb(self, msg):
        self.state = msg

    ## Update setpoint message
    def z_dir(self,z):
        self.sp.pose.position.z = self.local_pos_z + (z) 
        self.sp.pose.position.y = self.local_pos_y

    def yaw(self,x):
    	self.sp.pose.orientation.x = self.sp.pose.orientation.x + (x)

    def pitch(self,y):
    	self.sp.pose.orientation.y = self.sp.pose.orientation.y + (y)

    def roll(self,z):
    	self.sp.pose.orientation.z = self.sp.pose.orientation.z + (z)



    def x_dir(self,x):
    	self.sp.pose.position.x = self.local_pos_x + (x)
    	self.sp.pose.position.y = self.local_pos_y

    
    def y_dir(self,y):
    	self.sp.pose.position.x = self.local_pos_x
    	self.sp.pose.position.y = self.local_pos_y + (y)

    

# Main function
def main():
    global episode	
    # initiate node
    rospy.init_node('setpoint_node', anonymous=True)

    # flight mode object
    modes = fcuModes()

    # controller object
    cnt = Controller()

    # ROS loop rate
    rate = rospy.Rate(20.0)

    # Subscribe to drone state
    rospy.Subscriber('mavros/state', State, cnt.stateCb)

    # Subscribe to drone's local position
    rospy.Subscriber('mavros/local_position/pose', PoseStamped, cnt.posCb)

    # Setpoint publisher    
    sp_pub = rospy.Publisher('mavros/setpoint_position/local', PoseStamped, queue_size=1)


    # Make sure the drone is armed
    while not cnt.state.armed:
        modes.setArm()

        rate.sleep()

    # set in takeoff mode and takeoff to default altitude (3 m)
    # modes.setTakeoff()
    # rate.sleep()

    # We need to send few setpoint messages, then activate OFFBOARD mode, to take effect
    k=0
    while k<500:
        sp_pub.publish(cnt.sp)
        rate.sleep()
	#print(k)
        k = k + 1

    # activate OFFBOARD mode

   
    
    modes.setOffboardMode()
    fieldnames = ['Height','Orientation_in_x','Orientation_in_y','Orientation_in_x','Orientation_in_z','Orientation_in_w'] 
    Orientation_in_z = 0.0
    Orientation_in_y = 0.0
    Orientation_in_x = 0.0
    Orientation_in_w = 0.0
    Height = 0.0 
    with open('data'+ str(episode)+'.csv','w') as csv_file:
        csv_writer = csv.DictWriter(csv_file , fieldnames = fieldnames)
 
    rospy.sleep(1)
    if cnt.local_pos_z < 0.5 :
	print("Could not took off")	

    else:
	l = 0
        drone = Drone(n_action = 8,strategy = epsilon_greedy(start, end, decay),current_step = 0) #object of drone is created for each episode
        state = env.get_state(cnt)#torch.FloatTensor([[0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0]]) # initial state 	
        # ROS main loop
        #print(drone)
        while l<1000:
	    action = drone.get_action(state,policy_network,cnt,sp_pub,rate) # select the action going up or down etc..
	    #print(l,"steps")
	    reward = env.get_reward(state) # reward for each action
	    env.random_wind(drone,cnt,sp_pub,rate) # Introduction to random wind
            
            nxt_state = env.get_state(cnt)# get the new state
            memory.push(info(state,torch.FloatTensor([action]),nxt_state,torch.FloatTensor([reward]))) # store the tuple in replay memory buffer
            state = nxt_state
	    
	    if cnt.local_pos_z < 0.5 :
		print("may flip")
		break
	    with open ('data' +str(episode)+ '.csv' , 'a') as csv_file:
		csv_writer = csv.DictWriter(csv_file , fieldnames = fieldnames)

		information = {"Height" : cnt.sp.pose.position.z , "Orientation_in_x" : cnt.sp.pose.orientation.x , "Orientation_in_y" : cnt.sp.pose.orientation.y , "Orientation_in_z" : cnt.sp.pose.orientation.z , "Orientation_in_w" : cnt.sp.pose.orientation.w}
 		csv_writer.writerow(information)
            if memory.can_provide_sample(batch_size):
                experiences = memory.sample(batch_size)
	        #print(len(experiences))
                states , actions , rewards ,next_states = extract_tensors(experiences) # start of the training extracting the tensores from the memory buffer.
	        #print((states))
                for i in range(batch_size):
                    #print("batch")    
                    state1 = torch.FloatTensor([states[4*i+0].tolist(),states[4*i+1].tolist(),states[4*i+2].tolist(),states[4*i+3].tolist()]) # arrange the states in tensorfloat format which is the requirement of DQN 
                    next_state1 = torch.FloatTensor([next_states[4*i + 0].tolist(), next_states[4*i + 1].tolist(), next_states[4*i + 2].tolist(),next_states[4*i + 3].tolist()]) # arrange the states in tensorfloat format which is the requirement of DQN 

                    #print(i,"i value",state1)    
                    cqvals = Qval.get_current(policy_network,state1,actions[i]) # action-value pair for this state

                    
                    index = int(actions[i].item())-1 
                    
                    nxt_qvals = Qval.get_next(target_network,next_state1) # next state action-value pair from target network
                    sqvals = ((nxt_qvals)*gamma) 
                    sqvals[index] = sqvals[index] + rewards[i] # adding the reward to action taken
                  
                    loss  = nn.MSELoss()
                    loss = loss(sqvals , cqvals) #calculating the loss occured
                    
                    optimiser.zero_grad() # optimisation funciton inside the pytorch 
                    loss.backward()
                    optimiser.step()

            if l == 100:
                target_network.load_state_dict(policy_network.state_dict()) # updating the target network 
	        torch.save(target_network,'/home/alumno/Escritorio/NN_weights/Netwrok_weights.pth')              
          
    	    rate.sleep()
	    l = l+1
	    if cnt.local_pos_z > 6.0 :
		print("reached max height")
		break
        torch.save(target_network,'/home/alumno/Escritorio/NN_weights/Netwrok_weights' +str(episode)+'.pth')	

        cnt_reset = Controller()
        cnt_reset.sp.pose.position.x = 0
        cnt_reset.sp.pose.position.y = 0
        cnt_reset.sp.pose.position.z = 0
        cnt_reset.sp.pose.orientation.x = 0
        cnt_reset.sp.pose.orientation.y = 0
        cnt_reset.sp.pose.orientation.z = 0
        cnt_reset.sp.pose.orientation.w = 0
        sp_pub_reset = rospy.Publisher('mavros/setpoint_position/local', PoseStamped, queue_size=1)
        rate_reset = rospy.Rate(20.0)
        reset = 0	
        while reset <20 :
	    #print("reset")
	    sp_pub_reset.publish(cnt_reset.sp)
	    rate_reset.sleep()
	    reset = reset +1
	


episode =  0

def main_1():
    for i in range(100):
	global episode
	episode = i 
	print(i,"episode")
    	main()
	rospy.sleep(10)
		


if __name__ == '__main__':
	try:
		main_1()
	except rospy.ROSInterruptException:
		pass