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


info = namedtuple('info' ,
                  ('state' , 'action' , 'next_state' ,'reward')) # Defining the tuple for the saving the state , action , reward , nxt_state sequence



def take_action(self , action,cnt,sp_pub,rate , x = 0.1):
   	if action == 1:
		cnt.z_dir(x)
    	sp_pub.publish(cnt.sp)
    	rate.sleep()
	
   	if action == 2:
		cnt.z_dir(-x)
    	sp_pub.publish(cnt.sp)
    	rate.sleep()

    if action == 3:
		cnt.x_dir(x)
		sp_pub.publish(cnt.sp)
    	rate.sleep()

   	if action == 4:
		cnt.x_dir(-x)
		sp_pub.publish(cnt.sp)
    	rate.sleep()

   	if action == 5:
		cnt.y_dir(x)
		sp_pub.publish(cnt.sp)
    	rate.sleep()

   	if action == 6:
		cnt.y_dir(-x)
		sp_pub.publish(cnt.sp)
    	rate.sleep()
    		
   	if action == 7:
		cnt.x_ori(x)
    	sp_pub.publish(cnt.sp)
    	rate.sleep()

	if action == 8:
		cnt.x_ori(-x)
    	sp_pub.publish(cnt.sp)
    	rate.sleep()

   	if action == 9:
		cnt.y_ori(x)
    	sp_pub.publish(cnt.sp)
    	rate.sleep()

   	if action == 10:
		cnt.y_ori(-x)
    	sp_pub.publish(cnt.sp)
    	rate.sleep()

   	if action == 11:
		cnt.z_ori(x)
    	sp_pub.publish(cnt.sp)
    	rate.sleep()

   	if action == 12:
		cnt.z_ori(-x)
		sp_pub.publish(cnt.sp)
    	rate.sleep()

   	if action == 13:
		cnt.w_ori(x)
    	sp_pub.publish(cnt.sp)
    	rate.sleep()

   	if action == 14:
		cnt.w_ori(-x)
		sp_pub.publish(cnt.sp)
    	rate.sleep()
   	
    	



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
ep = 0
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
    		takeoffService(altitude = 1.0)
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
        self.ALT_SP = 1.0
        # update the setpoint message with the required altitude
        self.sp.pose.position.z = self.ALT_SP
        # Step size for position update
        self.STEP_SIZE = 2.0
		# Fence. We will assume a square fence for now
        self.FENCE_LIMIT = 5.0

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

    def x_ori(self,x):
    	self.sp.pose.orientation.x = self.sp.pose.orientation.x + (x)

    def y_ori(self,y):
    	self.sp.pose.orientation.y = self.sp.pose.orientation.y + (y)

    def z_ori(self,z):
    	self.sp.pose.orientation.z = self.sp.pose.orientation.z + (z)

    def w_ori(self,w):
    	self.sp.pose.orientation.w = self.sp.pose.orientation.w + (w)



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
    cnt_wind = Controller()

    # ROS loop rate
    rate_wind = rospy.Rate(20.0)

    # Subscribe to drone state
    rospy.Subscriber('mavros/state', State, cnt.stateCb)

    # Subscribe to drone's local position
    rospy.Subscriber('mavros/local_position/pose', PoseStamped, cnt.posCb)

    # Setpoint publisher    
    sp_pub_wind = rospy.Publisher('mavros/setpoint_position/local', PoseStamped, queue_size=1)


    while cnt_wind.posCb.local_pos_z > 0.2:
        r = random.random()
        if r > 0.5 :
            x = float(random.randint(3,6))
            dis = random.normalvariate(0.5,0.2)
            time = random.randint(3,7)
            for i in range(time):
                take_action(x,cnt_wind,sp_pub_wind,rate_wind,dis)
        else:
            x = float(random.randint(7,14))
            dis = random.normalvariate(0.3,0.1)
            time = random.randint(3,7)
            for i in range(time):
                take_action(x,cnt_wind,sp_pub_wind,rate_wind,dis)





episode =  0

def main_1():
    for i in range(100):
	#print(i,"episode")
    	main()
    t = random.randint(1, 12)
	rospy.sleep(t)
	#episode =+ 1 
		


if __name__ == '__main__':
	try:
		main_1()
	except rospy.ROSInterruptException:
		pass
