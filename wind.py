import rospy
from geometry_msgs.msg import Pose
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState
from gazebo_msgs.msg import ModelStates
import random
import thread

rospy.wait_for_service("/gazebo/set_model_state")
m = rospy.ServiceProxy("/gazebo/set_model_state",SetModelState)
command = ModelState()
command.model_name = "Kwad"

global  X , Y , Z , ORI_X , ORI_Y , ORI_Z  , ORI_W


def location_callback(msg):
    global X , Y , Z , ORI_X , ORI_Y , ORI_Z  , ORI_W
    ind = msg.name.index('Kwad')
    #print(ind ,"ds")
    orientationObj = msg.pose[ind].orientation
    positionObj = msg.pose[ind].position
    X = positionObj.x
    Y = positionObj.y
    Z = positionObj.z
    ORI_X = orientationObj.x
    ORI_Y = orientationObj.y
    ORI_Z = orientationObj.z
    ORI_W = orientationObj.w
    service(X , Y , Z , ORI_X , ORI_Y , ORI_Z  , ORI_W)


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


def service(X , Y , Z , ORI_X , ORI_Y , ORI_Z  , ORI_W):
    location = Pose()
    location.position.x = X
    location.position.y = Y
    location.position.z = Z
    location.orientation.x = ORI_X
    location.orientation.y = ORI_Y
    location.orientation.z = ORI_Z
    location.orientation.w = ORI_W

    r = random.random()
    if r > 0.5 :
        action = float(random.randint(0,3))
        dis = random.normalvariate(0.0005,0.0002)
        time = random.randint(130,140)
        for i in range(time):
            if action == 0:
                print("In +x direction by " , dis)
                location.position.x += dis
                
            elif action == 1:
                print("In -x direction by " , dis)
                location.position.x -= dis

            elif action == 2:
                print("In +y direction by " , dis)
                location.position.y += dis

            elif action == 3:
                print("In -y direction by " , dis)
                location.position.y -= dis
                
    else:
        action = float(random.randint(4,11))
        dis = random.normalvariate(0.0003,0.0001)
        time = random.randint(800,90)
        for i in range(time):
            if action == 4:
                print(" x Orientation increase by " , dis)
                location.orientation.x += dis

            elif action == 5:
                print(" x Orientation decrease by "  , dis)
                location.orientation.x -= dis
    
            elif action == 6:
                print(" y Orientation increase by " , dis)
                location.orientation.y += dis

            elif action == 7:
                print(" y Orientation decrease by " , dis)
                location.orientation.y -= dis
    
            elif action == 8:
                print(" z Orientation increase by " , dis)
                location.orientation.z += dis

            elif action == 9:
                print(" z Orientation decrease by " , dis)
                location.orientation.z -= dis
    
            elif action == 10:
                print(" w Orientation increase by " , dis)
                location.orientation.w += dis

            elif action == 11:
                print(" w Orientation decrease by " , dis)
                location.orientation.w -= dis
       
    
    command.pose = location
    m(command)
    rospy.sleep(10)


def main():
    rospy.init_node('wind_control')
    rospy.Subscriber("/gazebo/model_states" , ModelStates , location_callback)    
    
    rospy.spin()


if __name__ == main():
    main()