# An RMPC controller for Turtlebot3. 

# Definition: GMM-and-RMPC-based controller for getting the "optimal" control output to let the robot move to the goal. 
#
# Date of programming: 2022/10/11 ~ 20xx/xx/xx
#
# Current progress: N
# A (working fine with a solid theoretical base) / B (seems to be working fine) / C (working with problems or sometimes working)
# F (totally not working) / N (not completed)

# Libraries

# - Basics

import random
import threading
import rospy
import time
import math
import numpy as np
from numpy import linalg
from sklearn import mixture
from functools import partial
import matplotlib as mpl

# - ROS libraries
from std_msgs.msg import String
from std_msgs.msg import MultiArrayDimension
from std_msgs.msg import (Float32MultiArray, Float64MultiArray,
                          Int8MultiArray, Int16MultiArray,
                          Int32MultiArray, Int64MultiArray,
                          UInt8MultiArray, UInt16MultiArray,
                          UInt32MultiArray, UInt64MultiArray)

from geometry_msgs.msg import (PoseArray, Pose, Point, PoseWithCovarianceStamped, Twist, PoseStamped)
from nav_msgs.msg import (Odometry, Path)


# Global variables

# ROS parameters
# gmm_flag = rospy.get_param('gmm')
# dwa_random_param = 300

dwa_horizon_param = 10

# Storaged data

amcl_pose = None

# GMM parameters

gmm_mean = None
gmm_covariance = None
gmm_weight = None

MSE_array = None

planned_path = None

t_interval = 0.1

odom = Odometry()
# error_msg = Point()

initial_rotation_finish = False

path_following_finish = False

final_rotation_finish = False

stop_flag = False

start_time = None

# mse_list = []
# mse_calculation = 0

goal_x = 0.0
goal_y = 0.0
goal_z = 0.0
goal_w = 0.0

goal_heading = 0.0


# Methods

# Conversion from original atan2 to the angle system we are using

def atan2_customized(y, x): 
    rad = math.atan2(y, x) - np.pi / 2.0
    
    if rad < -np.pi: 
        rad += 2.0 * np.pi

    return rad


# Conversion between angles(radients) and quaternions
def quaternion_to_rad(z, w): 
    if w == 0.0:   
        rad = np.pi
    else: 
        rad = 2.0 * np.arctan(z / w) - np.pi / 2.0

    if rad < -np.pi: 
        rad = rad + 2.0 * np.pi

    return rad


def rad_to_quaternion(rad): 
    # Only in a 2-d context
    rad += np.pi / 2

    rad /= 2

    quaternion = [0, 0, 0, 0]

    quaternion[0] = 0
    quaternion[1] = 0

    quaternion[2] = np.sin(rad)

    quaternion[3] = np.cos(rad)
        
    return quaternion


def linear_distance(x1, x2, y1, y2): 
    d = np.sqrt(np.power(x1 - x2, 2) + np.power(y1 - y2, 2))

    return d