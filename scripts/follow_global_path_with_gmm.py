#!/usr/bin/env python3
# coding: utf-8
# 
# A contoller used in the simulation environment of Turtlebot 3. 

# Imports

# Basics
from importlib.resources import path
from random import random
import threading
import rospy
import time
import math
import numpy as np
from sklearn import mixture
from functools import partial
import matplotlib as mpl

# ROS libraries
from std_msgs.msg import String
from std_msgs.msg import MultiArrayDimension
from std_msgs.msg import (Float32MultiArray, Float64MultiArray,
                          Int8MultiArray, Int16MultiArray,
                          Int32MultiArray, Int64MultiArray,
                          UInt8MultiArray, UInt16MultiArray,
                          UInt32MultiArray, UInt64MultiArray)

from geometry_msgs.msg import (PoseArray, Pose, Point, PoseWithCovarianceStamped, Twist)
from nav_msgs.msg import (Odometry, Path)


# Global variables

# ROS parameters
# gmm_flag = rospy.get_param('gmm')
dwa_random_param = 100


# Storaged data

gmm_mean = None
gmm_covariance = None
gmm_weight = None

MSE_array = None

t_interval = 0.1

odom = Odometry()
error_msg = Point()

stop_flag = 0

start_time = None

mse_list = []
mse_calculation = 0

# Methods

# Conversion between angles and quaternions
def quaternion_to_rad(z, w): 
    if w == 0:   
        rad = np.pi
    else: 
        rad = 2 * np.arctan(z / w) - np.pi / 2

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


# Data type conversion
def _numpy_to_multiarray(multiarray_type, np_array):
    multiarray = multiarray_type()
    multiarray.layout.dim = [MultiArrayDimension('dim%d' % i,
                                                 np_array.shape[i],
                                                 np_array.shape[i] * np_array.dtype.itemsize) for i in range(np_array.ndim)]
    multiarray.data = np_array.reshape([1, -1])[0].tolist()
    return multiarray


def _multiarray_to_numpy(pytype, dtype, multiarray):
    dims = list(map(lambda x: x.size, multiarray.layout.dim))
    # print(multiarray.data)
    # print(pytype)
    # print(multiarray.layout.dim)
    # print(dims)
    # print(dtype)
    return np.array(multiarray.data, dtype=pytype).reshape(dims).astype(dtype)


to_multiarray_f64 = partial(_numpy_to_multiarray, Float64MultiArray)


to_numpy_f64 = partial(_multiarray_to_numpy, float, np.float64)


# Controller method
def control_with_gmm(): 

    global odom, gmm_mean, gmm_covariance, gmm_weight
    # Initialize command publisher
    pubCmd = rospy.Publisher('cmd_vel', Twist, queue_size=10)
    pubError = rospy.Publisher('error', Point, queue_size=10)

    original_v = odom.twist.twist.linear

    original_x = odom.pose.pose.position.x
    original_y = odom.pose.pose.position.y

    original_z = odom.pose.pose.orientation.z
    original_w = odom.pose.pose.orientation.w

    original_angular = odom.twist.twist.angular.z
    
    for i in range(dwa_random_param): 
        rand1 = random()
        rand2 = random()

        v = original_v + 0.10 * rand1
        
        if v <= 0.05: 
            v = 0.05
        
        if v >= 0.25: 
            v = 0.25

        a = original_angular + 1.0 * rand2

        x = original_x + t_interval * v * np.cos(0)
        







    
def control_with_no_gmm(): 
    pubCmd = rospy.Publisher('cmd_vel', Twist, queue_size=10) 

    global error_msg, odom

    cmd_vel_msg = Twist()

    cmd_vel_msg.linear.x = 0.05
    
    pubCmd.publish(cmd_vel_msg)





# def print_path(): 
#     global planned_path

#     rospy.loginfo(str(len(planned_path)))

#     for i, pose in enumerate(planned_path): 
#         rospy.loginfo('Point ' + str(i + 1))
#         rospy.loginfo('x = ' + str(pose.pose.position.x))
#         rospy.loginfo('y = ' + str(pose.pose.position.y))


# Callback methods

def callback_gmm_mean(data):
    global gmm_mean
    gmm_mean = to_numpy_f64(data)


def callback_gmm_covar(data):
    global gmm_covariance
    gmm_covariance = to_numpy_f64(data)


def callback_gmm_weight(data): 
    global gmm_weight
    gmm_weight = to_numpy_f64(data)


def callback_amcl_pose(data):
    global amcl_pose
    amcl_pose = data; 


def callback_odom(data): 
    global odom
    odom = data


# What shall we do here? 
def callback_path(data): 
    global planned_path
    planned_path = data.poses


def control(): 
    r = rospy.Rate(10)

    # while not rospy.is_shutdown(): 
    #     if gmm_mean is None or gmm_covariance is None or gmm_weight is None: 
    #         control_with_no_info()
    #     else: 
    #         control_with_gmm(gmm_mean, gmm_covariance, gmm_weight, amcl_pose, odom)
    #     r.sleep()

    while not rospy.is_shutdown(): 
        # rospy.loginfo('running... ')

        if planned_path is None: 
            rospy.loginfo('Waiting for path...')
        else: 
            if gmm_mean is None or gmm_covariance is None or gmm_weight is None: 
                control_with_no_gmm()
            else: 
                control_with_gmm() 
        r.sleep()



# Main method

if __name__ == '__main__':

    # rospy.init_node('state_feedback', anonymous=True)
    # Was trying to run controlling at a certain rate, but maybe not here
    # rate = rospy.Rate(5) # ROS Rate at 5Hz

    rospy.init_node('gmm_controller', anonymous=True)

    random.seed()
    
    # sub_mean = rospy.Subscriber('gmm_mean', Float64MultiArray, callback_gmm_mean)
    # sub_cov = rospy.Subscriber('gmm_covar', Float64MultiArray, callback_gmm_covar)
    # sub_weight = rospy.Subscriber('gmm_weight', Float64MultiArray, callback_gmm_weight)

    # sub_amcl_pose = rospy.Subscriber('amcl_pose', PoseWithCovarianceStamped, callback_amcl_pose)
    sub_odom = rospy.Subscriber('odom', Odometry, callback_odom)

    sub_path = rospy.Subscriber('move_base/NavfnROS/plan', Path, callback_path)

    control_thread = threading.Thread(target=control)
    control_thread.start()

    # start_time = rospy.get_time()

    rospy.spin()