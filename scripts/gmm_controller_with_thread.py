#!/usr/bin/env python3
# coding: utf-8
# Software License Agreement (BSD License)
#
# Copyright (c) 2008, Willow Garage, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of Willow Garage, Inc. nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

# A contoller used in the simulation environment of Turtlebot 3. 


# Imports

# Basics
import threading
import rospy
# import time
import math
import numpy as np
# from numpy.core.fromnumeric import mean
# from numpy.ma.core import concatenate
from numpy import linalg
from sklearn import mixture
from functools import partial
# import matplotlib as mpl
# import matplotlib.pyplot as plt

# ROS libraries
from std_msgs.msg import String
from std_msgs.msg import MultiArrayDimension
from std_msgs.msg import (Float32MultiArray, Float64MultiArray,
                          Int8MultiArray, Int16MultiArray,
                          Int32MultiArray, Int64MultiArray,
                          UInt8MultiArray, UInt16MultiArray,
                          UInt32MultiArray, UInt64MultiArray)

from geometry_msgs.msg import (PoseArray, Pose, Point, PoseWithCovarianceStamped, Twist)
from nav_msgs.msg import Odometry


# Global variables

# ROS Parameters
gmm_flag = rospy.get_param('gmm')

orient_x = rospy.get_param('orient_x')
orient_y = rospy.get_param('orient_y')
goal_x = rospy.get_param('goal_x')
goal_y = rospy.get_param('goal_y')

k = (goal_y - orient_y) / (goal_x - orient_x)
b = orient_y - k * orient_x

orient_ref = -np.arctan2(goal_x - orient_x, goal_y - orient_y)

# Feedback gains of the state feedback controller
# k1 = -1.0
k2 = -0.50
k3 = -0.1
k4 = -0.1

gmm_mean = None
gmm_covariance = None
gmm_weight = None
odom = Odometry()
error_msg = Point()

stop_flag = 0

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


# Pose error calculation
def get_err_position(x, y):
    d = (y - k * x - b) / math.sqrt(1 + math.pow(k, 2))
    
    # rospy.loginfo('linear_error = ' + str(d))
    
    return d

def get_err_orient(theta):
    # theta = 2 * np.arctan(z / w) - np.pi / 2
    
    err = theta - orient_ref

    if err < -np.pi: 
        err = err + 2.0 * np.pi
    if err > np.pi: 
        err = err - 2.0 * np.pi

    # rospy.loginfo('orientation_error = ' + str(err))

    return err

# Controller method
def control_with_gmm(means, covariances, weights, amcl_pose, new_amcl_pose, odom): 

    pubCmd = rospy.Publisher('cmd_vel', Twist, queue_size=10)
    pubError = rospy.Publisher('error', Point, queue_size=10)

    global error_msg

    linear_cmd = 0.0
    angular_cmd = 0.0

    orient_z = new_amcl_pose.pose.pose.orientation.z
    orient_w = new_amcl_pose.pose.pose.orientation.w

    theta = quaternion_to_rad(orient_z, orient_w)

    orientation_err = get_err_orient(theta)

    xForError = amcl_pose.pose.pose.position.x
    yForError = amcl_pose.pose.pose.position.y

    # xForError = new_amcl_pose.pose.pose.position.x
    # yForError = new_amcl_pose.pose.pose.position.y

    error_msg.x = get_err_position(xForError, yForError)
    error_msg.y = orientation_err

    # rospy.loginfo('linear error: ' + str(error_msg.x))
    # rospy.loginfo('angular error: ' + str(error_msg.y))


    if gmm_flag: 
        # Path following using GMM
        for i, (m, covar, weight) in enumerate(zip(means, covariances, weights)):
            
            eig_val, eig_vec = linalg.eigh(covar)

            v = 2.0 * np.sqrt(5.991) * np.sqrt(eig_val)
            u = eig_vec[0] / linalg.norm(eig_vec[0])
            

            angle = np.arctan(u[1] / u[0]) + 3 * np.pi / 2

            if angle > 2 * np.pi: 
                angle = angle - 2 * np.pi

            x = m[0]
            y = m[1]

            b = v[0]
            a = v[1]

            x_err = get_err_position(x, y)

            # Longitudinal direction
            l1 = np.abs(b / 2 * np.cos(angle - orient_ref)) + np.abs(a / 2 * np.sin(angle - orient_ref))

            # Lateral direction
            l2 = np.abs(b / 2 * np.sin(angle - orient_ref)) + np.abs(a / 2 * np.cos(angle - orient_ref))

            if ((-np.sin(theta) * (-np.sin(orient_ref)) + np.cos(theta) * np.cos(orient_ref)) 
                    * (goal_x - orient_x)) >= 0: 
                k1 = -0.50
            else: 
                k1 = 0.50

            k1 = k1 * np.abs(np.cos(orientation_err))
            
            cmd3 = k3 * l1

            if np.abs(cmd3) > 0.05: 
                cmd3 = cmd3 * 0.05 / np.abs(cmd3)

            cmd4 = k4 * l2

            if np.abs(cmd4) > 0.05: 
                cmd4 = cmd4 * 0.05 / np.abs(cmd4)

            angular_cmd += weight * (k1 * x_err + k2 * orientation_err) 

            linear_cmd += weight * (cmd3 + cmd4)
            
        rospy.loginfo('controlling with gmm')
        
    else: 
        # Path following without GMM

        x = new_amcl_pose.pose.pose.position.x
        y = new_amcl_pose.pose.pose.position.y
        x_err = get_err_position(x, y)

        if ((-np.sin(theta) * (-np.sin(orient_ref)) + np.cos(theta) * np.cos(orient_ref)) 
                * (goal_x - orient_x)) >= 0: 
            k1 = -0.50
        else: 
            k1 = 0.50

        k1 = k1 * np.abs(np.cos(orientation_err))

        angular_cmd = k1 * x_err + k2 * orientation_err
        # angular_cmd = k1 * x_err + k2 * orientation_err + 0.2 * (np.random.random(1) - 0.5)

        linear_cmd = 0.0

        rospy.loginfo('controlling w/o gmm')

    dist_goal = np.sqrt(np.power(x - goal_x, 2) + np.power(y - goal_y, 2))

    # rospy.loginfo('linear command: ' + str(linear_cmd))
    # rospy.loginfo('angular command: ' + str(angular_cmd))

    # rospy.loginfo('x coordinate: ' + str(x) + '; y coordinate: ' + str(y))

    cmd_vel_msg = Twist()

    cmd_vel_msg.linear.x = 0.20 + linear_cmd
    # cmd_vel_msg.linear.x = 0.20 + linear_cmd + 0.1 * (np.random.random(1) - 0.5)
        
    cmd_vel_msg.linear.y = 0.0    
    cmd_vel_msg.linear.z = 0.0

    cmd_vel_msg.angular.x = 0.0
    cmd_vel_msg.angular.y = 0.0
    cmd_vel_msg.angular.z = angular_cmd

    global stop_flag

    if (goal_y - orient_y) * (y - goal_y) > 0:
        stop_flag = 1
    elif (goal_y - orient_y) == 0: 
        if (goal_x - orient_x) * (x - goal_x) > 0:
            stop_flag = 1

    if stop_flag == 1: 
        cmd_vel_msg.linear.x = 0.0
        # cmd_vel_msg.angular.z = 0.0
    
    pubCmd.publish(cmd_vel_msg)
    pubError.publish(error_msg)

    
def control_with_no_info(): 

    pubCmd = rospy.Publisher('cmd_vel', Twist, queue_size=10) 

    pubError = rospy.Publisher('error', Point, queue_size=10)

    global error_msg

    error_msg.x = 0.0
    error_msg.y = 0.0

    cmd_vel_msg = Twist()

    cmd_vel_msg.linear.x = 0.20
    
    pubCmd.publish(cmd_vel_msg)

    pubError.publish(error_msg)


# Callback methods

def callback_gmm_mean(data):
    global gmm_mean
    rospy.loginfo('received gmm mean')
    gmm_mean = to_numpy_f64(data)

def callback_gmm_covar(data):
    global gmm_covariance
    rospy.loginfo('received gmm covariance')
    gmm_covariance = to_numpy_f64(data)

def callback_gmm_weight(data): 
    global gmm_weight
    rospy.loginfo('received gmm weight')
    gmm_weight = to_numpy_f64(data)

def callback_amcl_pose(data):
    repub_amcl_pose = rospy.Publisher('new_amcl_pose', PoseWithCovarianceStamped, queue_size=10)
    global amcl_pose, new_amcl_pose
    rospy.loginfo('received amcl_pose')
    amcl_pose = data; 

    new_amcl_pose = amcl_pose

    r = 0.2 * np.random.randn()

    direction = 2 * np.pi * np.random.random(1)

    new_amcl_pose.pose.pose.position.x += r * np.cos(direction)
    new_amcl_pose.pose.pose.position.y += r * np.sin(direction)

    orient = quaternion_to_rad(new_amcl_pose.pose.pose.orientation.z, new_amcl_pose.pose.pose.orientation.w)

    orient += np.pi / 18 * np.random.randn()

    new_amcl_pose.pose.pose.orientation.z = rad_to_quaternion(orient)[2]

    new_amcl_pose.pose.pose.orientation.w = rad_to_quaternion(orient)[3]

    repub_amcl_pose.publish(new_amcl_pose)


def control(): 
    # r = rospy.Rate(100)
    while not rospy.is_shutdown(): 
        if gmm_covariance is None: 
            control_with_no_info()
        else: 
            control_with_gmm(gmm_mean, gmm_covariance, gmm_weight, amcl_pose, new_amcl_pose, odom)
        # r.sleep()


# Main method

if __name__ == '__main__':

    # Was trying to run controlling at a certain rate
    # rospy.init_node('state_feedback', anonymous=True)
    # rate = rospy.Rate(5) # ROS Rate at 5Hz

    rospy.init_node('gmm_controller', anonymous=True)
    
    sub_mean = rospy.Subscriber('gmm_mean', Float64MultiArray, callback_gmm_mean)
    sub_cov = rospy.Subscriber('gmm_covar', Float64MultiArray, callback_gmm_covar)
    sub_weight = rospy.Subscriber('gmm_weight', Float64MultiArray, callback_gmm_weight)

    sub_odom = rospy.Subscriber('amcl_pose', PoseWithCovarianceStamped, callback_amcl_pose)

    pub = threading.Thread(target=control)
    pub.start()

    rospy.spin()