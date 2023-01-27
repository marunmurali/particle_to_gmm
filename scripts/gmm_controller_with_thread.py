#!/usr/bin/env python3
# coding: utf-8

# Definition: A state-feedback-based contoller used in the simulation environment of Turtlebot 3. 
#
# Date of programming: 2022/2 ~ 2022/6 (mainly)
#
# Current progress: B
# A (working fine with a solid theoretical base) / B (seems to be working fine) / C (working with problems or sometimes working)
# F (totally not working) / N (not completed)

# Copyright (c) 2022, Arun Muraleedharan, Li Hanjie
# 
# Original License Agreement: 
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


# Imports

# - Basics
import threading
from turtle import position
import rospy
import time
import math
import numpy as np
# from numpy.core.fromnumeric import mean
# from numpy.ma.core import concatenate
from numpy import linalg
from sklearn import mixture
from functools import partial
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)


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

## ROS Parameters
gmm_flag = rospy.get_param('gmm')

orient_x = rospy.get_param('orient_x')
orient_y = rospy.get_param('orient_y')
goal_x = rospy.get_param('goal_x')
goal_y = rospy.get_param('goal_y')

no_gmm_speed = rospy.get_param('no_gmm_speed')



## Robot position ground truth
gazebo_odom = Odometry()

gazebo_odom.pose.pose.position.x = orient_x
gazebo_odom.pose.pose.position.y = orient_y

## Feedback gains of the state feedback controller
# k1 = -1.0
k2 = -0.50
k3 = -0.10
k4 = -0.10

## GMM and odometry data
gmm_mean = None
gmm_covariance = None
gmm_weight = None

odom = Odometry()

gazebo_odom = Odometry()
gazebo_info = False

start_flag = False
stop_flag = False

path_following_start_time = None

# mse_list = []
# mse_calculation = False

# For corridor (later to be changed to parameter)
# count_time = 50.0

# For campus
count_time = 80.0

## Error data
# error_msg = Point()

cmd_vel_msg = Twist()

error_plt = np.empty(0)
speed_plt = np.empty(0)
a_speed_plt = np.empty(0)
t_plt = np.empty(0)
squared_error = np.empty(0)


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


def scalar_product(x1, x2, y1, y2): 
    return (x1 * x2 + y1 * y2)


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

def linear_distance(x1, x2, y1, y2):
    d = np.sqrt(np.power(x1 - x2, 2) + np.power(y1 - y2, 2))

    return d

# Controller method
def control_with_gmm(means, covariances, weights, amcl_pose, odom): 

    global error_msg, error_mse

    # global error_msg_new, error_msg_pos, previous_time

    global mse_list, mse_calculation, start_time, count_time

    global gazebo_odom

    global stop_flag

    global cmd_vel_msg

    pubCmd = rospy.Publisher('cmd_vel', Twist, queue_size=10)
    # pubError = rospy.Publisher('error', Point, queue_size=10)


    linear_cmd = 0.0
    angular_cmd = 0.0

    orient_z = amcl_pose.pose.pose.orientation.z
    orient_w = amcl_pose.pose.pose.orientation.w

    theta = quaternion_to_rad(orient_z, orient_w)

    orientation_err = get_err_orient(theta)

    if gmm_flag: 
        sum_squared_weight = 0.0

        # Path following using GMM
        for i, (m, covar, weight) in enumerate(zip(means, covariances, weights)):
            
            # rospy.loginfo(str(i) + str(weight))
            
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
            cmd4 = k4 * l2

            angular_cmd += math.pow(weight, 2) * (k1 * x_err + k2 * orientation_err) 
            linear_cmd += math.pow(weight, 2) * (cmd3 + cmd4)

            sum_squared_weight += math.pow(weight, 2)

        angular_cmd = angular_cmd / sum_squared_weight
        
        linear_cmd = linear_cmd / sum_squared_weight
        linear_cmd += 0.26

        if linear_cmd < 0.1: 
            linear_cmd = 0.1

        if linear_cmd > 0.26: 
            linear_cmd = 0.26
           
        # rospy.loginfo('speed: ' + str(linear_cmd))
        
    else: 
        # Path following without GMM

        x = amcl_pose.pose.pose.position.x
        y = amcl_pose.pose.pose.position.y
        x_err = get_err_position(x, y)

        if ((-np.sin(theta) * (-np.sin(orient_ref)) + np.cos(theta) * np.cos(orient_ref)) 
                * (goal_x - orient_x)) >= 0: 
            k1 = -0.50
        else: 
            k1 = 0.50

        k1 = k1 * np.abs(np.cos(orientation_err))

        angular_cmd = k1 * x_err + k2 * orientation_err
        # angular_cmd = k1 * x_err + k2 * orientation_err + 0.2 * (np.random.random(1) - 0.5)

        linear_cmd = no_gmm_speed

        # rospy.loginfo('controlling w/o gmm')

    # xForError = gazebo_odom.pose.pose.position.x
    # yForError = gazebo_odom.pose.pose.position.y

    # position_error = get_err_position(xForError, yForError)

    # zForError = amcl_pose.pose.pose.orientation.z
    # wForError = amcl_pose.pose.pose.orientation.w   

    # theta_error = quaternion_to_rad(zForError, wForError)

    # orientation_error = get_err_orient(theta_error)

    # rospy.loginfo('linear error: ' + str(error_msg.x))
    # rospy.loginfo('angular error: ' + str(error_msg.y))

    # dist_goal = np.sqrt(np.power(x - goal_x, 2) + np.power(y - goal_y, 2))

    # rospy.loginfo('linear command: ' + str(linear_cmd))
    # rospy.loginfo('angular command: ' + str(angular_cmd))

    # rospy.loginfo('x coordinate: ' + str(x) + '; y coordinate: ' + str(y))

    # Controlling of the robot

    cmd_vel_msg.linear.x = linear_cmd
    # cmd_vel_msg.linear.x = 0.20 + linear_cmd + 0.1 * (np.random.random(1) - 0.5)
        
    cmd_vel_msg.linear.y = 0.0    
    cmd_vel_msg.linear.z = 0.0

    cmd_vel_msg.angular.x = 0.0
    cmd_vel_msg.angular.y = 0.0
    cmd_vel_msg.angular.z = angular_cmd

    # New way of detecting path following finish
    if gmm_flag: 
        for i, m in enumerate(means):
        # if linear_distance(gmm_mean_matrix[0][i], goal.x, gmm_mean_matrix[1][i], goal.y) < 0.1:
        #     path_following_finish = True
            x = m[0]
            y = m[1]

            d = scalar_product(goal_x - x, goal_x - orient_x, goal_y - y, goal_y - orient_y) / linear_distance(goal_x, orient_x, goal_y, orient_y)

            if d < 0.0: 
                stop_flag = True   
    else: 
        x = amcl_pose.pose.pose.position.x
        y = amcl_pose.pose.pose.position.y

        d = scalar_product(goal_x - x, goal_x - orient_x, goal_y - y, goal_y - orient_y) / linear_distance(goal_x, orient_x, goal_y, orient_y)
        if d < 0.0: 
            stop_flag = True   


    # if dist_goal < 0.2: 
    #     stop_flag = True

    if stop_flag is True: 
        cmd_vel_msg.linear.x = 0.0
        cmd_vel_msg.angular.z = 0.0
    
    # error_msg.x = position_error
    # error_msg.y = cmd_vel_msg.linear.x
    
    pubCmd.publish(cmd_vel_msg)
    # pubError.publish(error_msg)


    
def control_with_no_info(): 

    pubCmd = rospy.Publisher('cmd_vel', Twist, queue_size=10) 

    pubError = rospy.Publisher('error', Point, queue_size=10)

    global error_msg, odom, cmd_vel_msg

    # xForError = odom.pose.pose.position.x
    # yForError = odom.pose.pose.position.y

    # error_msg.x = get_err_position(xForError, yForError)
    # error_msg.y = 0.0

    # cmd_vel_msg = Twist()

    cmd_vel_msg.linear.x = 0.01
    
    pubCmd.publish(cmd_vel_msg)

    # pubError.publish(error_msg)


# Callback methods

def callback_gmm_mean(data):
    global gmm_mean
    # rospy.loginfo('received gmm mean')
    gmm_mean = to_numpy_f64(data)

def callback_gmm_covar(data):
    global gmm_covariance
    # rospy.loginfo('received gmm covariance')
    gmm_covariance = to_numpy_f64(data)

def callback_gmm_weight(data): 
    global gmm_weight
    # rospy.loginfo('received gmm weight')
    gmm_weight = to_numpy_f64(data)

def callback_amcl_pose(data):
    # repub_amcl_pose = rospy.Publisher('new_amcl_pose', PoseWithCovarianceStamped, queue_size=10)
    global amcl_pose
    # rospy.loginfo('received amcl_pose')
    amcl_pose = data; 


def callback_odom(data): 
    global odom
    odom = data


def callback_gazebo_odom(data): 
    global gazebo_odom, gazebo_info
    gazebo_odom = data
    gazebo_info = True


def control(): 
    global path_following_start_time, start_flag

    r = rospy.Rate(10)

    while not rospy.is_shutdown(): 
        if not start_flag: 
            path_following_start_time = time.time()
            start_flag = True
            
        if gmm_mean is None or gmm_covariance is None or gmm_weight is None: 
            control_with_no_info()
        else: 
            control_with_gmm(gmm_mean, gmm_covariance, gmm_weight, amcl_pose, odom)
        r.sleep()


def measure():
    # error related
    global squared_error, error_plt, speed_plt, a_speed_plt, t_plt, gazebo_odom, cmd_vel_msg

    global start_flag, stop_flag, calc, plot_finish

    global path_following_start_time

    r_measure = rospy.Rate(10)

    while not rospy.is_shutdown():
        if not stop_flag: 
            if start_flag and gazebo_info: 
                # start_time = time.time()

                squared_error = np.append(squared_error, np.power(get_err_position(gazebo_odom.pose.pose.position.x, gazebo_odom.pose.pose.position.y), 2)) 
                speed_plt = np.append(speed_plt, cmd_vel_msg.linear.x)
                a_speed_plt = np.append(a_speed_plt, cmd_vel_msg.angular.z)
                error_plt = np.append(error_plt, get_err_position(gazebo_odom.pose.pose.position.x, gazebo_odom.pose.pose.position.y))
                t_plt = np.append(t_plt, time.time() - path_following_start_time)
                
                end_time = time.time()
                # rospy.loginfo('Runtime of the measurement program is ' + str(end_time - start_time))


        r_measure.sleep()


# For 1st secondary axis, convert and revert share the same function. 
def sub_conv(x): 
    return x

def sub2_conv(x): 
    return 0.2 * x

def sub2_rev(x): 
    return 5.0 * x


def result_plot(): 
    global t_plt, error_plt, speed_plt, a_speed_plt

    plt.rcParams['figure.figsize'] = 8, 5.5

    fig, ax = plt.subplots(constrained_layout=True)

    # plt_x_1 = np.empty(0)
    # for i in range(len(error_plt)): 
    #     plt_x_1 = np.append(plt_x_1, 0.1 * (i + 1))

    # plt_x_2 = np.empty(0)
    # for i in range(len(speed_plt)): 
    #     plt_x_2 = np.append(plt_x_2, 0.1 * (i + 1))

    # ax.plot(plt_x_1, error_plt, label='error (distance to the reference line)')

    # ax.plot(plt_x_2, speed_plt, label='error (distance to the reference line)')


    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, 0.5)

    ax.plot(t_plt, error_plt, label='error (distance to the reference line)', color='r')
    ax.plot(t_plt, speed_plt, label='speed command', color='g', linestyle ="dashed")
    ax.plot(t_plt, 5.0 * a_speed_plt, label='angular speed command', color='b', linestyle="dotted")

    ax.set_xlabel('t/s')
    ax.set_ylabel('error/m')
    ax.yaxis.set_major_locator(MultipleLocator(0.1))

    if gmm_flag: 
        ax.set_title('State-Feedback-with-GMM controller path following', fontsize=14)
    else: 
        ax.set_title('State-Feedback-without-GMM controller path following', fontsize=14)

    ax_sub = ax.secondary_yaxis('right', functions=(sub_conv, sub_conv))
    ax_sub.set_ylabel('speed/(m/s)')
    ax_sub.yaxis.set_major_locator(MultipleLocator(0.1))

    ax_sub2 = ax.secondary_yaxis(1.15, functions=(sub2_conv, sub2_rev))
    ax_sub2.set_ylabel('angular speed/(rad/s)')
    ax_sub2.yaxis.set_major_locator(MultipleLocator(0.02))

    ax.legend()

    plt.show()


def final_calculation(): 
    global squared_error

    path_following_end_time = time.time()
    
    error_mse = np.average(squared_error)

    path_following_time = path_following_end_time - path_following_start_time

    rospy.loginfo(str(path_following_start_time))

    rospy.loginfo('Path following time = ' + str(path_following_time) + 's. ')
    rospy.loginfo('MSE of error = ' + str(error_mse) + 'm^2')


# Main method

if __name__ == '__main__':
    # Was trying to run controlling at a certain rate
    # rospy.init_node('state_feedback', anonymous=True)
    # rate = rospy.Rate(5) # ROS Rate at 5Hz

    rospy.init_node('gmm_controller', anonymous=True)
    
    global k, b, orient_ref

    final = False

    k = (goal_y - orient_y) / (goal_x - orient_x)
    b = orient_y - k * orient_x

    orient_ref = -np.arctan2(goal_x - orient_x, goal_y - orient_y)
    
    sub_mean = rospy.Subscriber('gmm_mean', Float64MultiArray, callback_gmm_mean)
    sub_cov = rospy.Subscriber('gmm_covar', Float64MultiArray, callback_gmm_covar)
    sub_weight = rospy.Subscriber('gmm_weight', Float64MultiArray, callback_gmm_weight)

    sub_amcl_pose = rospy.Subscriber('amcl_pose', PoseWithCovarianceStamped, callback_amcl_pose)
    sub_odom = rospy.Subscriber('odom', Odometry, callback_odom)
    sub_gazebo_odom = rospy.Subscriber('gazebo_odom', Odometry, callback_gazebo_odom)

    pub = threading.Thread(target=control)
    pub.start()

    measure_thread = threading.Thread(target=measure)
    measure_thread.start()

    if gmm_flag == True: 
        rospy.loginfo('Running with GMM')
    else: 
        rospy.loginfo('Running without GMM')
        rospy.loginfo('control speed: ' + str(no_gmm_speed))

    while not rospy.is_shutdown():
        if stop_flag: 
            if not final: 
                final_calculation()
                result_plot()
                final = True