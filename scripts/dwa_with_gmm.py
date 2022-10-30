#!/usr/bin/env python3
# coding: utf-8

# Definition: A DWA controller used in the simulation experiment. 
#
# Date of programming: 2022/7/7 ~ 20xx/xx/xx
#
# Current progress: C
# A (working with a solid theoretical base) / B (seems to be working) / C (working with problems)
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

    # Global variables
    global odom, gmm_mean, gmm_covariance, gmm_weight, amcl_pose
    global goal_x, goal_y, goal_z, goal_w
    global initial_rotation_finish, path_following_finish, final_rotation_finish

    # Initialize command publisher
    pubCmd = rospy.Publisher('cmd_vel', Twist, queue_size = 10)
    # pubError = rospy.Publisher('error', Point, queue_size=10)

    # original_v = odom.twist.twist.linear.x

    original_x = amcl_pose.pose.pose.position.x
    original_y = amcl_pose.pose.pose.position.y

    original_z = amcl_pose.pose.pose.orientation.z
    original_w = amcl_pose.pose.pose.orientation.w

    # original_z = odom.pose.pose.orientation.z
    # original_w = odom.pose.pose.orientation.w

    original_heading = quaternion_to_rad(original_z, original_w)

    # rospy.loginfo('x: ' + str(original_x))
    # rospy.loginfo('y: ' + str(original_y))
    # rospy.loginfo('heading: ' + str(original_heading))

    # original_angular = odom.twist.twist.angular.z
    
    optimal_v = 0.0 
    optimal_a = 0.0

    original_v = odom.twist.twist.linear.x

    # We should change the sample space according to the robot's current velocity and acceleration limit

    # But currently it can't be found
    # Estimation can be made like a max acceleration of 1.0 m/s2

    v_range = np.array([-0.1, -0.08, -0.06, -0.04, -0.02, 0, 0.02, 0.04, 0.06, 0.08, 0.1]) + original_v
    # v_range = np.array([0.05, 0.07, 0.09, 0.11, 0.13, 0.15, 0.17, 0.19, 0.21, 0.23, 0.25])

    a_range = np.array([-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5])

    if initial_rotation_finish is False: 
        
        if linear_distance(original_x, goal_x, original_y, goal_y) < 2.0: 
            initial_rotation_finish = True
        
        
        follow_heading = 0.0
        if len(planned_path) < 5: 
            follow_heading = goal_heading
        else: 
            # rospy.loginfo(str(planned_path[9].pose.position.x))

            follow_heading = atan2_customized(planned_path[4].pose.position.y - planned_path[0].pose.position.y, planned_path[4].pose.position.x - planned_path[0].pose.position.x)

        heading_difference = original_heading - follow_heading

        if heading_difference > np.pi: 
            heading_difference = heading_difference - 2.0 * np.pi

        if heading_difference < -np.pi: 
            heading_difference = heading_difference + 2.0 * np.pi

        if heading_difference > 0.1: 
            optimal_a = -0.5
        elif heading_difference < -0.1: 
            optimal_a = 0.5

        else: 
            initial_rotation_finish = True

            rospy.loginfo('Initial rotation finished. ')

    elif path_following_finish is False: 

        optimal_v = 0.0

        optimal_a = 0.0
        
        for i, (m, covar, weight) in enumerate(zip(gmm_mean, gmm_covariance, gmm_weight)):
            
            # rospy.loginfo(str(i) + str(weight))
            
            eig_val, eig_vec = linalg.eigh(covar)

            v = 2.0 * np.sqrt(5.991) * np.sqrt(eig_val)
            u = eig_vec[0] / linalg.norm(eig_vec[0])
            

            angle = np.arctan(u[1] / u[0]) + 3 * np.pi / 2

            if angle > 2 * np.pi: 
                angle = angle - 2 * np.pi

            x = m[0]
            y = m[1]

            b_ellipse = v[0]
            a_ellipse = v[1]

            current_x = x
            current_y = y 
        
            optimal_cost_function = np.inf

            current_distance = linear_distance(current_x, goal_x, current_y, goal_y)

            for i in range(len(v_range)): 

                for j in range(len(a_range)): 

                    v = v_range[i]

                    a = a_range[j]

                    h = original_heading
                    x = current_x
                    y = current_y

                    for k1 in range(dwa_horizon_param): 

                        x -= t_interval * v * np.sin(h)
                        y += t_interval * v * np.cos(h)

                        h += t_interval * a

                    min_error = np.inf

                    # Calculating minimal distance to the path
                    for k2, pose in enumerate(planned_path[:min(len(planned_path), 25)]):  

                        d = linear_distance(x, pose.pose.position.x, y, pose.pose.position.y)

                        if d < min_error: 
                            min_error = d

                    # angular difference
                    # rad_diff = np.abs(goal_heading - h) 
                    # if rad_diff > np.pi: 
                    #     rad_diff = 2 * np.pi - rad_diff

                    remaining_distance = linear_distance(x, goal_x, y, goal_y)

                    # Edition 2
                    # cost_function = 1.0 * pow(min_error, 2) - 1.0 * pow(remaining_distance - current_distance, 2)
                    cost_function = 1.0 * pow(min_error, 2) + 1.0 * (remaining_distance - current_distance) * np.abs(remaining_distance - current_distance)

                    # Edition 1
                    # cost_function = 1.0 * min_error + 1.0 * remaining_distance

                    # cost_function = 1 * min_distance + 0.1 / np.pi * rad_diff + 1 * remaining_distance

                    if cost_function < optimal_cost_function: 
                        local_optimal_v = v
                        local_optimal_a = a
                        optimal_cost_function = cost_function

            optimal_v += weight * local_optimal_v
            optimal_a += weight * local_optimal_a

        if linear_distance(original_x, goal_x, original_y, goal_y) < 0.05: 
            path_following_finish = True
            rospy.loginfo('Path following finished. ')

    elif final_rotation_finish is False: 

        heading_difference = original_heading - goal_heading

        if heading_difference > np.pi: 
            heading_difference = heading_difference - 2.0 * np.pi

        if heading_difference < -np.pi: 
            heading_difference = heading_difference + 2.0 * np.pi
        
        if heading_difference > 0.1: 
            optimal_a = -0.5
        elif heading_difference < -0.1: 
            optimal_a = 0.5   
        else: 
            final_rotation_finish = True
            rospy.loginfo('Final rotation finished. ') 

    else: 
        pass
        
    cmd_vel_msg = Twist()

    cmd_vel_msg.linear.x = optimal_v        
    cmd_vel_msg.linear.y = 0.0    
    cmd_vel_msg.linear.z = 0.0

    cmd_vel_msg.angular.x = 0.0
    cmd_vel_msg.angular.y = 0.0
    cmd_vel_msg.angular.z = optimal_a

    pubCmd.publish(cmd_vel_msg)

    # rospy.loginfo('v: ' + str(optimal_v))
    # rospy.loginfo('a: ' + str(optimal_a))

    
def control_with_no_gmm(): 
    pubCmd = rospy.Publisher('cmd_vel', Twist, queue_size = 10) 

    # global error_msg, odom

    cmd_vel_msg = Twist()

    cmd_vel_msg.linear.x = 0.01
    
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
    amcl_pose = data


def callback_odom(data): 
    global odom
    odom = data


# What shall we do here? 
def callback_path(data): 
    global planned_path
    planned_path = data.poses


def callback_goal(data): 
    global initial_rotation_finish, path_following_finish, final_rotation_finish

    global goal_x, goal_y, goal_z, goal_w, goal_heading

    goal_x = data.pose.position.x
    goal_y = data.pose.position.y

    goal_z = data.pose.orientation.z
    goal_w = data.pose.orientation.w

    goal_heading = quaternion_to_rad(goal_z, goal_w)

    rospy.loginfo('heading of goal: ' + str(goal_heading))

    initial_rotation_finish = False
    path_following_finish = False
    final_rotation_finish = False


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

        start_time = time.time()

        if planned_path is None: 
            rospy.loginfo('Waiting for path...')
        else: 
            if gmm_mean is None or gmm_covariance is None or gmm_weight is None: 
                control_with_no_gmm()
            else: 
                control_with_gmm() 
            # control_with_gmm()

        end_time = time.time()

        print("Runtime of the program is %f" %(end_time - start_time))

        r.sleep()



# Main method

if __name__ == '__main__':

    # rospy.init_node('state_feedback', anonymous=True)

    rospy.init_node('gmm_controller', anonymous=True)

    # Preparing for randomization in DWA
    # random.seed()
    
    sub_mean = rospy.Subscriber('gmm_mean', Float64MultiArray, callback_gmm_mean)
    sub_cov = rospy.Subscriber('gmm_covar', Float64MultiArray, callback_gmm_covar)
    sub_weight = rospy.Subscriber('gmm_weight', Float64MultiArray, callback_gmm_weight)

    sub_amcl_pose = rospy.Subscriber('amcl_pose', PoseWithCovarianceStamped, callback_amcl_pose)

    sub_odom = rospy.Subscriber('odom', Odometry, callback_odom)

    sub_path = rospy.Subscriber('move_base/NavfnROS/plan', Path, callback_path)

    sub_goal = rospy.Subscriber('/move_base_simple/goal', PoseStamped, callback_goal)

    control_thread = threading.Thread(target=control)
    control_thread.start()

    # start_time = rospy.get_time()

    rospy.spin()