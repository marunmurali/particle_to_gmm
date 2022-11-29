#!/usr/bin/env python3
# coding: utf-8

# Definition: A DWA controller that can navigate the robot to the goal set in RViz while avoiding obstacles in the environment
#
# Date of programming: 2022/7/7 ~ 20xx/xx/xx
#
# Current progress: N(F)
# A (working with a solid theoretical base) / B (seems to be working) / C (working with problems)
# F (totally not working) / N (not completed)

# Libraries

# - Basics

# import random
import threading
import rospy
import time
import math
import numpy as np
from numpy import linalg
# from sklearn import mixture
from functools import partial
# import matplotlib as mpl

# - ROS libraries
# from std_msgs.msg import String
from std_msgs.msg import MultiArrayDimension
# from std_msgs.msg import (Float32MultiArray, Float64MultiArray,
#                           Int8MultiArray, Int16MultiArray,
#                           Int32MultiArray, Int64MultiArray,
#                           UInt8MultiArray, UInt16MultiArray,
#                           UInt32MultiArray, UInt64MultiArray)
from std_msgs.msg import Float64MultiArray

from geometry_msgs.msg import (
    Twist, Point, Pose, PoseStamped, PoseWithCovarianceStamped, PoseArray)
from nav_msgs.msg import (Odometry, Path, OccupancyGrid)
from sensor_msgs.msg import LaserScan


# Global variables

# ROS parameters
gmm_flag = rospy.get_param('gmm')
n_gmm = rospy.get_param('num_of_gmm_dist')

# DWA parameters
dwa_horizon_param = 10

# Spec of lidar
lidar_range_min = 0.16
lidar_range_max = 8.0

# Storaged data
amcl_pose = None

# GMM parameters
gmm_mean = None
gmm_covariance = None
gmm_weight = None

# Mean Square Error
MSE_array = None

# Planned path by navigation node
planned_path = None

# Costmap array
costmap = None

# Control time interval
t_interval = 0.1

# Odometry of the robot
odom = Odometry()

# Error message (for plotting)
# error_msg = Point()

# If AMCL is processed to GMM and control based on GMM can be done
gmm_info = False

# Control states
# initial_rotation_finish = True
initial_rotation_finish = False
path_following_finish = False
final_rotation_finish = False

stop_flag = False

# Information of goal set in RViz
goal_x = 0.0
goal_y = 0.0
goal_z = 0.0
goal_w = 0.0
goal_heading = 0.0

# Center coordinates and relative distance
gmm_mean_matrix = np.zeros((2, 10))
gmm_weight_matrix = np.zeros(10)
gmm_ellipse_direction = np.zeros(10)
relative_distance_matrix = np.zeros((10, 10))
distance_to_path = np.zeros(10)
distance_to_obstacle = np.zeros(10)

# How to save the covariance of gmm?
# Is that in x, y direction?
gmm_covariance_matrix = np.zeros((2, 10))

pubCmd = rospy.Publisher('cmd_vel', Twist, queue_size=10)

previous_v = 0.0

laser_scan = None

# DWA cost function coefficients
alpha_1 = 1.0
alpha_2 = 1.0
alpha_3 = 1.0
alpha_4 = 1.0
alpha_5 = 1.0


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
    rad += np.pi / 2.0

    rad /= 2.0

    quaternion = np.zeros(4)

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


def gmm_process():
    global gmm_info

    if gmm_mean is None or gmm_covariance is None or gmm_weight is None:
        pass

    else:
        for i, (m, covar, weight) in enumerate(zip(gmm_mean, gmm_covariance, gmm_weight)):

            # rospy.loginfo(str(i) + str(weight))

            eig_val, eig_vec = linalg.eigh(covar)

            v = 2.0 * np.sqrt(5.991) * np.sqrt(eig_val)

            # Eigenvectors of covariance matrix
            u = eig_vec[0] / linalg.norm(eig_vec[0])

            angle = np.arctan(u[1] / u[0]) + 3 * np.pi / 2

            if angle > 2 * np.pi:
                angle = angle - 2 * np.pi

            gmm_mean_matrix[0][i] = m[0]
            gmm_mean_matrix[1][i] = m[1]

            # [0][i] means b and [1][i] means a
            gmm_covariance_matrix[0][i] = v[0]
            gmm_covariance_matrix[1][i] = v[1]

            gmm_weight_matrix[i] = weight

        # Calculation of relative distance of GMM clusters
        for i in range(n_gmm):
            for j in range(1, n_gmm):
                relative_distance_matrix[i][j] = linear_distance(
                    gmm_mean_matrix[0][i], gmm_mean_matrix[0][j], gmm_mean_matrix[1][i], gmm_mean_matrix[1][j])

        gmm_info = True

        for i in range(n_gmm): 
            rospy.loginfo('weight of ' + str(i + 1) + 'th cluster: ' + str(gmm_weight_matrix[i]))


def robot_control(v, a):
    # Initialize command publisher

    global pubCmd

    cmd_vel_msg = Twist()

    cmd_vel_msg.linear.x = v
    cmd_vel_msg.linear.y = 0.0
    cmd_vel_msg.linear.z = 0.0

    cmd_vel_msg.angular.x = 0.0
    cmd_vel_msg.angular.y = 0.0
    cmd_vel_msg.angular.z = a

    rospy.loginfo('v: ' + str(v))
    rospy.loginfo('a: ' + str(a))
    pubCmd.publish(cmd_vel_msg)

def cost_function_calculation(min_dis, max_dev, rel_dis, spd_diff, cls_size):
    # Here a cost function considering distance to goal,

    j_1 = alpha_1 * np.power(min_dis, 2)
    j_2 = alpha_2 * np.power(max_dev, 2)
    j_3 = alpha_3 * np.power(rel_dis, 2)
    j_4 = alpha_4 * np.power(spd_diff, 2)
    j_5 = alpha_5 * cls_size

    # rospy.loginfo('min distance: ' + str(j_1))
    # rospy.loginfo('max deviation: ' + str(j_2))
    # rospy.loginfo('relative distance: ' + str(j_3))
    # rospy.loginfo('speed difference: ' + str(j_4))
    # rospy.loginfo('cluster size: ' + str(j_5))


    return (j_1 + j_2 + j_3 + j_4 + j_5)


# def refined_cost_function_calculation():
#     (j_1, j_2, j_3, j_4) = 0.0


#     return (j_1, j_2, j_3, j_4)


def path_following(original_heading):
    # Note: to use several kinds of cost function J.

    global path_following_finish, previous_v

    (optimal_v, optimal_a) = (0.0, 0.0)

    v_range = np.array([-0.25, -0.20, -0.15, -0.10, -0.05,
                       0.0, 0.05, 0.10, 0.15, 0.20, 0.25])

    a_range = np.array([-0.5, -0.4, -0.3, -0.2, -0.1,
                       0.0, 0.1, 0.2, 0.3, 0.4, 0.5])

    # for i_gmm in range(n_gmm):

    #     optimal_cost_function = np.inf

    #     current_x = gmm_mean_matrix[0][i_gmm]
    #     current_y = gmm_mean_matrix[1][i_gmm]

    #     weight = gmm_weight_matrix[i_gmm]

    #     for i in range(len(v_range)):

    #         v = v_range[i]

    #         # to limit the speed of robot to save some computational time
    #         for j in range(len(a_range)):

    #             a = a_range[j]

    #             dwa_heading = original_heading
    #             dwa_x = current_x
    #             dwa_y = current_y

    #             for k1 in range(dwa_horizon_param):

    #                 dwa_x -= t_interval * v * np.sin(dwa_heading)
    #                 dwa_y += t_interval * v * np.cos(dwa_heading)

    #                 dwa_heading += t_interval * a

    #             min_error = np.inf

    #             for k2 in range(min(len(planned_path) - 1, 19)):
    #                 pose = planned_path[k2]

    #                 error = linear_distance(
    #                     dwa_x, pose.pose.position.x, dwa_y, pose.pose.position.y)

    #                 if error < min_error:
    #                     min_error = error

    #             remaining_distance = linear_distance(
    #                 dwa_x, goal_x, dwa_y, goal_y)

    #             cost_function = 1.0 * np.power(min_error, 2) + 1.0 * np.power(
    #                 v - previous_v, 2) + 0.1 * np.power(remaining_distance, 2)
    #             # plus others

    #             # rospy.loginfo('error score: ' + str(1.0 * pow(min_error, 2)))
    #             # rospy.loginfo('speed score: ' + str(1.0 * pow(min_error, 2)))

    #             # if remaining_distance >= 1.0:
    #             #     cost_function = 1.0 * pow(min_error, 2) + 1.0 * pow(0.26 - abs(v), 2)
    #             # else:
    #             #     cost_function = 1.0 * pow(min_error, 2)

    #             # Edition 1
    #             # cost_function = 1.0 * min_error + 1.0 * remaining_distance

    #             # cost_function = 1 * min_distance + 0.1 / np.pi * rad_diff + 1 * remaining_distance

    #             if cost_function < optimal_cost_function:
    #                 local_optimal_v = v
    #                 local_optimal_a = a
    #                 optimal_cost_function = cost_function

    #     optimal_v += weight * local_optimal_v
    #     optimal_a += weight * local_optimal_a

    # rospy.loginfo('optimal v: ' + str(optimal_v))
    # rospy.loginfo('optimal a: ' + str(optimal_a))

    optimal_cost_function = np.inf
    
    for i in range(len(v_range)):

        v = v_range[i]

        for j in range(len(a_range)):

            a = a_range[j]

            # Computation of cost function

            # Including the following terms
            min_distance_to_obstacle = np.inf
            max_deviation_from_path = 0.0

            total_relative_distance = 0.0
            total_gmm_cluster_size = 0.0

            for i_gmm in range(n_gmm):
                current_x = gmm_mean_matrix[0][i_gmm]
                current_y = gmm_mean_matrix[1][i_gmm]
                dwa_heading = original_heading
                
                # Dynamic calculation
                dwa_x = current_x
                dwa_y = current_y

                for k1 in range(dwa_horizon_param):
                    dwa_x -= t_interval * v * np.sin(dwa_heading)
                    dwa_y += t_interval * v * np.cos(dwa_heading)

                    dwa_heading += t_interval * a

                # Clearance calculation
                # Since it's not possible by now, it's set to 1.0
                clearance = 1.0

                if clearance < min_distance_to_obstacle: 
                    min_distance_to_obstacle = clearance
                
                # Deviation calculation 
                distance_to_path = np.inf

                for k2 in range(min(len(planned_path) - 1, 19)):
                    pose = planned_path[k2]

                    error = linear_distance(
                        dwa_x, pose.pose.position.x, dwa_y, pose.pose.position.y)

                    if error < distance_to_path:
                        distance_to_path = error
                
                if distance_to_path > max_deviation_from_path: 
                    max_deviation_from_path = distance_to_path

            # Summation of relative distance
            sum_relative_distance = 0.0 
            relative_distance_matrix

            speed_diff = v - previous_v

            # l_gmm and r_gmm. a and b used here
            sum_cluster_size = gmm_covariance_matrix.sum()

            # Calculation of cost function

            cost_function = cost_function_calculation(min_distance_to_obstacle, max_deviation_from_path, sum_relative_distance, speed_diff, sum_cluster_size)

            if cost_function < optimal_cost_function: 
                optimal_cost_function = cost_function
                
                optimal_v = v
                optimal_a = a


    if linear_distance(gmm_mean_matrix[0][0], goal_x, gmm_mean_matrix[1][0], goal_y) < 0.05:
        path_following_finish = True
        rospy.loginfo('Path following finished successfully. ')

    robot_control(optimal_v, optimal_a)

    previous_v = optimal_v


def initial_rotation(original_heading):
    # Inputs: original x, y and heading of the robot
    # Outputs: an angular command
    # To instruct the robot to turn to (approximately) the direction of the start of the path

    global initial_rotation_finish

    (optimal_v, optimal_a) = (0.0, 0.0)

    # # When distance is very small,  not turning to save time.
    # if linear_distance(original_x, goal_x, original_y, goal_y) < 2.0:
    #     initial_rotation_finish = True

    follow_heading = 0.0

    if len(planned_path) < 5:
        follow_heading = goal_heading
    else:
        follow_heading = atan2_customized(
            planned_path[4].pose.position.y - planned_path[0].pose.position.y, planned_path[4].pose.position.x - planned_path[0].pose.position.x)

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

    # return (optimal_v, optimal_a)
    robot_control(optimal_v, optimal_a)


def final_rotation(original_heading):
    global final_rotation_finish

    (optimal_v, optimal_a) = (0.0, 0.0)

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

    robot_control(optimal_v, optimal_a)


# Controller method
def control_with_gmm():

    # Global variables
    # global odom, gmm_mean, gmm_covariance, gmm_weight, amcl_pose
    # global goal_x, goal_y, goal_z, goal_w
    global initial_rotation_finish, path_following_finish, final_rotation_finish
    # global costmap

    # pubError = rospy.Publisher('error', Point, queue_size=10)

    original_v = odom.twist.twist.linear.x

    original_x = amcl_pose.pose.pose.position.x
    original_y = amcl_pose.pose.pose.position.y

    original_z = amcl_pose.pose.pose.orientation.z
    original_w = amcl_pose.pose.pose.orientation.w
    original_heading = quaternion_to_rad(original_z, original_w)

    # rospy.loginfo('x: ' + str(original_x))
    # rospy.loginfo('y: ' + str(original_y))
    # rospy.loginfo('heading: ' + str(original_heading))

    # original_angular = odom.twist.twist.angular.z

    optimal_v = 0.0
    optimal_a = 0.0

    # rospy.loginfo(str(original_a))

    # We should change the sample space according to the robot's current velocity and acceleration limit
    # It can't be found in the document, but estimation can be made like a max acceleration of 1.0 m/(s^2)
    # Now it's given up

    # original_v = odom.twist.twist.linear.x
    # original_a = odom.twist.twist.angular.z

    if initial_rotation_finish is False:
        # initial_rotation(original_x, original_y, original_heading)
        initial_rotation(original_heading)

    elif path_following_finish is False:

        path_following(original_heading)

    elif final_rotation_finish is False:
        final_rotation(original_heading)

    else:
        pass

    # robot_control(optimal_v, optimal_a)


def control_with_no_gmm():
    robot_control(0.01, 0)


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
    gmm_process()


def callback_amcl_pose(data):
    global amcl_pose
    amcl_pose = data
    # gmm_process()


def callback_odom(data):
    global odom
    odom = data


# What shall we do here?
def callback_path(data):
    global planned_path
    planned_path = data.poses


def callback_goal(data):
    global initial_rotation_finish, path_following_finish, final_rotation_finish

    global goal_x, goal_y, goal_heading

    goal_x = data.pose.position.x
    goal_y = data.pose.position.y

    goal_z = data.pose.orientation.z
    goal_w = data.pose.orientation.w

    goal_heading = quaternion_to_rad(goal_z, goal_w)

    # rospy.loginfo('The goal is set. ')

    initial_rotation_finish = False
    path_following_finish = False
    final_rotation_finish = False


def callback_costmap(data):
    global costmap

    costmap = data.data

def callback_laser_scan(data): 
    global laser_scan

    laser_scan = data.ranges


def control():
    r = rospy.Rate(2)

    while not rospy.is_shutdown():
        rospy.loginfo('running... ')

        start_time = time.time()

        if planned_path is None:
            rospy.loginfo('Waiting for path...')
            pass

        else:
            # if gmm_mean is None or gmm_covariance is None or gmm_weight is None:
            if gmm_info is False:
                control_with_no_gmm()

            else:
                control_with_gmm()

        end_time = time.time()

        rospy.loginfo('Runtime of the program is ' + str(end_time - start_time))

        r.sleep()


# Main function
if __name__ == '__main__':

    rospy.init_node('gmm_controller', anonymous=True)

    rospy.Subscriber('gmm_mean', Float64MultiArray, callback_gmm_mean)
    rospy.Subscriber('gmm_covar', Float64MultiArray, callback_gmm_covar)
    rospy.Subscriber('gmm_weight', Float64MultiArray, callback_gmm_weight)

    sub_amcl_pose = rospy.Subscriber(
        'amcl_pose', PoseWithCovarianceStamped, callback_amcl_pose)

    sub_odom = rospy.Subscriber('odom', Odometry, callback_odom)

    sub_path = rospy.Subscriber('move_base/NavfnROS/plan', Path, callback_path)

    sub_goal = rospy.Subscriber(
        '/move_base_simple/goal', PoseStamped, callback_goal)

    sub_costmap = rospy.Subscriber(
        '/move_base/local_costmap/costmap', OccupancyGrid, callback_costmap)

    sub_laser_scan = rospy.Subscriber('/scan', LaserScan, callback_laser_scan)

    control_thread = threading.Thread(target=control)
    control_thread.start()

    calculation_thread = threading.Thread(target=control)
    calculation_thread.start()

    rospy.spin()
