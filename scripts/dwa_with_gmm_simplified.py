#!/usr/bin/env python3
# coding: utf-8

# Definition: A refined DWA controller with comparison of multiple settings of costs function
#
# Date of programming: 2022/12/5～2023/1/10
# Current progress: N
# A (working with a solid theoretical base) / B (seems to be working) / C (working with problems)
# F (totally not working) / N (on progress)

# 20230109 Question: shall we use a single set of start and goal points? Or shall we manually set a simple path?  
# Can we define a path in the global variables? 
# 20230110 Asnwer: I am doing that. 

# Clearance calculation is still a problem. 

# Libraries

## Basics

# import random
import threading
import rospy
import time
import math
import numpy as np
from functools import partial
import matplotlib as mpl

## ROS libraries
from std_msgs.msg import (MultiArrayDimension, Float64MultiArray)
from geometry_msgs.msg import (
    Twist, Point, Pose, PoseStamped, PoseWithCovarianceStamped, PoseArray)
from nav_msgs.msg import (Odometry, Path, OccupancyGrid)
from sensor_msgs.msg import LaserScan


# Global variables

## ROS parameters
gmm_flag = rospy.get_param('gmm')
n_gmm = rospy.get_param('num_of_gmm_dist')

## DWA parameters
dwa_horizon_param = 10

## Spec of lidar
lidar_range_min = 0.16
lidar_range_max = 5.0

## Lidar data step length (equal to degree), for saving computational time:
lidar_step = 30

## Control time interval
t_interval = 0.2

## If AMCL is processed to GMM and control based on GMM can be done
gmm_info = False

# Storaged data

## AMCL pose
amcl_pose = None

## GMM parameters
gmm_mean = None
gmm_covariance = None
gmm_weight = None

## Mean Square Error
MSE_array = None

# ## Planned path by navigation node
# planned_path = None

# ## Costmap array
# costmap = None

## Data of lidar scan
laser_scan = None
laser_scan_coordinate = np.zeros((2, 360))

## Odometry of the robot
odom = Odometry()
gazebo_odom = Odometry()

## Error message (for plotting)
# error_msg = Point()

## Control states
initial_rotation_finish = False
path_following_finish = False
final_rotation_finish = False
# stop_flag = False

## Information of goal set in RViz
start_point = Pose()
start_point.position.x= rospy.get_param('orient_x')
start_point.position.y= rospy.get_param('orient_y')

goal = Pose()
goal.position.x = rospy.get_param('goal_x')
goal.position.y = rospy.get_param('goal_y')

## slope and intercept of the path
k = (goal.position.y - start_point.position.y) / (goal.position.x - start_point.position.x)
b = start_point.position.y - k * start_point.position.x

# goal_heading_angle = 0.0
goal_heading_angle = -np.arctan2(goal.position.x - start_point.position.x, goal.position.y - start_point.position.y)


## Center coordinates and relative distance
gmm_mean_matrix = np.zeros((2, 10))
gmm_weight_matrix = np.zeros(10)
gmm_ellipse_direction = np.zeros(10)
relative_distance_matrix = np.zeros((10, 10))

distance_to_path = np.zeros(10)
distance_to_obstacle = np.zeros(10)

gmm_covariance_matrix = np.zeros((2, 10))
# How to save the covariance of gmm?
# Is that in x, y direction?

cost_function_gmm_cluster = np.zeros(10)

## Previous speed and angular speed
previous_v = 0.0
previous_a = 0.0

## DWA cost function coefficients
alpha = np.zeros(6)
alpha[0] = 1.0
alpha[1] = 1.0
alpha[2] = 1.0
alpha[3] = 1.0
alpha[4] = 1.0
alpha[5] = 1.0

## Speed command publisher
pubCmd = rospy.Publisher('cmd_vel', Twist, queue_size=10)


# Methods

# Conversion from original atan2 to the angle system we are using
def atan2_customized(y, x):
    # Increasing counter-clockwise, 0 facing upwards
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


def get_err_position(x, y):
    d = (y - k * x - b) / math.sqrt(1 + math.pow(k, 2))
    
    # rospy.loginfo('linear_error = ' + str(d))
    
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
    return np.array(multiarray.data, dtype=pytype).reshape(dims).astype(dtype)


to_multiarray_f64 = partial(_numpy_to_multiarray, Float64MultiArray)


to_numpy_f64 = partial(_multiarray_to_numpy, float, np.float64)


def gmm_process():
    global gmm_info

    if (gmm_mean is None) or (gmm_covariance is None) or (gmm_weight is None):
        pass

    else:
        for i, (m, covar, weight) in enumerate(zip(gmm_mean, gmm_covariance, gmm_weight)):

            # rospy.loginfo(str(i) + str(weight))

            eig_val, eig_vec = np.linalg.eigh(covar)

            v = 2.0 * np.sqrt(5.991) * np.sqrt(eig_val)

            # Eigenvectors of covariance matrix
            u = eig_vec[0] / np.linalg.norm(eig_vec[0])

            angle = np.arctan(u[1] / u[0]) + 3 * np.pi / 2

            if angle > 2 * np.pi:
                angle = angle - 2 * np.pi

            # Mean value of gmm distribution
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


def cost_function_calculation(dis_goal, min_dis_obs, max_dev, spd_diff, cls_rel_dis, cls_size):

    j = np.zeros(6)

    # Distance to the goal
    j[0] = alpha[0] * np.power(dis_goal, 1)
    # Minimal distance to the obstacle
    j[1] = alpha[1] * np.power(min_dis_obs, 1)
    # Max deviation from the path
    j[2] = alpha[2] * np.power(max_dev, 1)
    # Speed difference
    j[3] = alpha[3] * np.power(spd_diff, 2)
    # Relative distance of clusters * Not included now
    j[4] = alpha[4] * np.power(cls_rel_dis, 1)
    # Cluster size * Not included now
    j[5] = alpha[5] * np.power(cls_size, 1)

    # rospy.loginfo('distance to goal' + str(j[0]))
    # rospy.loginfo('min distance to obstacle: ' + str(j[1]))
    # rospy.loginfo('max deviation: ' + str(j[2]))
    # rospy.loginfo('speed difference: ' + str(j[3]))
    # rospy.loginfo('relative distance: ' + str(j[4]))
    # rospy.loginfo('cluster size: ' + str(j[5]))

    return np.sum(j)


# The path following function
def path_following(original_heading):
    global path_following_finish, previous_v, previous_a, cost_function_gmm_cluster

    start_time = time.time()

    lidar_safety_flag = True
    speed_flag = True

    optimal_v = np.zeros(10)
    optimal_a = np.zeros(10)

    # cost_function = np.zeros(10)

    # laser_scan_x = np.zeros((10, 360))
    # laser_scan_y = np.zeros((10, 360))

    # Velocity space with dynamic constraints
    v_range = np.array([-0.10, -0.08, -0.06, -0.04, -0.02,
                        0.0, 0.02, 0.04, 0.06, 0.08, 0.10]) + previous_v
    a_range = np.array([-0.10, -0.08, -0.06, -0.04, -0.02,
                        0.0, 0.02, 0.04, 0.06, 0.08, 0.10]) + previous_a

    # Why convert lidar data to global coordinate? It's unnecessary.
    for i in range(len(laser_scan)):
        laser_scan_theta = np.pi / 180.0 * i
        laser_scan_coordinate[0][i] = -laser_scan[i] * np.sin(laser_scan_theta)
        laser_scan_coordinate[1][i] = laser_scan[i] * np.cos(laser_scan_theta)


    for i in range(len(v_range)):

        v = v_range[i]

        for j in range(len(a_range)):

            a = a_range[j]

            # Computation of cost function
            speed_diff = v - previous_v

            for i_gmm in range(n_gmm):
                cost_function_gmm_cluster[i_gmm] = np.inf

                current_x = gmm_mean_matrix[0][i_gmm]
                current_y = gmm_mean_matrix[1][i_gmm]

                # Including the following terms
                min_distance_to_obstacle = np.inf
                max_deviation_from_path = 0.0

                total_relative_distance = 0.0
                total_gmm_cluster_size = 0.0

                # Kinematic calculation
                dwa = Point(current_x, current_y, 0)
                dwa_heading = original_heading

                dwa_local = Point(0.0, 0.0, 0.0)
                dwa_heading_local = 0.0

                for k1 in range(dwa_horizon_param):
                    dwa.x -= t_interval * v * np.sin(dwa_heading)
                    dwa.y += t_interval * v * np.cos(dwa_heading)
                    dwa_heading += t_interval * a

                    dwa_local.x -= t_interval * v * np.sin(dwa_heading_local)
                    dwa_local.y += t_interval * v * np.cos(dwa_heading_local)
                    dwa_heading_local += t_interval * a

                # Distance to goal calculation
                distance_to_goal = linear_distance(
                    goal.position.x, dwa.x, goal.position.y, dwa.y)

                # Clearance calculation
                # min_distance_to_obstacle = np.inf
                clearance = np.inf

                # i_angle = np.round(atan2_customized(dwa_local.y, dwa_local.x) / lidar_step)

                # # Limitation processing
                # if i_angle == len(laser_scan):
                #     i_angle = 0

                for i in range(len(laser_scan)): 
                    distance_to_obstacle = linear_distance(dwa_local.x, laser_scan_coordinate[0][i], dwa_local.y, laser_scan_coordinate[1][i])
                    if distance_to_obstacle < clearance: 
                        clearance = distance_to_obstacle


                # Deviation calculation
                distance_to_path = get_err_position(dwa.x, dwa.y)

                # if len(planned_path) < 2:
                #     distance_to_path = 0.0

                # else:

                #     for k2 in range(min(len(planned_path), 29)):
                #         pose = planned_path[k2]

                #         error = linear_distance(
                #             dwa.x, pose.pose.position.x, dwa.y, pose.pose.position.y)

                #         if error < distance_to_path:
                #             distance_to_path = error

                #     if distance_to_path > max_deviation_from_path:
                #         max_deviation_from_path = distance_to_path

                # Summation of relative distance
                # sum_relative_distance = np.sum(relative_distance_matrix)

                # l_gmm and r_gmm. a and b used here
                # sum_cluster_size = np.sum(gmm_covariance_matrix)

                # Calculation of cost function

                if (np.abs(v) > 0.26) or (np.abs(a) > 1.82):
                    speed_flag = False

                if (lidar_safety_flag == True) and (speed_flag == True):
                    cost_function = cost_function_calculation(distance_to_goal, clearance, distance_to_path, speed_diff, 0.0, 0.0)
                else:
                    cost_function = np.inf

                if cost_function < cost_function_gmm_cluster[i_gmm]:
                    cost_function_gmm_cluster[i_gmm] = cost_function

                    optimal_v[i_gmm] = v
                    optimal_a[i_gmm] = a

    # Now it's the weighted average of optimal output
    for i in range(n_gmm):
        final_optimal_v = optimal_v[i] * gmm_weight_matrix[i]
        final_optimal_a = optimal_a[i] * gmm_weight_matrix[i]

    # max_cost_function_index = cost_function_gmm_cluster.index(max(cost_function_gmm_cluster))
    # max_cost_function_index = cost_function_gmm_cluster.where(cost_function_gmm_cluster == )


    # final_optimal_v = optimal_v[max_cost_function_index]
    # final_optimal_a = optimal_a[max_cost_function_index]

    # How to determine if the navigation is over?
    # I think AMCL position is better.
    for i in range(n_gmm):
        if linear_distance(gmm_mean_matrix[0][i], goal.position.x, gmm_mean_matrix[1][i], goal.position.y) < 0.1:
            path_following_finish = True
            rospy.loginfo('Path following finished successfully. ')

    if path_following_finish is False:
        robot_control(final_optimal_v, final_optimal_a)
    else:
        pass

    (previous_v, previous_a) = (final_optimal_v, final_optimal_a)

    path_following_finish_time = time.time()
    rospy.loginfo('Path following calculation time: ' + str(path_following_finish_time - start_time))


def initial_rotation(original_heading):
    global initial_rotation_finish

    (optimal_v, optimal_a) = (0.0, 0.0)

    # # When distance is very small,  not turning to save time.
    # if linear_distance(original_x, goal_x, original_y, goal_y) < 2.0:
    #     initial_rotation_finish = True

    follow_heading = goal_heading_angle

    # if len(planned_path) < 5:
    #     follow_heading = goal_heading_angle
    # else:
    #     follow_heading = atan2_customized(
    #         planned_path[4].pose.position.y - planned_path[0].pose.position.y, planned_path[4].pose.position.x - planned_path[0].pose.position.x)

    heading_difference = original_heading - follow_heading

    if heading_difference > np.pi:
        heading_difference = heading_difference - 2.0 * np.pi

    elif heading_difference < -np.pi:
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

    heading_difference = original_heading - goal_heading_angle

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
    global initial_rotation_finish, path_following_finish, final_rotation_finish
    # global costmap

    original_z = amcl_pose.pose.pose.orientation.z
    original_w = amcl_pose.pose.pose.orientation.w
    original_heading = quaternion_to_rad(original_z, original_w)

    if initial_rotation_finish is False:
        # initial_rotation(original_x, original_y, original_heading)
        initial_rotation(original_heading)

    elif path_following_finish is False:
        path_following(original_heading)

    elif final_rotation_finish is False:
        final_rotation(original_heading)

    else:
        pass


def control_with_no_gmm():
    rospy.loginfo('Controlling the robot without AMCL result. ')

    robot_control(0.01, 0)


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


# def callback_odom(data):
#     global odom
#     odom = data


def callback_gazebo_odom(data):
    global gazebo_odom
    gazebo_odom = data


# What shall we do here? 
# Or we send the path as topic. 
def callback_path(data):
    global planned_path
    planned_path = data.poses


def callback_goal(data):
    global initial_rotation_finish, path_following_finish, final_rotation_finish
    global goal, goal_heading_angle

    goal = data.pose

    goal_heading_angle = quaternion_to_rad(
        data.pose.orientation.z, data.pose.orientation.w)

    # rospy.loginfo('The goal is set. ')

    initial_rotation_finish = False
    path_following_finish = False
    final_rotation_finish = False


def callback_costmap(data):
    global costmap

    costmap = data.data


def callback_laser_scan(data):
    global laser_scan

    laser_scan = data.ranges[::lidar_step]

    # rospy.loginfo(len(laser_scan))


def control():
    r = rospy.Rate(5)

    while not rospy.is_shutdown():
        rospy.loginfo('running... ')

        start_time = time.time()

        # if planned_path is None:
        #     rospy.loginfo('Waiting for path...')
        #     pass

        # else:
        if gmm_info is False:
            control_with_no_gmm()

        else:
            control_with_gmm()

        end_time = time.time()

        # rospy.loginfo('Runtime of the program is ' + str(end_time - start_time))

        r.sleep()


# Main function
if __name__ == '__main__':

    rospy.init_node('gmm_controller', anonymous=True)

    rospy.Subscriber('gmm_mean', Float64MultiArray, callback_gmm_mean)
    rospy.Subscriber('gmm_covar', Float64MultiArray, callback_gmm_covar)
    rospy.Subscriber('gmm_weight', Float64MultiArray, callback_gmm_weight)

    sub_amcl_pose = rospy.Subscriber(
        'amcl_pose', PoseWithCovarianceStamped, callback_amcl_pose)

    # sub_odom = rospy.Subscriber('odom', Odometry, callback_odom)

    sub_gazebo_odom = rospy.Subscriber('gazebo_odom', Odometry, callback_gazebo_odom)

    sub_path = rospy.Subscriber('move_base/NavfnROS/plan', Path, callback_path)

    sub_goal = rospy.Subscriber(
        '/move_base_simple/goal', PoseStamped, callback_goal)

    sub_costmap = rospy.Subscriber(
        '/move_base/local_costmap/costmap', OccupancyGrid, callback_costmap)

    sub_laser_scan = rospy.Subscriber('/scan', LaserScan, callback_laser_scan)

    control_thread = threading.Thread(target=control)
    control_thread.start()

    # calculation_thread = threading.Thread(target=control)
    # calculation_thread.start()

    rospy.spin()
