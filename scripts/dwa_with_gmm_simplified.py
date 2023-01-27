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
# import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import statistics

## ROS libraries
from std_msgs.msg import (MultiArrayDimension, Float64MultiArray)
from geometry_msgs.msg import (
    Twist, Point, Pose, PoseStamped, PoseWithCovarianceStamped, PoseArray)
from nav_msgs.msg import (Odometry, Path, OccupancyGrid)
from sensor_msgs.msg import LaserScan


# Global variables

## Information of goal set in RViz
start_point = Point()
start_point.x= rospy.get_param('orient_x')
start_point.y= rospy.get_param('orient_y')

goal = Point()
goal.x = rospy.get_param('goal_x')
goal.y = rospy.get_param('goal_y')

# Here we suppose that k doesn't equal to 0 or inf.  

## ROS parameters
gmm_flag = rospy.get_param('gmm')
n_gmm = rospy.get_param('num_of_gmm_dist')
swflag = rospy.get_param("squared_weight")

## DWA parameters
dwa_horizon_param = 20

## Spec of lidar
lidar_range_min = 0.16
lidar_range_max = 5.0

## Lidar data step length (equal to degree), for saving computational time:
lidar_step = 30

## Control time interval
t_interval = 0.1

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
# n_laser_scan = 360

## Odometry of the robot
odom = Odometry()
gazebo_odom = Odometry()

## Error message (for plotting)
# error_msg = Point()

## Control states
initial_rotation_finish = False
path_following_finish = False
final_rotation_finish = False
init = False
calc = False
path_following_start_time = None

## Investigating time of GMM and DWA
gmm_time = None
# dwa_time = None

# (k, b) = (0.0, 0.0)


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

final_optimal_v = 0.0
final_optimal_a = 0.0

## DWA cost function coefficients
alpha = np.zeros(7)
# Distance to the goal
# Minimal distance to the obstacle
# Max deviation from the path
# Speed difference
# Relative distance of clusters * Not included now
# Cluster size * Not included now
# Angular speed
alpha[0] = 1.0
alpha[1] = -0.1
alpha[2] = 2.0
alpha[3] = 10.0
# alpha[4] = 1.0
# alpha[5] = 1.0
alpha[6] = 10.0

## Speed command publisher
pubCmd = rospy.Publisher('cmd_vel', Twist, queue_size=10)
cmd_vel_msg = Twist()

## Error data
error_plt = np.empty(0)
speed_plt = np.empty(0)
a_speed_plt = np.empty(0)
t_plt = np.empty(0)
squared_error = np.empty(0)

plot_finish = False


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
    return np.array(multiarray.data, dtype=pytype).reshape(dims).astype(dtype)


to_multiarray_f64 = partial(_numpy_to_multiarray, Float64MultiArray)
to_numpy_f64 = partial(_multiarray_to_numpy, float, np.float64)


def gmm_process():
    global gmm_mean_matrix, gmm_weight_matrix

    global gmm_info, gmm_time

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
            # gmm_covariance_matrix[0][i] = v[0]
            # gmm_covariance_matrix[1][i] = v[1]

            gmm_weight_matrix[i] = weight
        

        # Calculation of relative distance of GMM clusters
        # for i in range(n_gmm):
        #     for j in range(1, n_gmm):
        #         relative_distance_matrix[i][j] = linear_distance(
        #             gmm_mean_matrix[0][i], gmm_mean_matrix[0][j], gmm_mean_matrix[1][i], gmm_mean_matrix[1][j])

        gmm_time = time.time()

        if gmm_info is False:
            gmm_info = True


def robot_control(v, a):
    # Initialize command publisher

    global pubCmd, cmd_vel_msg

    cmd_vel_msg.linear.x = v
    cmd_vel_msg.linear.y = 0.0
    cmd_vel_msg.linear.z = 0.0

    cmd_vel_msg.angular.x = 0.0
    cmd_vel_msg.angular.y = 0.0
    cmd_vel_msg.angular.z = a

    rospy.loginfo('v: ' + str(v))
    rospy.loginfo('a: ' + str(a))
    pubCmd.publish(cmd_vel_msg)


def cost_function_calculation(dis_goal, min_dis_obs, max_dev, spd_diff, angular_speed):

    j = np.zeros(7)

    # Distance to the goal
    j[0] = alpha[0] * np.power(dis_goal, 1)

    # Minimal distance to the obstacle
    j[1] = alpha[1] * np.power(min_dis_obs, 1)
    # Max deviation from the path
    j[2] = alpha[2] * np.power(max_dev, 1)
    # Speed difference
    j[3] = alpha[3] * np.power(spd_diff, 2)
    # Relative distance of clusters *Not included now
    j[4] = 0.0
    # Cluster size *Not included now
    j[5] = 0.0
    # Angular speed
    j[6] = alpha[6] * np.power(angular_speed, 2)

    # rospy.loginfo('distance to goal: ' + str(j[0]))
    # rospy.loginfo('min distance to obstacle: ' + str(j[1]))
    # rospy.loginfo('max deviation: ' + str(j[2]))
    # rospy.loginfo('speed difference: ' + str(j[3]))
    # rospy.loginfo('angular speed: ' + str(j[6]))

    # rospy.loginfo('Total cost function: ' + str(np.sum(j)))

    return np.sum(j)


# The path following function
def path_following(original_heading):
    global path_following_finish, previous_v, previous_a, cost_function_gmm_cluster

    global squared_error, error_msg, error_plt, speed_plt, a_speed_plt, t_plt

    global final_optimal_v, final_optimal_a

    start_time = time.time()

    lidar_safety_flag = True
    speed_flag = True

    optimal_v = np.zeros(10)
    optimal_a = np.zeros(10)

    # rospy.loginfo('Time difference between path following and GMM: ' + str(start_time - gmm_time))
    
    for i_gmm in range(n_gmm): 
        cost_function_gmm_cluster[i_gmm] = np.inf

    # cost_function = np.zeros(10)

    # laser_scan_x = np.zeros((10, 360))
    # laser_scan_y = np.zeros((10, 360))

    # Velocity space with dynamic constraints
    v_range = np.array([-0.05, -0.04, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05]) + previous_v
    a_range = np.array([-0.05, -0.04, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05]) + previous_a

    # Why convert lidar data to global coordinate? It's unnecessary.
    # 2023/1/13 Repaired a major problem
    for i in range(len(laser_scan)):
        laser_scan_theta = np.pi / 180.0 * i * lidar_step
        laser_scan_coordinate[0][i] = -laser_scan[i] * np.sin(laser_scan_theta)
        laser_scan_coordinate[1][i] = laser_scan[i] * np.cos(laser_scan_theta)

    for i in range(len(v_range)):

        v = v_range[i]

        speed_diff = v - previous_v

        for j in range(len(a_range)):

            a = a_range[j]

            # Computation of cost function

            for i_gmm in range(n_gmm):
                current_x = gmm_mean_matrix[0][i_gmm]
                current_y = gmm_mean_matrix[1][i_gmm]

                # Including the following terms
                # min_distance_to_obstacle = np.inf
                # replaced by "clearance"
                # max_deviation_from_path = 0.0
                # replaced by "distance_to_path"

                # total_relative_distance = 0.0
                # total_gmm_cluster_size = 0.0

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
                distance_to_goal = linear_distance(goal.x, dwa.x, goal.y, dwa.y)

                # Clearance calculation
                clearance = np.inf

                # i_angle = np.round(atan2_customized(dwa_local.y, dwa_local.x) / lidar_step)

                # # Limitation processing
                # if i_angle == len(laser_scan):
                #     i_angle = 0

                for i_scan in range(len(laser_scan)): 
                    distance_to_obstacle = linear_distance(dwa_local.x, laser_scan_coordinate[0][i_scan], dwa_local.y, laser_scan_coordinate[1][i_scan])
                    if distance_to_obstacle < clearance: 
                        clearance = distance_to_obstacle

                # rospy.loginfo('clearance' + str(clearance))

                if clearance == np.inf: 
                    clearance = 5.0

                # Deviation calculation
                distance_to_path = np.abs(get_err_position(dwa.x, dwa.y))

                # Calculation of cost function

                if (np.abs(v) > 0.26) or (np.abs(a) > 1.82):
                    speed_flag = False

                if (lidar_safety_flag is True) and (speed_flag is True): 
                    # cost_function = cost_function_calculation(distance_to_goal, 0.0, distance_to_path, speed_diff, a)
                    cost_function = cost_function_calculation(distance_to_goal, clearance, distance_to_path, speed_diff, a)
                    # cost_function = cost_function_calculation(distance_to_goal, 0.0, 0.0, 0.0, 0.0)


                else:
                    cost_function = np.inf

                # rospy.loginfo('i_gmm = ' + str(i_gmm))

                # rospy.loginfo('v: ' + str(v))
                # rospy.loginfo('a: ' + str(a))

                # rospy.loginfo('cost function = ' + str(cost_function))

                if cost_function < cost_function_gmm_cluster[i_gmm]:
                    cost_function_gmm_cluster[i_gmm] = cost_function

                    optimal_v[i_gmm] = v
                    optimal_a[i_gmm] = a

    # Now it's the weighted average of optimal output

    (final_optimal_v, final_optimal_a) = (0.0, 0.0)
    
    if swflag: 
    
        sum_squared_weight = 0.0
        

        for i in range(n_gmm):
            if cost_function_gmm_cluster[i] < np.inf: 
                sw = math.pow(gmm_weight_matrix[i], 2)
                final_optimal_v += optimal_v[i] * sw
                final_optimal_a += optimal_a[i] * sw

                sum_squared_weight += sw

        final_optimal_v /= sum_squared_weight
        final_optimal_a /= sum_squared_weight

    else: 
        sum_weight = 0.0

        for i in range(n_gmm):
            if cost_function_gmm_cluster[i] < np.inf: 
                w = gmm_weight_matrix[i]
                final_optimal_v += optimal_v[i] * w
                final_optimal_a += optimal_a[i] * w

                sum_weight += w

        final_optimal_v /= sum_weight
        final_optimal_a /= sum_weight

    if np.abs(v) > 0.26:
        v = v / np.abs(v) * 0.26

    if np.abs(a) > 1.82: 
        a = a / np.abs(a) * 1.82

    # final_optimal_v = optimal_v[max_cost_function_index]
    # final_optimal_a = optimal_a[max_cost_function_index]

    # Finding the largest cost function among clusters

    # How to determine if the navigation is over?
    # Implemented method based on vector multiplication
    for i in range(n_gmm):
        # if linear_distance(gmm_mean_matrix[0][i], goal.x, gmm_mean_matrix[1][i], goal.y) < 0.1:
        #     path_following_finish = True

        d = scalar_product(goal.x - gmm_mean_matrix[0][i], goal.x - start_point.x, goal.y - gmm_mean_matrix[1][i], goal.y - start_point.y) / linear_distance(goal.x, start_point.x, goal.y, start_point.y)

        # rospy.loginfo(str(i) + "scalar product = " + str(d))

        if d < 0.0: 
            path_following_finish = True

    if path_following_finish is False:
        robot_control(final_optimal_v, final_optimal_a)

    else:
        rospy.loginfo('Path following finished. ')

    (previous_v, previous_a) = (final_optimal_v, final_optimal_a)

    path_following_finish_time = time.time()
    
    # rospy.loginfo('Path following calculation time: ' + str(path_following_finish_time - start_time))


def initial_rotation(original_heading):
    global initial_rotation_finish

    (optimal_v, optimal_a) = (0.0, 0.0)

    heading_difference = original_heading - goal_heading_angle

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


def final_calculation(): 
    global error_mse, squared_error, path_following_start_time

    path_following_end_time = time.time()

    # rospy.loginfo(str(path_following_end_time))
    
    error_mse = np.average(squared_error)

    path_following_time = path_following_end_time - path_following_start_time

    rospy.loginfo('Path following time = ' + str(path_following_time) + 's. ')
    rospy.loginfo('MSE of error = ' + str(error_mse) + 'm^2')
    rospy.loginfo('squared: ' + str(swflag))



# Controller method
def control_with_gmm():
    # Global variables
    global initial_rotation_finish, path_following_finish, final_rotation_finish
    global error_mse, path_following_start_time, init, calc
    global previous_v

    original_z = amcl_pose.pose.pose.orientation.z
    original_w = amcl_pose.pose.pose.orientation.w
    original_heading = quaternion_to_rad(original_z, original_w)

    if initial_rotation_finish is False:
        # initial_rotation(original_x, original_y, original_heading)
        if not init: 
            path_following_start_time = time.time()
            rospy.loginfo(str(path_following_start_time))
            init = True

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
    ax.set_title('DWA-with-GMM controller path following', fontsize=14)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))

    ax_sub = ax.secondary_yaxis('right', functions=(sub_conv, sub_conv))
    ax_sub.set_ylabel('speed/(m/s)')
    ax_sub.yaxis.set_major_locator(MultipleLocator(0.1))

    ax_sub2 = ax.secondary_yaxis(1.15, functions=(sub2_conv, sub2_rev))
    ax_sub2.set_ylabel('angular speed/(rad/s)')
    ax_sub2.yaxis.set_major_locator(MultipleLocator(0.02))

    ax.legend()

    plt.show()


def control():
    r_control = rospy.Rate(10)

    while not rospy.is_shutdown():
        if final_rotation_finish is False: 
            start_time = time.time()
            if gmm_info is False:
                control_with_no_gmm()

            else:
                control_with_gmm()

            end_time = time.time()
            # rospy.loginfo('Runtime of the controller program is ' + str(end_time - start_time))
        else: 
            pass

        r_control.sleep()


def measure():
    # error related
    global squared_error, error_plt, speed_plt, a_speed_plt, t_plt, gazebo_odom, cmd_vel_msg

    global path_following_finish, calc, plot_finish

    r_measure = rospy.Rate(10)

    while not rospy.is_shutdown():
        if not path_following_finish: 
            if init: 
                start_time = time.time()

                squared_error = np.append(squared_error, np.power(get_err_position(gazebo_odom.pose.pose.position.x, gazebo_odom.pose.pose.position.y), 2)) 
                speed_plt = np.append(speed_plt, cmd_vel_msg.linear.x)
                a_speed_plt = np.append(a_speed_plt, cmd_vel_msg.angular.z)
                error_plt = np.append(error_plt, get_err_position(gazebo_odom.pose.pose.position.x, gazebo_odom.pose.pose.position.y))
                t_plt = np.append(t_plt, time.time() - path_following_start_time)
                
                end_time = time.time()
                # rospy.loginfo('Runtime of the measurement program is ' + str(end_time - start_time))


        r_measure.sleep()

# Main function
if __name__ == '__main__':
    global goal_heading_angle, k, b

    final = False

    rospy.init_node('gmm_controller', anonymous=True)

    rospy.Subscriber('gmm_mean', Float64MultiArray, callback_gmm_mean)
    rospy.Subscriber('gmm_covar', Float64MultiArray, callback_gmm_covar)
    rospy.Subscriber('gmm_weight', Float64MultiArray, callback_gmm_weight)

    sub_amcl_pose = rospy.Subscriber(
        'amcl_pose', PoseWithCovarianceStamped, callback_amcl_pose)

    # sub_odom = rospy.Subscriber('odom', Odometry, callback_odom)

    sub_gazebo_odom = rospy.Subscriber('gazebo_odom', Odometry, callback_gazebo_odom)

    sub_path = rospy.Subscriber('move_base/NavfnROS/plan', Path, callback_path)

    sub_goal = rospy.Subscriber('/move_base_simple/goal', PoseStamped, callback_goal)

    sub_costmap = rospy.Subscriber('/move_base/local_costmap/costmap', OccupancyGrid, callback_costmap)

    sub_laser_scan = rospy.Subscriber('/scan', LaserScan, callback_laser_scan)

    # slope and intercept of the path
    k = (goal.y - start_point.y) / (goal.x - start_point.x)
    b = start_point.y - k * start_point.x

    goal_heading_angle = atan2_customized(goal.y - start_point.y, goal.x - start_point.x)
    
    control_thread = threading.Thread(target=control)
    control_thread.start()

    measure_thread = threading.Thread(target=measure)
    measure_thread.start()

    while not rospy.is_shutdown():
        if path_following_finish: 
            if not final: 
                final_calculation()
                result_plot()
                final = True
