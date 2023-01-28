#!/usr/bin/env python3
# coding: utf-8

# Definition: A fully virtual experiment
#
# Date of programming: 2023/1/24
# Current progress: N
# A (working with a solid theoretical base) / B (seems to be working) / C (working with problems)
# F (totally not working) / N (on progress)

# The robot is moving from point(0, 0) to point(20, 0) (m) and the speed is set constant as 0.25 m/s
# Only state feedback controller is used

# LIBRARIES
# import random
import time
import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import rospy
from geometry_msgs.msg import (Twist, Point, Pose, PoseStamped, PoseWithCovarianceStamped, PoseArray)

# CONSTANTS

LIN_ERR_COEFF = -0.1
ANG_ERR_COEFF = -0.1

NEAR_CLUSTER_DEVIATION = 0.1
FAR_CLUSTER_DEVIATION = 1.0

# NEAR_CLUSTER_DEVIATION = 0.0
# FAR_CLUSTER_DEVIATION = 0.0

START = Point(0.0, 0.0, 0.0)
GOAL = Point(20.0, 0.0, 0.0)

T_INTERVAL = 0.1

# N_GMM = 3

SPEED = 0.25


# GLOBAL VARIABLES
# From top to bottom: AMCL, GMM_flat, GMM_square,
# path_x = np.zeros((4, 1))
# path_y = np.zeros((4, 1))

# error = np.zeros((4, 1))


# FUNCTIONS

# (ok)Refreshing states
def localization_refresh(position_ground_truth): 
    c = np.zeros((2, 3))

    for i in range(2):
        c[i, 0] = position_ground_truth[i] + NEAR_CLUSTER_DEVIATION * np.random.randn()
    
    for n_clu in range(1, 3): 
        for i in range(2): 
            c[i, n_clu] = position_ground_truth[i] + FAR_CLUSTER_DEVIATION * np.random.randn()

    w = np.zeros(3)

    w[0] = 0.667 * np.random.rand() + 0.333
    w[1] = (1 - w[0]) * np.random.rand()
    w[2] = 1 - w[0] - w[1]

    a = np.zeros(2)

    for n_clu in range(3): 
        for i in range(2): 
            a[i] += w[n_clu] * c[i, n_clu]

    return (c, w, a)

# Simulation function
def simulation(gmm, gmm_mode="square"): 
    position_ground_truth = np.zeros(2)
    orientation_ground_truth = 0.0

    clusters_mean = np.zeros((2, 3))
    AMCL_mean = np.zeros(2)
    weights = np.zeros(3)

    x = np.ndarray([0])
    y = np.ndarray([0])

    (clusters_mean, weights, AMCL_mean) = localization_refresh(position_ground_truth)

    path_following_finish = False

    t = 0.0

    step = 0

    while not path_following_finish: 
        step += 1

        # Calculation of angular speed command
        if gmm: 
            ang_speed = 0.0
            
            if gmm_mode == "square": 
                squared_weights = np.square(weights)
                sum_squared_weights = np.sum(squared_weights)

                for n_cls in range(3): 
                    ang_speed += (clusters_mean[1][n_cls] * LIN_ERR_COEFF + orientation_ground_truth * ANG_ERR_COEFF) * squared_weights[n_cls]
            
            elif gmm_mode == "max": 
                n_max = np.argmax(weights)
                ang_speed += clusters_mean[1][n_max] * LIN_ERR_COEFF + orientation_ground_truth * ANG_ERR_COEFF

            # Flat as default
            else: 
                for n_cls in range(3): 
                    ang_speed += (clusters_mean[1][n_cls] * LIN_ERR_COEFF + orientation_ground_truth * ANG_ERR_COEFF) * weights[n_cls]
            

        else: 
            ang_speed = AMCL_mean[1] * LIN_ERR_COEFF + orientation_ground_truth * ANG_ERR_COEFF

        # Kinematics
        for i in range(10): 
            # Ground truth
            position_ground_truth[0] += SPEED * 0.1 * T_INTERVAL * math.cos(orientation_ground_truth)
            position_ground_truth[1] += SPEED * 0.1 * T_INTERVAL * math.sin(orientation_ground_truth)

            # Virtual clusters
            for n_cls in range(3): 
                clusters_mean[0][n_cls] += SPEED * 0.1 * T_INTERVAL * math.cos(orientation_ground_truth)
                clusters_mean[1][n_cls] += SPEED * 0.1 * T_INTERVAL * math.sin(orientation_ground_truth)

            AMCL_mean[0] += SPEED * 0.1 * T_INTERVAL * math.cos(orientation_ground_truth)
            AMCL_mean[1] += SPEED * 0.1 * T_INTERVAL * math.sin(orientation_ground_truth)

            orientation_ground_truth += 0.1 * T_INTERVAL * ang_speed

        if position_ground_truth[0] > GOAL.x: 
            path_following_finish = True
        
        if np.mod(step, 10) == 0: 
            (clusters_mean, weights, AMCL_mean) = localization_refresh(position_ground_truth)

        x = np.append(x, position_ground_truth[0])
        y = np.append(y, position_ground_truth[1])

        t += T_INTERVAL

        # print("x = " + str(position_ground_truth[0]) + ", y = " + str(position_ground_truth[1]) + ", t = " + str(t))

    return (x, y, t)

def plot(): 
    pass

# Main function
def main(): 
    np.random.seed(seed=int(time.time())) 

    (x_gmm_flat, y_gmm_flat, t_gmm_flat) = simulation(True, "flat")
    (x_gmm_square, y_gmm_square, t_gmm_square) = simulation(True, "square")
    (x_gmm_max, y_gmm_max, t_gmm_max) = simulation(True, "max")

    (x_amcl, y_amcl, t_amcl) = simulation(False)

    mse_gmm_flat = np.average(np.square(y_gmm_flat))
    mse_gmm_square = np.average(np.square(y_gmm_square))
    mse_gmm_max = np.average(np.square(y_gmm_max))
    mse_amcl = np.average(np.square(y_amcl))

    print("GMM normal: " + str(mse_gmm_flat))
    print("GMM squared: " + str(mse_gmm_square))
    print("GMM max: " + str(mse_gmm_max))
    print("AMCL: " + str(mse_amcl))

    plt.rcParams['figure.figsize'] = 10, 6.5
    plt.rcParams['font.size'] = 20

    fig, ax = plt.subplots(constrained_layout=True)

    ax.set_xlim(0, 20)
    ax.set_ylim(-0.5, 0.5)

    ax.plot(x_gmm_flat, y_gmm_flat, label="GMM_nm", color='r', linestyle ="dashed", lw=2)
    ax.plot(x_gmm_square, y_gmm_square, label="GMM_sq", color='g', linestyle ="dashed", lw=2)
    ax.plot(x_gmm_max, y_gmm_max, label="GMM_mx", color='b', linestyle="dashed", lw=2)
    ax.plot(x_amcl, y_amcl, label="nonGMM", color='k', lw=2)

    ax.plot([0, 20], [0, 0], color='k', linestyle="dotted")

    ax.legend()

    ax.set_title("Paths by different controllers following path on X-axis", fontsize=20)

    plt.show()


if __name__ == "__main__":
    main()





