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

# import itertools
import rospy
import time
import math
import numpy as np
from numpy.core.fromnumeric import mean
from numpy.ma.core import concatenate
from numpy import linalg
from sklearn import mixture
from functools import partial
from scripts.ref2 import controller
#from sklearn.mixture import BayesianGaussianMixture

# ROS libraries
from std_msgs.msg import String
from std_msgs.msg import MultiArrayDimension
from std_msgs.msg import (Float32MultiArray, Float64MultiArray,
                          Int8MultiArray, Int16MultiArray,
                          Int32MultiArray, Int64MultiArray,
                          UInt8MultiArray, UInt16MultiArray,
                          UInt32MultiArray, UInt64MultiArray)

from geometry_msgs.msg import PoseArray
from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseWithCovarianceStamped
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

# import matplotlib as mpl
# import matplotlib.pyplot as plt

orient_x = rospy.get_param('orient_x')
orient_y = rospy.get_param('orient_y')
goal_x = rospy.get_param('goal_x')
goal_y = rospy.get_param('goal_y')

k = (goal_y - orient_y) / (goal_x - orient_x)
b = orient_y - k * orient_x

orient_ref = -np.arctan2(goal_x - orient_x, goal_y - orient_y)

# k1 = -1.0
k2 = -0.50
k3 = 0.1
k4 = -0.1

# Data conversion methods
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

# Pose calculation methods
def get_err_position(x, y):

    d = (y - k * x - b) / math.sqrt(1 + math.pow(k, 2))
    
    rospy.loginfo('linear_error = ' + str(d))
    
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
def controller(means, covariances, weights, odom): 

    pubCmd = rospy.Publisher('cmd_vel', Twist, queue_size=10) 

    linear_cmd = 0.0
    angular_cmd = 0.0

    orient_z = odom.pose.pose.orientation.z
    orient_w = odom.pose.pose.orientation.w

    theta = 2 * np.arctan(orient_z / orient_w) - np.pi / 2

    if theta < -np.pi: 
        theta = theta + 2.0 * np.pi

    orientation_err = get_err_orient(theta)

    for (m, covar) in enumerate(zip(means, covariances)):
    # for (m, covar, weight) in enumerate(zip(means, covariances, weights)):
        eig_val, eig_vec = linalg.eigh(covar)

        v = 2.0 * np.sqrt(5.991) * np.sqrt(eig_val)
        u = eig_vec[0] / linalg.norm(eig_vec[0])
        

        angle = np.arctan(u[1] / u[0]) + 3 * np.pi / 2

        if angle > 2 * np.pi: 
            angle = angle - 2 * np.pi

        # angle = 180.0 * angle / np.pi  # convert to degrees

        x = m[0]
        y = m[1]

        b = v[0]
        a = v[1]

        x_err = get_err_position(x, y)

        # On the direction of path
        l1 = np.abs(b / 2 * np.cos(angle - orient_ref)) + np.abs(a / 2 * np.sin(angle - orient_ref))

        # On the perpendicular direction
        l2 = np.abs(b / 2 * np.sin(angle - orient_ref)) + np.abs(a / 2 * np.cos(angle - orient_ref))

        if ((-np.sin(theta) * (-np.sin(orient_ref)) + np.cos(theta) * np.cos(orient_ref)) 
                * (goal_x - orient_x)) >= 0: 
            k1 = -0.50
        else: 
            k1 = 0.50

        cmd3 = k3 * l1

        if np.abs(cmd3) > 1.0: 
            cmd3 = cmd3 / np.abs(cmd3)

        cmd4 = k4 * l2

        if np.abs(cmd4) > 1.0: 
            cmd4 = cmd4 / np.abs(cmd4)


        angular_cmd += 1 / 3 * (k1 * x_err + k2 * orientation_err)

        linear_cmd += 1 / 3 * (cmd3 + cmd4)

    rospy.loginfo('angular command: ' + str(angular_cmd))

    # rospy.loginfo('x coordinate: ' + str(x) + '; y coordinate: ' + str(y))

    cmd_vel_msg = Twist()

    cmd_vel_msg.linear.x = 0.20 + linear_cmd
        
    cmd_vel_msg.linear.y = 0.0    
    cmd_vel_msg.linear.z = 0.0

    cmd_vel_msg.angular.x = 0.0
    cmd_vel_msg.angular.y = 0.0
    cmd_vel_msg.angular.z = angular_cmd
    
    pubCmd.publish(cmd_vel_msg)
    # pubCovar.publish(gmm_covar)

class MyNode:

    # Initializer
    def __init__(self):     

        rospy.init_node('gmm_controller', anonymous=True)

        r = rospy.Rate(3)

        self.mean = None
        self.covariance = None
        self.weight = None
        self.odom = Odometry()

        start = time.time()

        rospy.Subscriber('gmm_mean', Float64MultiArray, self.callback_gmm_mean)
        rospy.Subscriber('gmm_covar', Float64MultiArray, self.callback_gmm_covar)
        # rospy.Subscriber('gmm_weight', Float64MultiArray, self.callback_gmm_weight)
        rospy.Subscriber('odom', Odometry, self.callback_odom)

        # total time taken
        end = time.time()
        print('Runtime of the program is %f' %(end - start))

        r.sleep()

    def callback_gmm_mean(self,data):
        self.mean = to_numpy_f64(data)

    def callback_gmm_covar(self,data):
        self.covariance = to_numpy_f64(data)

        # rowSize = data.layout.dim[0].size
        # tempArray = []
        # for i in range(rowSize):
        #     if len(tempArray) == 0:
        #         tempArray = [[data.data[i*4], data.data[i*4+1]], [data.data[i*4+2], data.data[i*4+3]]]
        #         print(np.shape(tempArray))
        #     else:
        #         tempArray =np.stack([tempArray, [[data.data[i*4], data.data[i*4+1]], [data.data[i*4+2], data.data[i*4+3]]]], axis=0)
        # self.covariance = np.array(tempArray)
        # print(self.covariance)
        # print(np.shape(self.covariance))

    def callback_gmm_weight(self, data): 
        self.weight = to_numpy_f64(data)

    def callback_odom(self, data): 
        self.odom = data; 

        self.control()

# Main method
if __name__ == '__main__':

    # rospy.init_node('state_feedback', anonymous=True)
    # rate = rospy.Rate(5) # ROS Rate at 5Hz

    try: 
        controller()
    except rospy.ROSInitException: 
        pass