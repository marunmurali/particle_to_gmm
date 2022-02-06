#!/usr/bin/env python3
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
#
# Revision $Id$

## Simple talker demo that listens to std_msgs/Strings published
## to the 'chatter' topic

import math
# from cmath import sqrt, tan
from numpy.core.fromnumeric import mean
import rospy
import time
import numpy as np
from sklearn import mixture
#from sklearn.mixture import BayesianGaussianMixture
from std_msgs.msg import String
from std_msgs.msg import MultiArrayDimension
from std_msgs.msg import (Float32MultiArray, Float64MultiArray,
                          Int8MultiArray, Int16MultiArray,
                          Int32MultiArray, Int64MultiArray,
                          UInt8MultiArray, UInt16MultiArray,
                          UInt32MultiArray, UInt64MultiArray)
# from geometry_msgs.msg import PoseArray
from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseWithCovarianceStamped
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from functools import partial

k = -15.51428899
b = -457.9540552

x1 = -29.0157584068721
y1 = -7.79519393496102
x2 = -29.8366912334359
y2 = 4.94099518011244


def get_err_x(global_x, global_y):

    rospy.loginfo(global_x)
    rospy.loginfo(global_y)    
    d = (global_y - k * global_x - b) / math.sqrt(1 + math.pow(k, 2))

    rospy.loginfo(d)

    return d

def get_err_orient(global_theta):
    
    theta_ref = 3 * math.pi / 2 - math.atan2(y2 - y1, x2 - x1)
    
    err = global_theta - theta_ref

    return err


def callback(data): 
    pubCmd = rospy.Publisher('cmd_vel', Twist, queue_size=10) 

    start = time.time()


    x = data.pose.pose.position.x
    y = data.pose.pose.position.y

    orient = data.twist.twist.angular.z

    k1 = 0.01
    k2 = -0.1

    
    x_err = get_err_x(x, y)

    orient_err = get_err_orient(orient)
    angular_cmd = k1 * x_err + k2 * orient_err
    rospy.loginfo(angular_cmd)

    # rospy.loginfo(rospy.get_caller_id() + 'Number of particles %d', len(data.poses))

    cmd_vel_msg = Twist()

    cmd_vel_msg.linear.x = 1.0
    cmd_vel_msg.linear.y = 0.0    
    cmd_vel_msg.linear.z = 0.0

    cmd_vel_msg.angular.x = 0.0
    cmd_vel_msg.angular.y = 0.0
    cmd_vel_msg.angular.z = angular_cmd
    
    # initialize empty list
    # PoseForGmmArr = []
    # gmm_mean = Float64MultiArray()
    # gmm_covar = Float64MultiArray()
    # gmm_mean = []
    # gmm_covar = []
    # for i in range(len(data.poses)):
    #     newrow = ([data.poses[i].position.x, data.poses[i].position.y])
    #     PoseForGmmArr.append(newrow)
    # PoseForGmm = np.array(PoseForGmmArr)
    #print(PoseForGmm.shape)
    #dpgmm = mixture.BayesianGaussianMixture(n_components=5, covariance_type="full").fit(PoseForGmm)
    #dpgmm = BayesianGaussianMixture(n_components=5, covariance_type="full").fit(PoseForGmm)
    # dpgmm = mixture.GaussianMixture(n_components=5, covariance_type="full").fit(PoseForGmm)

    # gmm_mean = toMultiArray(dpgmm.means_)
    # gmm_covar = toMultiArray(dpgmm.covariances_)
    # gmm_mean = to_multiarray_f64(dpgmm.means_)
    # gmm_covar = to_multiarray_f64(dpgmm.covariances_)
    pubCmd.publish(cmd_vel_msg)
    # pubCovar.publish(gmm_covar)

    # total time taken
    end = time.time()
    print("Runtime of the program is %f" %(end - start))

def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('state_feedback', anonymous=True)

    rospy.Subscriber('odom', Odometry, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()
