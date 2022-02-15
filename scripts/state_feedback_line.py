#!/usr/bin/env python3

import math
# from cmath import sqrt, tan
# from numpy.core.fromnumeric import mean
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

x1 = rospy.get_param('~orient_x')
y1 = rospy.get_param('~orient_y')
x2 = rospy.get_param('~goal_x')
y2 = rospy.get_param('~goal_y')

k = (y2 - y1) / (x2 - x1)
b = y2- k * x1

k1 = 0.5
k2 = -0.5


def get_err_x(x, y):

    d = (y - k * x - b) / math.sqrt(1 + math.pow(k, 2))
    
    rospy.loginfo('linear_error = ' + str(d))
    
    return d

def get_err_orient(z, w):
    
    theta_ref =  -math.atan2(x2 - x1, y2 - y1)

    theta = 2 * math.atan(z / w) - math.pi / 2
    
    err = theta - theta_ref

    if err < -math.pi: 
        err += math.pi
    if err > math.pi: 
        err -= math.pi

    rospy.loginfo('angular_error = ' + str(err))

    return err


def callback(data): 
    pubCmd = rospy.Publisher('cmd_vel', Twist, queue_size=10) 

    start = time.time()

    x = data.pose.pose.position.x
    y = data.pose.pose.position.y

    rospy.loginfo('x coordinate: ' + str(x) + '; y coordinate: ' + str(y))

    orient_z = data.pose.pose.orientation.z
    orient_w = data.pose.pose.orientation.w

    x_err = get_err_x(x, y)

    orient_err = get_err_orient(orient_z, orient_w)
    angular_cmd = k1 * x_err + k2 * orient_err
    rospy.loginfo('angular command: ' + str(angular_cmd))

    # rospy.loginfo(rospy.get_caller_id() + 'Number of particles %d', len(data.poses))

    cmd_vel_msg = Twist()

    b2 = y2 - (-1 / k) * x2
    
    if (y - (-1 / k) * x - b2) > 0:
        cmd_vel_msg.linear.x = 0.0
    else: 
        cmd_vel_msg.linear.x = 0.20
        
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

    # prints total time taken
    end = time.time()
    print('Runtime of the program is %f s. ' %(end - start))

def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.

    rospy.Subscriber('odom', Odometry, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':

    rospy.init_node('state_feedback', anonymous=True)
    rate = rospy.Rate(2) # ROS Rate at 2Hz

    listener()
