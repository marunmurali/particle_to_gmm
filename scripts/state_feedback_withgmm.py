#!/usr/bin/env python3

# A state feedback demo in the simulation environment of Turtlebot 3. 

# The publish (output): 

import math
# from cmath import sqrt, tan
# from numpy.core.fromnumeric import mean
import rospy
import time
import numpy as np
from sklearn import mixture
#from sklearn.mixture import BayesianGaussianMixture
# from std_msgs.msg import String
from std_msgs.msg import MultiArrayDimension
from std_msgs.msg import (Float32MultiArray, Float64MultiArray,
                          Int8MultiArray, Int16MultiArray,
                          Int32MultiArray, Int64MultiArray,
                          UInt8MultiArray, UInt16MultiArray,
                          UInt32MultiArray, UInt64MultiArray)
# from geometry_msgs.msg import PoseArray
from geometry_msgs.msg import PoseArray
from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseWithCovarianceStamped
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from functools import partial

# import matplotlib as mpl
# mpl.switch_backend('agg') 
# import matplotlib.pyplot as plt

orient_x = rospy.get_param('orient_x_param')
orient_y = rospy.get_param('orient_y_param')
goal_x = rospy.get_param('goal_x_param')
goal_y = rospy.get_param('goal_y_param')

# using_gmm = rospy.get_param('using_gmm_param')

k = (goal_y - orient_y) / (goal_x - orient_x)
b = goal_y - k * goal_x

k1 = 0.5
k2 = -0.5

orient_ref =  -math.atan2(goal_x - orient_x, goal_y - orient_y)

mean = None
covariance = None
weight = None

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


def get_err_x(x, y):

    d = (y - k * x - b) / math.sqrt(1 + math.pow(k, 2))
    
    rospy.loginfo('linear_error = ' + str(d))
    
    return d

def get_err_orient(z, w):
    
    # angle: 0 being upwards, larger by ccw, in radius

    theta = 2 * math.atan(z / w) - math.pi / 2
    
    err = theta - orient_ref

    if err < -math.pi: 
        err += math.pi
    if err > math.pi: 
        err -= math.pi

    rospy.loginfo('angular_error = ' + str(err))

    return err


def callback_gmm_mean(data): 
    # Do we have to clear all variables one more time? 
    # mean = None
    # covariance = None
    mean = to_numpy_f64(data)

def callback_gmm_covar(data): 
    covariance = to_numpy_f64(data)

def callback_gmm_weight(data): 
    weight = to_numpy_f64(data)

def callback_odom(data): 
    start = time.time()

    speed_cmd = 0.0
    angular_cmd = 0.0

    for (m, covar, wt) in enumerate(zip(mean, covariance, weight)):
        v, w = np.linalg.eigh(covar)
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        u = w[0] / np.linalg.norm(w[0])

        angle = np.arctan(u[1] / u[0])
        angle = 180.0 * angle / np.pi  # convert to degrees

        x = m[0]
        y = m[1] 
        orient_z = data.pose.pose.orientation.z
        orient_w = data.pose.pose.orientation.w

        x_err = get_err_x(x, y)

        orient_err = get_err_orient(orient_z, orient_w) 

        angular_cmd += wt * (k1 * x_err + k2 * orient_err)

    rospy.loginfo('angular command: ' + str(angular_cmd))

    # rospy.loginfo('x coordinate: ' + str(x) + '; y coordinate: ' + str(y))

    cmd_vel_msg = Twist()

    if (y - (-1 / k) * x - (orient_y - (-1 / k) * orient_x)) > 0:
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

    # total time taken
    end = time.time()
    print('Runtime of the program is %f' %(end - start))

def my_node():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.

    rospy.Subscriber('gmm_mean', Float64MultiArray, callback_gmm_mean)
    rospy.Subscriber('gmm_covar', Float64MultiArray, callback_gmm_covar)
    rospy.Subscriber('gmm_weight', Float64MultiArray, callback_gmm_weight)
    rospy.Subscriber('odom', Odometry, callback_odom)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':

    rospy.init_node('state_feedback', anonymous=True)
    rate = rospy.Rate(5) # ROS Rate at 5Hz

    my_node()
