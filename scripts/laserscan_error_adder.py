#!/usr/bin/env python3
# Software License Agreement (BSD License)
#
# Copyright (c) 2021, Arun Muraleedharan.
# All rights reserved.

# Includes

# from numpy.core.fromnumeric import mean
from turtle import distance
import rospy
import time
import numpy as np

from sensor_msgs.msg import LaserScan
# from std_msgs.msg import String
# from std_msgs.msg import MultiArrayDimension
# from std_msgs.msg import (Float32MultiArray, Float64MultiArray,
#                           Int8MultiArray, Int16MultiArray,
#                           Int32MultiArray, Int64MultiArray,
#                           UInt8MultiArray, UInt16MultiArray,
#                           UInt32MultiArray, UInt64MultiArray)
# from geometry_msgs.msg import PoseArray

# from functools import partial


# Global variables

add_noise = rospy.get_param('lidar_noise')

# add_noise = True


# Methods


def callback(data):
    # start = time.time()

    
    
    pub_scan = rospy.Publisher('new_scan', LaserScan, queue_size=1)

    new_scan = data

    if add_noise and (not (data is None)): 
        new_ranges = list(new_scan.ranges)

        # change from random to Gaussian noise

        # 3% stddev when range <= 6.0m
        # 5% stddev when range > 6.0m

        # for i, distance in enumerate(new_scan.ranges):
        
        for i in range(len(new_ranges)): 

            distance = new_ranges[i]

            if distance < 0.3: 
                error = np.random.normal(0, 0.01 * distance)
            elif distance <= 6.0: 
                error = np.random.normal(0, 0.03 * distance)
            else: 
                error = np.random.normal(0, 0.05 * distance)

            new_ranges[i] = new_ranges[i] + error

        new_scan.ranges = tuple(new_ranges)

    pub_scan.publish(new_scan)

    # total time taken
    # end = time.time()
    # rospy.loginfo("Runtime of GMM publisher is %f" % (end - start))


def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('laserscan_error_adder', anonymous=True)

    rospy.Subscriber('scan', LaserScan, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()


if __name__ == '__main__':
    listener()
