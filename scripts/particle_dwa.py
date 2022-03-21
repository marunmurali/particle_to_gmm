#!/usr/bin/env python3
# Software License Agreement (BSD License)
#
# Copyright (c) 2021~2022, Arun Muraleedharan, Li Hanjie.
# All rights reserved.

# Includes

# from numpy.core.fromnumeric import mean
import threading
import rospy
import time
import numpy as np
# from std_msgs.msg import MultiArrayDimension
# from std_msgs.msg import (Float32MultiArray, Float64MultiArray,
#                           Int8MultiArray, Int16MultiArray,
#                           Int32MultiArray, Int64MultiArray,
#                           UInt8MultiArray, UInt16MultiArray,
#                           UInt32MultiArray, UInt64MultiArray)

# Messages and services
from geometry_msgs.msg import PoseArray, Twist


from std_srvs.srv import Empty

# from functools import partial


# Global variables
amcl_particles = None



# Methods

# def _numpy_to_multiarray(multiarray_type, np_array):
#     multiarray = multiarray_type()
#     multiarray.layout.dim = [MultiArrayDimension('dim%d' % i,
#                                                  np_array.shape[i],
#                                                  np_array.shape[i] * np_array.dtype.itemsize) for i in range(np_array.ndim)]
#     multiarray.data = np_array.reshape([1, -1])[0].tolist()
#     return multiarray


# to_multiarray_f64 = partial(_numpy_to_multiarray, Float64MultiArray)


def callback(data):
    start = time.time()


    rospy.loginfo(rospy.get_caller_id() +
                  'Number of particles %d', len(data.poses))


    for i in range(len(data.poses)):

        # DWA
        pass
    
    # total time taken
    end = time.time()
    rospy.loginfo("Runtime of particle-DWA is %f" % (end - start))

def control_with_amcl(): 
    # using dwa for local controlling


    pass

def control_with_no_amcl(): 

    # Or just letting AMCL reset? 

    srv = rospy.ServiceProxy('request_nomotion_update', Empty)
    
    if not rospy.is_shutdown():
        srv()

def control(): 
    r = rospy.Rate(10)
    while not rospy.is_shutdown(): 
        if amcl_particles is None: 
            control_with_no_amcl()
        else: 
            control_with_amcl()
        r.sleep()



if __name__ == '__main__':

    rospy.init_node('particle_dwa', anonymous=True)

    sub = rospy.Subscriber('particlecloud', PoseArray, callback)

    pub = threading.Thread(target=control)
    pub.start()

    rospy.spin()