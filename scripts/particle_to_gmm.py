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

# Simple talker demo that listens to std_msgs/Strings published
# to the 'chatter' topic

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
from geometry_msgs.msg import PoseArray
from functools import partial


def _numpy_to_multiarray(multiarray_type, np_array):
    multiarray = multiarray_type()
    multiarray.layout.dim = [MultiArrayDimension('dim%d' % i,
                                                 np_array.shape[i],
                                                 np_array.shape[i] * np_array.dtype.itemsize) for i in range(np_array.ndim)]
    multiarray.data = np_array.reshape([1, -1])[0].tolist()
    return multiarray


to_multiarray_f64 = partial(_numpy_to_multiarray, Float64MultiArray)


def callback(data):
    pubMean = rospy.Publisher(
        'gmm_mean', Float64MultiArray, queue_size=10)  # GMM Mean
    pubCovar = rospy.Publisher(
        'gmm_covar', Float64MultiArray, queue_size=10)  # GMM Covariance
    start = time.time()
    rospy.loginfo(rospy.get_caller_id() +
                  'Number of particles %d', len(data.poses))
    # initialize empty list
    PoseForGmmArr = []
    gmm_mean = Float64MultiArray()
    gmm_covar = Float64MultiArray()
    #gmm_mean = []
    #gmm_covar = []
    for i in range(len(data.poses)):
        newrow = ([data.poses[i].position.x, data.poses[i].position.y])
        PoseForGmmArr.append(newrow)
    PoseForGmm = np.array(PoseForGmmArr)
    # print(PoseForGmm.shape)
    #dpgmm = mixture.BayesianGaussianMixture(n_components=5, covariance_type="full").fit(PoseForGmm)
    #dpgmm = BayesianGaussianMixture(n_components=5, covariance_type="full").fit(PoseForGmm)
    # n_components is maximum in Bayesian

    dpgmm = mixture.GaussianMixture(
        n_components=5, covariance_type="full").fit(PoseForGmm)

    # note: reducing n_components

    # gmm_mean = toMultiArray(dpgmm.means_)
    # gmm_covar = toMultiArray(dpgmm.covariances_)
    gmm_mean = to_multiarray_f64(dpgmm.means_)
    gmm_covar = to_multiarray_f64(dpgmm.covariances_)
    pubMean.publish(gmm_mean)
    pubCovar.publish(gmm_covar)

    # total time taken
    end = time.time()
    print("Runtime of the program is %f" % (end - start))


def toMultiArray(matrix):
    temp = Float64MultiArray()
    # TODO empty the temp
    # write layout
    for i in range(np.size(np.shape(matrix))):
        shapeArr = np.shape(matrix)
        # print("loop from to")
        # print(i)
        # print(np.size(np.shape(matrix)))
        # print("shape")
        # print(np.shape(matrix))
        # print("element")
        # print(shapeArr[i])
        # print(np.shape(matrix))
        temp.layout.dim.append(MultiArrayDimension())
        temp.layout.dim[i].label = "dim"+str(i)
        temp.layout.dim[i].size = shapeArr[i]
        temp.layout.dim[i].stride = matrix.strides[i]
    temp.layout.data_offset = 0
    # write data
    temp.data = matrix.flatten()

    return temp


def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('particle_to_gmm', anonymous=True)

    rospy.Subscriber('particlecloud', PoseArray, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()


if __name__ == '__main__':
    listener()
