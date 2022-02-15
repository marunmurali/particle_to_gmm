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

from tkinter import *

import itertools
from numpy.core.fromnumeric import mean
from numpy.ma.core import concatenate
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
import matplotlib as mpl
# plt.switch_backend('agg') 
import matplotlib.pyplot as plt

from numpy import linalg
from functools import partial
color_iter = itertools.cycle(
    ["navy", "c", "cornflowerblue", "gold", "darkorange"])


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


def plot_results(means, covariances, index, title):
    plt.clf()
    splot = plt.subplot(1, 1, 1)
    #splot = plt.subplots(figsize=(5, 2.7))
    for i, (mean, covar, color) in enumerate(zip(means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180.0 * angle / np.pi  # convert to degrees

        # rospy.loginfo('id: ' + str(i))
        rospy.loginfo('shorter radius: ' + str(v[0]))
        rospy.loginfo('longer radius: ' + str(v[1]))
        rospy.loginfo('angular command: ' + str(180.0 + angle))

        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180.0 + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    plt.xlim(mean[0]-0.3, mean[0]+0.3)
    plt.ylim(mean[1]-0.3, mean[1]+0.3)

    plt.xticks(())
    plt.yticks(())
    plt.title(title)
    plt.draw()
    plt.pause(0.00000000001)


class MyNode:
    def __init__(self):
        # In ROS, nodes are uniquely named. If two nodes with the same
        # name are launched, the previous one is kicked off. The
        # anonymous=True flag means that rospy will choose a unique
        # name for our 'listener' node so that multiple listeners can
        # run simultaneously.
        rospy.init_node('gmm_visualizer', anonymous=True)

        self.mean = None
        self.covariance = None

        rospy.Subscriber('gmm_mean', Float64MultiArray, self.MeanCallBack)
        rospy.Subscriber('gmm_covar', Float64MultiArray, self.CovarCallBack)

    def MeanCallBack(self, data):
        # print(type(data))
        self.mean = to_numpy_f64(data)
        # rowSize = data.layout.dim[0].size
        # tempArray = []
        # for i in range(rowSize):
        #     if len(tempArray) == 0:
        #         tempArray = [data.data[i*2], data.data[i*2+1]]
        #     else:
        #         tempArray =np.vstack([tempArray, [data.data[i*2], data.data[i*2+1]]])
        # self.mean = np.array(tempArray)
        # print(np.shape(self.mean))
        # print(self.mean)

    def CovarCallBack(self, data):
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
        self.try_plot()

    def try_plot(self):
        if self.covariance is None or self.mean is None:
            return
        else:
            plot_results(
                self.mean,
                self.covariance,
                1,
                "Bayesian Gaussian Mixture with a Dirichlet process prior",
            )


if __name__ == '__main__':
    plt.figure()
    plt.ion()
    node = MyNode()
    plt.show()

    plt.clf()
    plt.close()

    # spin() simply keeps python from exiting until this node is stopped

    rospy.spin()
