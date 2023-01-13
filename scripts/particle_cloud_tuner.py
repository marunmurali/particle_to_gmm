#!/usr/bin/env python3
# coding: utf-8

# A program that tries to prevent the particle cloud from converging to a wrong position by giving the samples a random noise. 

# Good but not good. The AMCL node doesn't READ the topic at all. It has its own copy of samples which seems unwritable. 

# Includes

# from numpy.core.fromnumeric import mean
import rospy
import time
import numpy as np
import random
import threading

# from std_msgs.msg import String
from geometry_msgs.msg import PoseArray


# Global variables

particlecloud = PoseArray()


# Methods

def callback(data):
    global particlecloud

    particlecloud = data

def tuner():
    r = rospy.Rate(2)

    while not rospy.is_shutdown():
        start_time = time.time()

        data = particlecloud

        publisher = rospy.Publisher('particlecloud', PoseArray, queue_size=10)  # New particle cloud

        # initialize empty list
        for i_particle in range(len(data.poses)):
            random_x = random.gauss(0.0, 0.05)
            random_y = random.gauss(0.0, 0.05)

            data.poses[i_particle].position.x += random_x
            data.poses[i_particle].position.y += random_y

        publisher.publish(data)

        # total time taken
        end_time = time.time()

        rospy.loginfo("Runtime of AMCL sumple tuner is %f" % (end_time - start_time))

        r.sleep()


if __name__ == '__main__':
    random.seed()

    rospy.init_node('particle_cloud_tuner', anonymous=True)

    rospy.Subscriber('particlecloud', PoseArray, callback)

    thread = threading.Thread(target=tuner())
    thread.start()

    rospy.spin()