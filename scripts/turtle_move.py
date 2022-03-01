#!/usr/bin/env python3

import rospy
import time
from geometry_msgs.msg import PoseWithCovarianceStamped
from geometry_msgs.msg import Twist
from turtle_global_base import turtle_control
import math
import numpy as np

def angleCalculator(x1, y1, x2, y2):
    if(x2 == x1):
        return(90)
    else:
        return(math.degrees(np.arctan((y2-y1)/(x2-x1))))

if __name__ == '__main__':

    spinner = turtle_control()
    angle_threshold = 5

    #initialize MPC parameters, pass to listener
    while(True):
              
        #target distance
        dataIn = [0]*2 
        dataIn[0] = float(input('Please enter goal pose x:'))
        dataIn[1] = float(input('Please enter goal pose y:'))


        angle_request = angleCalculator(spinner.feedback.pose.pose.position.x, spinner.feedback.pose.pose.position.y, dataIn[0], dataIn[1])
        
        while not rospy.is_shutdown():
            
            while(not spinner.setOrientation(angle_request, angle_threshold)):
                print("Still turning")
                rospy.sleep(0.02)
            break