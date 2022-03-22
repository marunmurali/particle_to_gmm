#!/usr/bin/env python3

from std_srvs.srv import Empty

import rospy

def send_amcl_service():

    rospy.init_node('amcl_service_sender', anonymous=True)

    r = rospy.Rate(2)

    s = rospy.ServiceProxy('global_localization', Empty)
    # s = rospy.ServiceProxy('request_nomotion_update', Empty)
    
    if not rospy.is_shutdown():
        s()


if __name__ == "__main__":

    send_amcl_service()