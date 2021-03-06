import rospy
from std_msgs.msg import Int32
import threading
counter = 0
pub = rospy.Publisher("/counter", Int32, queue_size=10)
def callback_temperature_sensor(msg):
    rospy.loginfo(msg.data)
def publisher_thread():
    rate = rospy.Rate(5) # ROS Rate at 5Hz
    while not rospy.is_shutdown():
        global counter
        counter += 1
        msg = Int32()
        msg.data = counter
        pub.publish(counter)
        rate.sleep()
if __name__ == '__main__':
    rospy.init_node("rospy_rate_test")
    
    sub = rospy.Subscriber("/temperature", Int32, callback_temperature_sensor)
    worker = threading.Thread(target=publisher_thread)
    worker.start()
    
    rospy.spin()