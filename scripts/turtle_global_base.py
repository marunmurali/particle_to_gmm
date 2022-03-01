#!/usr/bin/env python3

from black import nullcontext
from click import command
from scipy.fft import next_fast_len
from sympy import N, true
import rospy
import time
from geometry_msgs.msg import PoseWithCovarianceStamped
from geometry_msgs.msg import Twist
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import osqp
import numpy as np
import scipy as sp
from scipy import sparse
import math

class turtle_control:

    feedback = PoseWithCovarianceStamped()
    command = Twist()
    wheel_radius = 0.127
    steer_max = math.pi * 0.5
    wheel_angle_velocity_threshold = 0.017
    wheel_base = 1.1
    spin_speed = 0.2

    def publisher(self, value):
        self.pub.publish(value)

    def callback(self, data):
        self.feedback = data
        

    def setOrientation(self, angle_request, angle_threshold):
        explicit_quat = [self.feedback.pose.pose.orientation.x, self.feedback.pose.pose.orientation.y, self.feedback.pose.pose.orientation.z, self.feedback.pose.pose.orientation.w]
        
        (roll, pitch, yaw) = euler_from_quaternion (explicit_quat)
        yaw = math.degrees(yaw)
        
        print("Present angle is :", yaw)
        print("Target angle is :", angle_request)
        print("Angle error is :", abs(angle_request - yaw))
        print("Angle error threshold is :", angle_threshold)
        spin_speed = 0.2 if (angle_request >= yaw) else -0.2
        self.command.angular.z = spin_speed
        self.pub.publish(self.command)
        time.sleep(0.1) #to reach required angle
        
        if(abs(angle_request - yaw) < angle_threshold):
            self.command.angular.z = 0
            self.pub.publish(self.command)
            return True        
        return False
    
    def setLiftHeight(self, lift_height_request, lift_height_threshold):
        self.command.lift_height = lift_height_request


        #self.publisher(self.command)
        time.sleep(0.2) #to reach required angle
        if(abs(lift_height_request - self.feedback.lift_height) < lift_height_threshold):
            return True
        return False    
    
    
    
    def MoveStraight(self, move_straight_request, wheel_angle_threshold):

        wheel_angle_start = self.feedback.wheel_angle
        wheel_angle_goal = wheel_angle_start + move_straight_request / self.wheel_radius


        # Discrete time model of a flaptter
        #0.00254 must be taking care of the m conversion already
        Ad = sparse.csc_matrix([
        [1.,      0.020],
        [0.,      0.9661 ]
        ])
        Bd = sparse.csc_matrix([
        [0.     ],
        [0.0315 ]])
        [nx, nu] = Bd.shape

        # Constraints
        u0 = 0.0
        umin = np.array([-6.0]) - u0
        umax = np.array([10.0]) - u0
        xmin = np.array([wheel_angle_start-100.0, -6.0])
        xmax = np.array([wheel_angle_goal+100.0, 10.0])
        # Objective function
        Q = sparse.diags([1., 0.]) #2
        QN = sparse.diags([3., 0.]) #5
        R = 0.3*sparse.eye(1) #0.2

        # Initial and reference states
        x0 = np.zeros(2)
        xr = np.array([wheel_angle_goal,0.0])

        # Prediction horizon
        N = 200

        # Cast MPC problem to a QP: x = (x(0),x(1),...,x(N),u(0),...,u(N-1))
        # - quadratic objective
        P = sparse.block_diag([sparse.kron(sparse.eye(N), Q), QN,
                            sparse.kron(sparse.eye(N), R)], format='csc')
        # - linear objective
        q = np.hstack([np.kron(np.ones(N), -Q.dot(xr)), -QN.dot(xr),
                    np.zeros(N*nu)])
        # - linear dynamics
        Ax = sparse.kron(sparse.eye(N+1),-sparse.eye(nx)) + sparse.kron(sparse.eye(N+1, k=-1), Ad)
        Bu = sparse.kron(sparse.vstack([sparse.csc_matrix((1, N)), sparse.eye(N)]), Bd)
        Aeq = sparse.hstack([Ax, Bu])
        leq = np.hstack([-x0, np.zeros(N*nx)])
        ueq = leq
        # - input and state constraints
        Aineq = sparse.eye((N+1)*nx + N*nu)
        lineq = np.hstack([np.kron(np.ones(N+1), xmin), np.kron(np.ones(N), umin)])
        uineq = np.hstack([np.kron(np.ones(N+1), xmax), np.kron(np.ones(N), umax)])
        # - OSQP constraints
        A = sparse.vstack([Aeq, Aineq], format='csc')
        l = np.hstack([leq, lineq])
        u = np.hstack([ueq, uineq])

        # Create an OSQP object
        prob = osqp.OSQP()

        # Setup workspace
        prob.setup(P, q, A, l, u, warm_start=True)

        while (True):
            start=time.time()
            #stop condition
            wheel_angle_error = xr[0]-self.feedback.wheel_angle

            if (abs(wheel_angle_error) < wheel_angle_threshold and abs(self.feedback.wheel_angle_velocity) < self.wheel_angle_velocity_threshold):
                rospy.loginfo("Finished MoveStraight successfully")
                self.command.wheel_angle_velocity = 0
                #self.publisher(self.command)
                return true

            #initialize the current state
            x0 = np.array([self.feedback.wheel_angle, self.feedback.wheel_angle_velocity_filtered])
            l[:nx] = -x0
            u[:nx] = -x0
            prob.update(l=l, u=u)

            #solve
            res = prob.solve()
            
            # Check solver status
            if res.info.status != 'solved':
                self.command.wheel_angle_velocity = 0.0
                #self.publisher(self.command)
                rospy.signal_shutdown('done')
                raise ValueError('OSQP did not solve the problem!')
            else:
                #pick the results, publish command
                self.command.wheel_angle_velocity = res.x[-N*nu:-(N-1)*nu]
                print("velocity command is", self.command.wheel_angle_velocity)
                #self.publisher(self.command)
            
            # check total time taken
            end = time.time()
            print("Runtime of the program is %f" %(end - start))



    def __init__(self):
        rospy.init_node('turtle_mpc_global', anonymous=True)
        #Creating a Subscriber
        self.sub =  rospy.Subscriber('amcl_pose', PoseWithCovarianceStamped, self.callback)
        #Creating Publisher
        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)