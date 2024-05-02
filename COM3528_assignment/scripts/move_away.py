#!/usr/bin/env python3
"""

"""
# Imports
##########################
import os
from math import radians  # This is used to reset the head pose
import numpy as np  # Numerical Analysis library
import cv2  # Computer Vision library

import rospy  # ROS Python interface
from sensor_msgs.msg import CompressedImage  # ROS CompressedImage message
from sensor_msgs.msg import JointState  # ROS joints state message
from cv_bridge import CvBridge, CvBridgeError  # ROS -> OpenCV converter
from geometry_msgs.msg import TwistStamped  # ROS cmd_vel (velocity control) message


import miro2 as miro  # Import MiRo Developer Kit library

try:  # For convenience, import this util separately
    from miro2.lib import wheel_speed2cmd_vel  # Python 3
except ImportError:
    from miro2.utils import wheel_speed2cmd_vel  # Python 2
##########################


class MiRoClient:
    """
    Script settings below
    """
    TICK = 0.02  # This is the update interval for the main control loop in secs
    CAM_FREQ = 1  # Number of ticks before camera gets a new frame, increase in case of network lag
    SLOW = 0.1  # Radial speed when turning on the spot (rad/s)
    FAST = 0.4  # Linear speed when kicking the ball (m/s)
    DEBUG = False  # Set to True to enable debug views of the cameras
    ##NOTE The following option is relevant in MiRoCODE
    NODE_EXISTS = False  # Disables (True) / Enables (False) rospy.init_node

    def reset_head_pose(self):
        """
        Reset MiRo head to default position, to avoid having to deal with tilted frames
        """
        self.kin_joints = JointState()  # Prepare the empty message
        self.kin_joints.name = ["tilt", "lift", "yaw", "pitch"]
        self.kin_joints.position = [0.0, radians(34.0), 0.0, 0.0]
        t = 0
        while not rospy.core.is_shutdown():  # Check ROS is running
            # Publish state to neck servos for 1 sec
            self.pub_kin.publish(self.kin_joints)
            rospy.sleep(self.TICK)
            t += self.TICK
            if t > 1:
                break

    def drive(self, speed_l=0.1, speed_r=0.1):  # (m/sec, m/sec)
        """
        Wrapper to simplify driving MiRo by converting wheel speeds to cmd_vel
        """
        # Prepare an empty velocity command message
        msg_cmd_vel = TwistStamped()

        # Desired wheel speed (m/sec)
        wheel_speed = [speed_l, speed_r]

        # Convert wheel speed to command velocity (m/sec, Rad/sec)
        (dr, dtheta) = wheel_speed2cmd_vel(wheel_speed)

        # Update the message with the desired speed
        msg_cmd_vel.twist.linear.x = dr
        msg_cmd_vel.twist.angular.z = dtheta

        # Publish message to control/cmd_vel topic
        self.vel_pub.publish(msg_cmd_vel)



    def detect_looming(self):
        """
        returns left or right depending on which camera sees looming stimuli
        """
        if self.just_switched:  # Print once
            print("MiRo is looking for looming stimuli")
            self.just_switched = False

        self.drive(self.SLOW, self.SLOW)
        #if looming stimuli is detected:
        #if detected in left camera
        self.looming_direction=1
        #if detected in right camera
        #self.looming_direction=2
        self.status_code = 2  # Switch to the second action
        self.just_switched = True


    def turn_away(self,direction):
        """
        #once looming stimuli has been detected, move away from it
        """
        if self.just_switched:  # Print once
            print("MiRo is turning away from looming stimulil")
            self.just_switched = False

        # If looming in right camera, rotate anticlockwise
        if direction==2:
            self.drive(-self.FAST, self.FAST)
        # if looming in left camera, rotate clockwise
        if direction==1:
            self.drive(self.FAST, -self.FAST)

         #once miro has turned away, go back to the beginning  
        else:
            self.status_code = 0  # Go back to square 1...
            self.just_switched = True




  

    def __init__(self):
        # Initialise a new ROS node to communicate with MiRo
        if not self.NODE_EXISTS:
            rospy.init_node("move_away", anonymous=True)
        # Give it some time to make sure everything is initialised
        rospy.sleep(2.0)
        # Initialise CV Bridge
        self.image_converter = CvBridge()
        # Individual robot name acts as ROS topic prefix
        topic_base_name = "/" + os.getenv("MIRO_ROBOT_NAME")
        # Create a new publisher to send velocity commands to the robot
        self.vel_pub = rospy.Publisher(
            topic_base_name + "/control/cmd_vel", TwistStamped, queue_size=0
        )
        # Create a new publisher to move the robot head
        self.pub_kin = rospy.Publisher(
            topic_base_name + "/control/kinematic_joints", JointState, queue_size=0
        )

        # Set the default frame width (gets updated on receiving an image)
        self.frame_width = 640
        # Action selector to reduce duplicate printing to the terminal
        self.just_switched = True
        # Bookmark
        self.bookmark = 0
        #left=1,right=2
        self.looming_direction=0
        # Move the head to default pose
        self.reset_head_pose()

    def loop(self):
        """
        Main control loop
        """
        print(" press CTRL+C to halt...")
        # Main control loop iteration counter
        self.counter = 0
        # This switch loops through MiRo behaviours:
        # look for stimuli, move away from stimuli
        self.status_code = 0
        while not rospy.core.is_shutdown():

            # Step 1. detect looming stimuli
            if self.status_code == 1:

                self.detect_looming()

            # Step 2. turn away from it
            elif self.status_code == 2:

                self.turn_away(self.looming_direction)

            # Fall back
            else:
                self.status_code = 1

            # Yield
            self.counter += 1
            rospy.sleep(self.TICK)


# This condition fires when the script is called directly
if __name__ == "__main__":
    main = MiRoClient()  # Instantiate class
    main.loop()  # Run the main control loop
