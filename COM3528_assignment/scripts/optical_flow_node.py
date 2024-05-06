#!/usr/bin/env python3

import os
import rospy
import time
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from geometry_msgs.msg import TwistStamped  # ROS cmd_vel (velocity control) message
from cv_bridge import CvBridge
import cv2
import numpy as np

try:  # For convenience, import this util separately
    from miro2.lib import wheel_speed2cmd_vel  # Python 3
except ImportError:
    from miro2.utils import wheel_speed2cmd_vel  # Python 2

def draw_flow(img, flow, step=16):
    """
    draw the flow arrows over an image and return the edited image
    """
    print ("drawing_flow")
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y, x].T

    lines = np.vstack([x, y, x-fx, y-fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)

    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(img_bgr, lines, 0, (255, 0, 0))

    for (x1, y1), (_x2, _y2) in lines:
        cv2.circle(img_bgr, (x1, y1), 1,(0, 255, 0), -1)

    return img_bgr



def draw_movement(flow, frame):
    """
    draw red pixels where looming points are on the input frame
    """
    # get the magnitudes and angles of the flow
    magnitude, angle = cv2.cartToPolar(flow[...,0],flow[...,1])
    # Define a threshold for at what magnitude is a point considered a looming point
    threshold= 3.0
    # filter the magnitude 2D list using the looming threshold
    looming_mask = magnitude > threshold
    # store the value for red in the picture where there are 'looming points'
    frame[looming_mask] = (0, 0, 255)
    return frame



def detect_looming_and_looming_direction(magnitude, angle):
    """
    detect points where there is looming when there is 'approaching looming' and return number of looming points
    """
    looming = 0
    direction = ""
    
    # Calculate mean magnitude to determine if the object is moving closer or expanding
    mean_magnitude = np.mean(magnitude)
    #print(f"mean_magnitude: {mean_magnitude}")
    # Define a threshold for what is considered an "approach"
    approach_threshold = 2.5
    
    # Define a threshold for at what magnitude is a point considered a looming point
    looming_threshold = 3.0
    
    # filter the magnitude 2D list using the looming threshold
    looming_mask = magnitude > looming_threshold
    
    # if looming is approaching count the number of looming points and return it
    if mean_magnitude > approach_threshold:
        looming = sum(map(sum, looming_mask))
        direction, mean_angle_deg = find_direction_of_looming(magnitude, angle, looming_threshold)
        #print(f"The average direction of the looming is {direction}")

    return looming, direction

def cropleft(frame):
    height, width, x= frame.shape
    top_left_y = int(height * 0.40)  # Crop off top 25%
    bottom_right_y = int(height * 0.90)  # Crop off bottom 25%
    top_left_x = 0  # Start at the left edge
    bottom_right_x = int(width * 0.85)  # End at the right edge
    # Crop the image
    cropped_image = frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
    return cropped_image

def cropright(frame):
    height, width, x= frame.shape
    top_left_y = int(height * 0.40)  # Crop off top 25%
    bottom_right_y = int(height * 0.90)  # Crop off bottom 25%
    top_left_x = int(width * 0.15)  # Start at the left edge
    bottom_right_x = width  # End at the right edge
    # Crop the image
    cropped_image = frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
    return cropped_image

def find_direction_of_looming(magnitude, angle, threshold):
    mask = magnitude < threshold
    angle[mask] = None
    direction = ""
    color = 2 # blue
    angle_deg = np.degrees(angle)

    mean_angle_deg = np.nanmean(angle_deg)
    if np.isnan(mean_angle_deg):
        direction = "none"
        color = 2 # green
    else:
        if (mean_angle_deg > 270 or mean_angle_deg <= 90):
            direction = "Right"
            color = 0 # red
        elif (mean_angle_deg > 90 and mean_angle_deg <= 270):
            direction = "Left"
            color = 120 # blue
    return direction, color



class OpticalFlowNode:

    def __init__(self):
        node_name = "optical_flow_node"
        rospy.init_node(node_name, anonymous=True)
        self.bridge = CvBridge()
        self.prev_frame_left = None
        self.prev_frame_right = None
        self.left_camera_sub = rospy.Subscriber('/miro/sensors/caml/compressed', CompressedImage, self.left_image_callback)
        self.right_camera_sub = rospy.Subscriber('/miro/sensors/camr/compressed', CompressedImage, self.right_image_callback)
        #self.looming_pub = rospy.Publisher('/looming', String, queue_size=1)
        #self.optical_flowl_pub = rospy.Publisher('/optical_flowl', Image, queue_size=0)
        #self.optical_flowr_pub = rospy.Publisher('/optical_flowr', Image, queue_size=0)
        self.state = 0
        topic_base_name = "/" + os.getenv("MIRO_ROBOT_NAME")
        self.vel_pub  = rospy.Publisher(topic_base_name + "/control/cmd_vel", TwistStamped, queue_size=0)
        self.flow_left = None
        self.flow_right = None
        print("initialising")



    def drive(self, speed_l, speed_r):  # (m/sec, m/sec)
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



    def left_image_callback(self, msg):
        """
        Handle the counting of 'looming points' in the left camera
        """
        # get the current frame from imgmsg
        current_frame = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
        current_frame_cropped = cropleft(current_frame)
        # convert color image to grayscale
        current_gray = cv2.cvtColor(current_frame_cropped, cv2.COLOR_BGR2GRAY)

        # Check if we have a previous frame
        if self.prev_frame_left is not None and self.state == 0:
            # Calculate optical flow using Lucas-Kanade method
            self.flow_left = cv2.calcOpticalFlowFarneback(
                self.prev_frame_left, current_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            
            #optical_flow_img = draw_movement(optical_flow_left, current_frame_cropped)
            #self.optical_flowl_pub.publish(self.bridge.cv2_to_imgmsg(optical_flow_img, "bgr8"))
            
        # Store the current frame for the next iteration
        self.prev_frame_left = current_gray
        self.compare_and_publish()



    def right_image_callback(self, msg):
        """
        Handle the counting of 'looming points' in the left camera
        """
        # get the current frame from imgmsg
        current_frame = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
        current_frame_cropped = cropright(current_frame)
        # convert color image to grayscale
        current_gray = cv2.cvtColor(current_frame_cropped, cv2.COLOR_BGR2GRAY)

        # Check if we have a previous frame
        if self.prev_frame_right is not None and self.state == 0:
            # Calculate optical flow using Lucas-Kanade method
            self.flow_right = cv2.calcOpticalFlowFarneback(
                self.prev_frame_right, current_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            
        # Store the current frame for the next iteration
        self.prev_frame_right = current_gray
        self.compare_and_publish()



    def compare_and_publish(self):
        """
        Check for approaching looming and direction of looming and handle movement
        """
        print(self.state)
        looming_left = 0
        looming_right = 0
        looming_dir_left = ""
        looming_dir_right = ""
        print(f"directionL : {self.looming_dir_left} directionR : {self.looming_dir_right}")
        if self.state == 0: # looming detection state
            mag_left, ang_left = cv2.cartToPolar(self.flow_left[..., 0], self.flow_left[..., 1])
            looming_left, looming_dir_left = detect_looming_and_looming_direction(mag_left, ang_left)
            mag_right, ang_right = cv2.cartToPolar(self.flow_right[..., 0], self.flow_right[..., 1])
            looming_right, looming_dir_right = detect_looming_and_looming_direction(mag_right, ang_right)
            #print("detecting looming...")
             # calculate total looming and print all the looming values
            difference_percentage = 0.5
            total_looming = looming_left + looming_right
            difference = total_looming * difference_percentage
            #print(f"left: {looming_left}, right: {looming_right}, total {total_looming}")
            if total_looming > 0:
                if (looming_left - looming_right) > difference and looming_dir_left == "Left" and looming_dir_right == "":
                    self.state = 1
                    #print("looming direciton left in left camera only")
                elif (looming_right - looming_left) > difference and looming_dir_right == "Right" and looming_dir_left == "":
                    self.state = 2
                    #print("looming direction right in right camera only")
                elif looming_dir_left == "Left" and looming_dir_right == "Right" and total_looming > 10000:
                    self.state = 2
                    print("IN FRONT turning random direction")
            else:
                #print("no looming - moving forward")
                self.drive(0.15,0.15)
        elif self.state == 1: # turning right state
            start = time.time()
            print("turning right")
            #turn right fast
            while time.time() - start < 1:
                self.drive(0.2,0.05)
            #turn right a bit slower
            while time.time() - start < 0.5:
                self.drive(0.1,0.05)
            self.state = 3
        elif self.state == 2: # turning left state
            start = time.time()
            print("turning left")
            #turn left fast
            while time.time() - start < 1:
                self.drive(0.05,0.2)
            #turn left slower
            while time.time() - start < 0.5:
                self.drive(0.05,0.1)
            self.state = 3
        elif self.state == 3: # stay still for 0.5s
            start = time.time()
            while time.time() - start < 3:
                self.drive(0.0,0.0)
                looming_left = 0
                looming_right = 0
                looming_dir_left = ""
                looming_dir_right = ""
            self.state = 0

if __name__ == '__main__':
    try:
        node = OpticalFlowNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
