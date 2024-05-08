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

def draw_flow(flow, img, step=16):
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
    threshold= 5.0
    # filter the magnitude 2D list using the looming threshold
    looming_mask = magnitude > threshold
    # store the value for red in the picture where there are 'looming points'
    frame[looming_mask] = (0, 0, 255)
    return frame


def find_avg_angle (flow, threshold, step=16):
    h,w = flow.shape[:2]
    h, w = flow.shape[:2]
    y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T

    # Calculate the magnitude of each vector
    magnitudes = np.sqrt(fx ** 2 + fy ** 2)

    # Avoid division by zero in case of zero-length vectors
    nonzero = magnitudes > threshold
    normalized_fx = np.zeros_like(fx)
    normalized_fy = np.zeros_like(fy)

    # Normalize only non-zero vectors
    normalized_fx[nonzero] = fx[nonzero] / magnitudes[nonzero]
    normalized_fy[nonzero] = fy[nonzero] / magnitudes[nonzero]

    # Compute the average direction
    average_fx = np.mean(normalized_fx)
    average_fy = np.mean(normalized_fy)

    # Optionally, convert average direction to an angle
    average_angle = np.arctan2(average_fy, average_fx)  # Result in radians
    average_angle_degrees = np.degrees(average_angle)  # Convert to degrees if needed
    print(average_angle_degrees)
    return average_angle_degrees

def find_direction_from_angle(avg_angle):
    #print(mean_angle_deg)
    if avg_angle == 0:
        direction = "none"
        color = 2 # green
    else:
        if (avg_angle > 270 or avg_angle <= 90):
            direction = "Right"
            color = 0 # red
        elif (avg_angle > 90 and avg_angle <= 270):
            direction = "Left"
            color = 120 # blue
    return direction, color


def detect_looming_and_looming_direction(flow):
    """
    detect points where there is looming when there is 'approaching looming' and return number of looming points
    """
    magnitude, angle = cv2.cartToPolar(flow[...,0],flow[...,1])
    looming = 0
    direction = ""
    
    # Calculate mean magnitude to determine if the object is moving closer or expanding
    mean_magnitude = np.mean(magnitude)
    #print(f"mean_magnitude: {mean_magnitude}")
    # Define a threshold for what is considered an "approach"
    approach_threshold = 2.5
    
    # Define a threshold for at what magnitude is a point considered a looming point
    looming_threshold = 5.0
    
    # filter the magnitude 2D list using the looming threshold
    looming_mask = magnitude > looming_threshold
    
    # if looming is approaching count the number of looming points and return it
    if mean_magnitude > approach_threshold:
        looming = sum(map(sum, looming_mask))
        avg_angle = find_avg_angle(flow, looming_threshold)
        direction = find_direction_from_angle(avg_angle)
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
"""
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
"""


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
        self.optical_flowl_pub = rospy.Publisher('/optical_flowl', Image, queue_size=0)
        self.optical_flowr_pub = rospy.Publisher('/optical_flowr', Image, queue_size=0)
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
        self.current_frame_croppedl = cropleft(current_frame)
        # convert color image to grayscale
        current_gray = cv2.cvtColor(self.current_frame_croppedl, cv2.COLOR_BGR2GRAY)

        # Check if we have a previous frame
        if self.prev_frame_left is not None and self.state == 0:
            # Calculate optical flow using Lucas-Kanade method
            self.flow_left = cv2.calcOpticalFlowFarneback(
                self.prev_frame_left, current_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            
        # Store the current frame for the next iteration
        self.prev_frame_left = current_gray
        self.compare_and_publish()



    def right_image_callback(self, msg):
        """
        Handle the counting of 'looming points' in the left camera
        """
        # get the current frame from imgmsg
        current_frame = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.current_frame_croppedr = cropright(current_frame)
        # convert color image to grayscale
        current_gray = cv2.cvtColor(self.current_frame_croppedr, cv2.COLOR_BGR2GRAY)

        # Check if we have a previous frame
        if self.prev_frame_right is not None and self.state == 0:
            # Calculate optical flow using Lucas-Kanade method
            self.flow_right = cv2.calcOpticalFlowFarneback(
                self.prev_frame_right, current_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            
        # Store the current frame for the next iteration
        self.prev_frame_right = current_gray
        #self.compare_and_publish()



    def compare_and_publish(self):
        """
        Check for approaching looming and direction of looming and handle movement
        """
        print(self.state)
        
        if self.state == 0: # looming detection state
            #self.drive(0.0, 0.0)
            print ("Detecting looming...")
            if self.flow_left is not None and self.flow_right is not None:
                looming_left, looming_dir_left = detect_looming_and_looming_direction(self.flow_left)
                looming_right, looming_dir_right = detect_looming_and_looming_direction(self.flow_right)

                optical_flow_img1 = draw_flow(self.flow_left, self.prev_frame_left)
                self.optical_flowl_pub.publish(self.bridge.cv2_to_imgmsg(optical_flow_img1, "bgr8"))

                optical_flow_img2 = draw_flow(self.flow_right, self.prev_frame_right)
                self.optical_flowr_pub.publish(self.bridge.cv2_to_imgmsg(optical_flow_img2, "bgr8"))
            if looming_dir_right == looming_dir_left:
                looming_dir_left = ""
                looming_dir_right = ""
            # calculate total looming and print all the looming values
            difference_percentage = 0.5
            total_looming = looming_left + looming_right
            difference = total_looming * difference_percentage
            print(f"left: {looming_left}, right: {looming_right}, total {total_looming}")
            print(f"directions:                                                                     {looming_dir_left}                                  {looming_dir_right}")
            if total_looming > 1000:
                if (looming_left - looming_right) > difference and looming_dir_left == "Left" and looming_dir_right == "none":
                    self.state = 1
                    print("looming detected going LEFT")
                elif (looming_right - looming_left) > difference and looming_dir_right == "Right" and looming_dir_left == "none":
                    self.state = 2
                    print("looming detected going RIGHT")
                elif looming_dir_left == "Left" and looming_dir_right == "Right" and total_looming > 10000:
                    self.state = 3
                    print("looming IN FRONT turning random direction")
                else:
                    self.state = 0
                    self.drive(0.15, 0.15)
            elif total_looming < 1000 and looming_dir_left == "none" and looming_dir_right == "none":
                print("no looming - moving forward")
                self.drive(0.15,0.15)
                self.state = 0
            else:
                self.drive(0.0,0.0)
                self.state = 0

        elif self.state == 1: # turning right state
            start = time.time()
            print("starting turning right...")
            #turn right fast
            while time.time() - start < 0.5:
                self.drive(0.2,0.05)
            #turn right a bit slower
            while time.time() - start < 0.2:
                self.drive(0.1,0.05)
            while time.time() - start < 0.1:
                self.drive(0.0,0.0)
            print("finished turning right") 
            self.state = 4

        elif self.state == 2: # turning left state
            start = time.time()
            print("starting turning left")
            #turn left fast
            while time.time() - start < 0.5:
                self.drive(0.05,0.2)
            #turn left slower
            while time.time() - start < 0.2:
                self.drive(0.05,0.1)
            while time.time() - start < 0.1:
                self.drive(0.05,0.1)
            print("finished turning left")
            self.state = 4

        elif self.state == 3: # turning left state
            start = time.time()
            print("reversing...")
            #turn left fast
            while time.time() - start < 1:
                self.drive(-0.05,-0.05)
            print("stopped reversing... now turning...")
            #turn left slower
            while time.time() - start < 0.5:
                self.drive(0.05,0.2)
            while time.time() - start < 0.5:
                self.drive(0.0,0.0)
            print("stopped turning")
            self.state = 4

        elif self.state == 4: # stay still for 0.5s
            self.drive(0.0,0.0)
            looming_left = 0
            looming_right = 0
            looming_dir_left = "none"
            looming_dir_right = "none"
            self.state = 0

if __name__ == '__main__':
    try:
        node = OpticalFlowNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
