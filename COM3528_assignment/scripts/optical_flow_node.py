#!/usr/bin/env python3

import os
import rospy
import time
from sensor_msgs.msg import Image, CompressedImage
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
    for debugging - draw the flow arrows over an image and return the edited image
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
    for debugging - draw red pixels where looming points are on the input frame
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


def find_avg_angle (flow, threshold, step):
    """
    overlays a grid of points onto the flow array to take the angle values of those points - basically simplifies
    """
    # get height and width of flow array
    h, w = flow.shape[:2]
    # create the x and y values for the points where we will take values
    y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)
    # get the values of the flow at each of those points
    fx, fy = flow[y, x].T

    # Calculate the magnitude of each vector
    magnitudes = np.sqrt(fx ** 2 + fy ** 2)

    # create a mask (boolean array) that is true when the magnitudes are above a threshold
    threshold_mask = magnitudes > threshold
    normalized_fx = np.zeros_like(fx)
    normalized_fy = np.zeros_like(fy)

    # Normalize only the vectors that correspond to magnitudes over the threshold
    normalized_fx[threshold_mask] = fx[threshold_mask] / magnitudes[threshold_mask]
    normalized_fy[threshold_mask] = fy[threshold_mask] / magnitudes[threshold_mask]

    # Compute the average direction
    average_fx = np.mean(normalized_fx)
    average_fy = np.mean(normalized_fy)

    # convert average direction to an angle
    average_angle = np.arctan2(average_fy, average_fx)  # Result in radians
    average_angle_degrees = np.degrees(average_angle)  # Convert to degrees
    #print(average_angle_degrees)
    return average_angle_degrees


def find_direction_from_angle(avg_angle):
    """
    takes an angle and returns a string for its direction (also can return a color to be drawn for debugging)
    """
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
    return direction


def detect_looming_and_looming_direction(flow, step=16):
    """
    detect points where there is looming when there is 'approaching looming', checks the average direction and magnitudes for the optical flow
    and return number of looming points, direction of flow and magnitude of flow
    """
    # get height and width of flow array
    h, w = flow.shape[:2]
    # create the x and y values for the points where we will take values
    y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)
    # get the values of the flow at each of those points
    fx, fy = flow[y, x].T

    # Calculate the magnitude of each vector
    magnitudes = np.sqrt(fx ** 2 + fy ** 2)
    looming = 0 # set default value for looming
    direction = "none" # set default value for direction
    
    # Calculate mean magnitude to determine if the object is moving closer or expanding
    mean_magnitude = np.mean(magnitudes)

    # Define a threshold for what is considered an "approach"
    approach_threshold = 5.0
    
    # only calculate when there is looming stimulus or approaching stimulus
    if mean_magnitude > approach_threshold:
        print("approaching")
        looming = np.sum(magnitudes > approach_threshold) # count number of looming points
        avg_angle = find_avg_angle(flow, approach_threshold, step) # find avg angle of flow
        direction = find_direction_from_angle(avg_angle) # find direction from angle
        #print(f"The average direction of the looming is {direction}")

    return looming, direction, mean_magnitude

def cropleft(frame):
    """
    crop the left side camera image
    """
    height, width, x= frame.shape
    top_left_y = int(height * 0.40)  # Crop off top 40%
    bottom_right_y = int(height * 0.90)  # Crop off bottom 10%
    top_left_x = 0  # Start at the left edge
    bottom_right_x = int(width * 0.85)  # Crop off right 15%
    cropped_image = frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x] # Crop the image
    return cropped_image

def cropright(frame):
    """
    crop the right side camera image
    """
    height, width, x= frame.shape
    top_left_y = int(height * 0.40)  # Crop off top 40%
    bottom_right_y = int(height * 0.90)  # Crop off bottom 10%
    top_left_x = int(width * 0.15)  # Crop off left 15%
    bottom_right_x = width  # End at the right edge
    cropped_image = frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x] # Crop the image
    return cropped_image


class OpticalFlowNode:

    def __init__(self):
        node_name = "optical_flow_node"
        rospy.init_node(node_name, anonymous=True)
        self.bridge = CvBridge()
        self.left_camera_sub = rospy.Subscriber('/miro/sensors/caml/compressed', CompressedImage, self.left_image_callback)
        self.right_camera_sub = rospy.Subscriber('/miro/sensors/camr/compressed', CompressedImage, self.right_image_callback)
        
        self.optical_flowl_pub = rospy.Publisher('/optical_flowl', Image, queue_size=0)
        self.optical_flowr_pub = rospy.Publisher('/optical_flowr', Image, queue_size=0)
        
        topic_base_name = "/" + os.getenv("MIRO_ROBOT_NAME")
        self.vel_pub  = rospy.Publisher(topic_base_name + "/control/cmd_vel", TwistStamped, queue_size=0)
        
        self.state = 0
        self.prev_frame_left = None
        self.prev_frame_right = None
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
        # get the current frame from imgmsg and crop the image
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
        # get the current frame from imgmsg and crop the image
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

    def get_next_state(self, mean_magnitude_left, mean_magnitude_right, total_looming, looming_dir_left, looming_dir_right):
        """
        handle the decision for the next state and set the nodes state to what state is decided
        """
        print(f"left: {mean_magnitude_left}, right: {mean_magnitude_right}")
        print(f"directions:                                                                     {looming_dir_left}                                  {looming_dir_right}")
        if total_looming > 100:
            if mean_magnitude_left > mean_magnitude_right and looming_dir_left == "Left" and looming_dir_right == "none":
                self.state = 1 # set to turning right state
                print("looming detected going LEFT")
            elif mean_magnitude_right > mean_magnitude_left and looming_dir_right == "Right" and looming_dir_left == "none":
                self.state = 2 # set to turning left state
                print("looming detected going RIGHT")
            elif np.abs(mean_magnitude_left - mean_magnitude_right) < 0.5 or looming_dir_left == "Left" and looming_dir_right == "Right":
                self.state = 3 # set to reverse state
                print("looming IN FRONT turning random direction")
            else:
                self.state = 0 # just set the state back to 0 (if conditions aren't met)
                self.drive(0.15, 0.15)
        elif total_looming < 1000 and looming_dir_left == "none" and looming_dir_right == "none":
            print("no looming - moving forward")
            self.drive(0.15,0.15)
            self.state = 0
        else:
            self.drive(0.0,0.0)
            self.state = 0



    def compare_and_publish(self):
        """
        Check for approaching looming and direction of looming and handle movement
        """
        print(self.state)
        # set the initial values for each variable incase there is no flow
        looming_left = 0
        looming_right = 0
        looming_dir_left = "none"
        looming_dir_right = "none"
        mean_magnitude_left = 0
        mean_magnitude_right = 0
        if self.state == 0: # looming detection state
            print ("Detecting looming...")
            # if there is optical flow then find the number of looming points, direction, and magnitudes
            if self.flow_left is not None and self.flow_right is not None:
                looming_left, looming_dir_left, mean_magnitude_left = detect_looming_and_looming_direction(self.flow_left)
                looming_right, looming_dir_right, mean_magnitude_right= detect_looming_and_looming_direction(self.flow_right)

            # calculate total looming and print all the looming values
            total_looming = looming_left + looming_right

            self.set_next_state(mean_magnitude_left, mean_magnitude_right, total_looming, looming_dir_left, looming_dir_right)

        elif self.state == 1: # turning right state
            start = time.time()
            print("starting turning right...")
            
            while time.time() - start < 1:
                self.drive(0.2,0.0) #turn right fast
                rospy.sleep(0.005)
            
            print("finished turning right")
            self.state = 4

        elif self.state == 2: # turning left state
            start = time.time()
            print("starting turning left")
            
            while time.time() - start < 1:
                self.drive(0.0,0.2) # turn left fast
                rospy.sleep(0.005)
            
            print("finished turning left")
            self.state = 4

        elif self.state == 3: # reverse then turn state
            start = time.time()
            print("reversing...")
            
            while time.time() - start < 1:
                self.drive(-0.05,-0.05) #reverse
                rospy.sleep(0.005)
            
            start = time.time()
            print("stopped reversing... now turning...")
            
            while time.time() - start < 1:
                self.drive(0.05,0.2) # turn left
                rospy.sleep(0.005)
            
            print("stopped turning")
            self.state = 4

        elif self.state == 4: # reset state
            self.drive(0.0,0.0)
            self.state = 0
            self.prev_frame_left = None
            self.prev_frame_right = None
            self.flow_left = None
            self.flow_right = None

if __name__ == '__main__':
    try:
        node = OpticalFlowNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
