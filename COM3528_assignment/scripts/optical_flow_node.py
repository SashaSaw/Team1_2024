#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np

def draw_flow(img, flow, step=16):
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
    magnitude, angle = cv2.cartToPolar(flow[...,0],flow[...,1])
    threshold= 5.0
    looming_mask = magnitude > threshold
    frame[looming_mask] = [0, 0, 255]
    return frame

def detect_looming_towards(flow):
    output = 0
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Calculate mean magnitude to determine if the object is moving closer or expanding
    mean_magnitude = np.mean(magnitude)
    # Define a threshold for what you consider to be an "approach"
    approach_threshold = 3.0  # adjust based on your specific needs

    looming_threshold = 5.0

    looming_mask = magnitude > looming_threshold

    if mean_magnitude > approach_threshold:
        looming = sum(map(sum, looming_mask))
        output = looming

    return output

class OpticalFlowNode:
    def __init__(self):
        node_name = "optical_flow_node"
        rospy.init_node(node_name, anonymous=True)
        self.bridge = CvBridge()
        self.prev_frame = None
        self.left_camera_sub = rospy.Subscriber('/miro/sensors/caml/compressed', CompressedImage, self.left_image_callback)
        self.right_camera_sub = rospy.Subscriber('/miro/sensors/camr/compressed', CompressedImage, self.right_image_callback)
        self.looming_pub = rospy.Publisher('/looming', String, queue_size=1)
        self.optical_flow_pub = rospy.Publisher('/optical_flow', Image, queue_size=1)
        self.looming_right=0
        self.counter = 0
        self.looming_left=0
        print("initialising")


    def left_image_callback(self, msg):
        current_frame = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")

        # Convert the current frame to grayscale
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        # Check if we have a previous frame
        if self.prev_frame is not None:
            # Calculate optical flow using Lucas-Kanade method
            optical_flow = cv2.calcOpticalFlowFarneback(
                self.prev_frame, current_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

            # Drawing the flow over the gray image
            #optical_flow_img = draw_flow(current_gray,optical_flow)
            self.looming_left = detect_looming_towards(optical_flow)

            print(f"left: {self.looming_left}")
            print(f"right: {self.looming_right}")
            total_looming = self.looming_left + self.looming_right
            print(total_looming)
            if total_looming > 0:
                if self.looming_left > self.looming_right:
                    self.looming_pub.publish("Something approaches from the Left")
                elif self.looming_right > self.looming_left:
                    self.looming_pub.publish("Something approaches from the right")
            
            img = draw_movement(optical_flow, current_frame)
            # Publish the optical flow image
            self.optical_flow_pub.publish(self.bridge.cv2_to_imgmsg(img, encoding="bgr8"))
            
        # Store the current frame for the next iteration
        self.prev_frame = current_gray

    def right_image_callback(self, msg):
        current_frame = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")

        # Convert the current frame to grayscale
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        # Check if we have a previous frame
        if self.prev_frame is not None:
            # Calculate optical flow using Lucas-Kanade method
            optical_flow = cv2.calcOpticalFlowFarneback(
                self.prev_frame, current_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

            # Drawing the flow over the gray image
            #optical_flow_img = draw_flow(current_gray,optical_flow)
            self.looming_right = detect_looming_towards(optical_flow)
            
        # Store the current frame for the next iteration
        self.prev_frame = current_gray

if __name__ == '__main__':
    try:
        node = OpticalFlowNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
