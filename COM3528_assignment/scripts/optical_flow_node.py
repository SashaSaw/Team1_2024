#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image, CompressedImage
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

class OpticalFlowNode:
    def __init__(self):
        node_name = "optical_flow_node"
        rospy.init_node(node_name, anonymous=True)
        self.bridge = CvBridge()
        self.prev_frame = None
        self.image_sub = rospy.Subscriber('/miro/sensors/caml/compressed', CompressedImage, self.image_callback)
        self.optical_flow_pub = rospy.Publisher('/optical_flow', Image, queue_size=10)
        rospy.on_shutdown(self.cleanup)
        print("initialising")
        img = np.zeros((512, 512, 3), np.uint8)
        cv2.imshow('Test Window', img)

    def cleanup(self):
        cv2.destroyAllWindows()

    def image_callback(self, msg):
        current_frame = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")

        # Convert the current frame to grayscale
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        # Check if we have a previous frame
        if self.prev_frame is not None:
            # Calculate optical flow using Lucas-Kanade method
            optical_flow = cv2.calcOpticalFlowFarneback(
                self.prev_frame, current_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

            # Drawing the flow over the gray image
            optical_flow_img = draw_flow(current_gray,optical_flow)

            # Publish the optical flow image
            self.optical_flow_pub.publish(self.bridge.cv2_to_imgmsg(optical_flow_img, encoding="bgr8"))
            
        # Store the current frame for the next iteration
        self.prev_frame = current_gray

if __name__ == '__main__':
    try:
        node = OpticalFlowNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
