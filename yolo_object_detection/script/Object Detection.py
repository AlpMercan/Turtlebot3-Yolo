#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import darknet

def load_class_names(names_file):
    with open(names_file, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    return class_names

class YOLODetector:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.callback)
        self.network, self.class_names, self.class_colors = darknet.load_network(
            "/home/alp/darknet/cfg/yolov4.cfg",
            "/home/alp/darknet/data/coco.data",
            "/home/alp/darknet/yolov4.weights",
            batch_size=1)
        self.width = darknet.network_width(self.network)
        self.height = darknet.network_height(self.network)


    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        darknet_image = darknet.make_image(self.width, self.height, 3)
        frame_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (self.width, self.height), interpolation=cv2.INTER_LINEAR)

        darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())
        detections = darknet.detect_image(self.network, self.class_names, darknet_image, thresh=0.2)
        darknet.free_image(darknet_image)
        if detections:
            print("Detections:")
            for label, confidence, bbox in detections:
                print(f"Detected: {label} with confidence {confidence}")
        else:
            print("No detections.")





def main():
    yolo_detector = YOLODetector()
    rospy.init_node('yolo_detector', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
