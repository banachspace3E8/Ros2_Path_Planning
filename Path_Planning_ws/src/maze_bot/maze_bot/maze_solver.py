#Shishir Khanal
#12/31/2014
#Script direct robot to end goal

import rclpy
import cv2
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import os
import numpy as np
from .bot_localization import bot_localizer

class maze_solver(Node):
    def __init__(self):
        super().__init__('maze_solving_node')
        self.subscriber = self.create_subscription(Image, '/upper_camera/image_raw', self.get_video_feed_cb, 10)
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        timer_period = 0.2
        self.timer = self.create_timer(timer_period, self.maze_solving)
        self.bridge = CvBridge()
        self.vel_msg = Twist()
        self.bot_localizer = bot_localizer()
        self.sat_view = np.zeros((100,100))

    def get_video_feed_cb(self, data):
        frame = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        self.sat_view = frame
        cv2.imshow("Satellite_View", frame)
        cv2.waitKey(1)

    def maze_solving(self):
        frame_disp = self.sat_view.copy()
        self.bot_localizer.localize_bot(self.sat_view, frame_disp)
        self.vel_msg.linear.x = 0.3
        self.vel_msg.angular.z = 0.5
        self.publisher_.publish(self.vel_msg)


def main(args=None):
    rclpy.init(args=args)
    image_subscriber = maze_solver()
    rclpy.spin(image_subscriber)
    rclpy.shutdown()

if __name__ == '__main__':
    main()