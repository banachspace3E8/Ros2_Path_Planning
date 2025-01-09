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
from .bot_mapping import bot_mapper
from .bot_pathplanning import bot_pathplanner
from . import config

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
        self.bot_mapper = bot_mapper()
        self.bot_pathplanner = bot_pathplanner()
        self.sat_view = np.zeros((100,100))

    def get_video_feed_cb(self, data):
        frame = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        self.sat_view = frame
        cv2.imshow("Satellite_View", frame)
        cv2.waitKey(1)

    def maze_solving(self):
        #Display frame
        frame_disp = self.sat_view.copy()
        #Localize Robot
        self.bot_localizer.localize_bot(self.sat_view, frame_disp)
        #Create a map and store interest points in a graph
        self.bot_mapper.graphify(self.bot_localizer.maze_og)
        #Display feasible paths
        start = self.bot_mapper.Graph.start
        end = self.bot_mapper.Graph.end
        maze = self.bot_mapper.maze
        self.bot_pathplanner.find_and_display_path(self.bot_mapper.Graph.graph, start, end, maze, method="DFS")
        self.bot_pathplanner.find_and_display_path(self.bot_mapper.Graph.graph, start, end, maze, method="DFS_Shortest")
        self.bot_pathplanner.find_and_display_path(self.bot_mapper.Graph.graph, start, end, maze, method="A_star")
        self.bot_pathplanner.find_and_display_path(self.bot_mapper.Graph.graph, start, end, maze,method="Dijkstra")
        if config.debug and config.debug_pathplanning:
            print("\nNodes Visited [Dijkstra,  A*] --> [ {},  {} ]".format(self.bot_pathplanner.dijkstra.dijiktra_nodes_visited,self.bot_pathplanner.astar.astar_nodes_visited))
        
        self.vel_msg.linear.x = 0.0
        self.vel_msg.angular.z = 0.0
        self.publisher_.publish(self.vel_msg)

def main(args=None):
    rclpy.init(args=args)
    image_subscriber = maze_solver()
    rclpy.spin(image_subscriber)
    rclpy.shutdown()

if __name__ == '__main__':
    main()