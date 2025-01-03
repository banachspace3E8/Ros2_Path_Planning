#Shishir Khanal
#12/12/2014
#Script to localize the bot
from rclpy.node import Node
import cv2
import numpy as np
from .utilities import ret_smallest_obj, ret_largest_obj
from . import config

class bot_localizer(Node):

    def __init__(self):
        self.is_bg_extracted = False
        self.bg_model = []
        self.maze_og = []
        #transformation vars
        self.orig_X = 0
        self.orig_Y = 0
        self.orig_rows = 0
        self.orig_cols = 0
        self.transform_arr = []
        self.orig_rot = 0
        self.rot_mat = 0
        self.loc_car = 0

    #provide region of interest mask and its contour
    #provides a hull that encloses the maze
    @staticmethod
    def ret_regOfIntrst_boundinghull(regOfIntrst_mask, cnts):
        maze_enclosure = np.zeros_like(regOfIntrst_mask)
        if cnts:
            cnts_ = np.concatenate(cnts)
            cnts_ = np.array(cnts_)
            cv2.fillConvexPoly(maze_enclosure, cnts_, 255)
        cnts_largest = cv2.findContours(maze_enclosure, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
        hull = cv2.convexHull(cnts_largest[0])
        cv2.drawContours(maze_enclosure, [hull], 0, 255)
        return hull

    #connect disconnected objects that are close enough
    @staticmethod
    def connect_objs(bin_img):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        return(cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel))

    def update_frameofreference_parameters(self, X, Y, W, H, rot_angle):
        self.orig_X = X
        self.orig_Y = Y
        self.orig_rows = H
        self.orig_cols = W
        self.orig_rot = rot_angle
        self.transform_arr = [X, Y, W, H]
        #Rotation Matrix
        self.rot_mat = np.array(
                                [
                                    [np.cos(np.deg2rad(self.orig_rot)) , np.sin(np.deg2rad(self.orig_rot))],
                                    [-np.sin(np.deg2rad(self.orig_rot)) , np.cos(np.deg2rad(self.orig_rot))]
                                ])
        self.rot_mat_rev = np.array(
                                [
                                    [np.cos(np.deg2rad(-self.orig_rot)) , np.sin(np.deg2rad(-self.orig_rot))],
                                    [-np.sin(np.deg2rad(-self.orig_rot)) , np.cos(np.deg2rad(-self.orig_rot))]
                                ])

    def extract_bg(self, frame):
        #extract mask of region of interest present in frame: canny edge detection algorithm
        #       Convert to grayscale image
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(frame_gray, 50, 150, None, 3)
        edges = self.connect_objs(edges)
        cnts = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
        regOfIntrst_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

        #Extract mask of all region of interest
        for idx,_ in enumerate(cnts):
            cv2.drawContours(regOfIntrst_mask, cnts, idx, 255, -1)
        
        #remove car from region of interest
        #Heuristic: Car is the smallest object in the maze
        min_cntr_idx = ret_smallest_obj(cnts)
        regOfIntrst_noCar_mask = regOfIntrst_mask.copy()
        #check whether min contour index was legit
        if min_cntr_idx != -1:
            cv2.drawContours(regOfIntrst_noCar_mask, cnts, min_cntr_idx, 0, -1)
            #Draw dilated car mask
            car_mask = np.zeros_like(regOfIntrst_mask)
            cv2.drawContours(car_mask, cnts, min_cntr_idx, 255, -1)
            cv2.drawContours(car_mask, cnts, min_cntr_idx, 255, 3)
            #Area that doesnt include car
            NotCar_mask = cv2.bitwise_not(car_mask)
            frame_car_remvd = cv2.bitwise_and(frame, frame, mask=NotCar_mask)
            
            base_clr = frame_car_remvd[0][0]
            Ground_replica = np.ones_like(frame) * base_clr
            #get region where only the car was present and fill with background pixels
            self.bg_model = cv2.bitwise_and(Ground_replica, Ground_replica, mask = car_mask)
            self.bg_model = cv2.bitwise_or(self.bg_model, frame_car_remvd)
        #Extract the maze Maze Entry on Top
        # a. Find dimension of hull enclosing largest contour
        hull = self.ret_regOfIntrst_boundinghull(regOfIntrst_mask, cnts)
        [X, Y, W, H] = cv2.boundingRect(hull)
        #b. Crop maze_mask from the image
        maze = regOfIntrst_noCar_mask[Y:Y+H, X:X+W]
        #invert the image to detect occupancy grids by excluding walls
        maze_OccupencyGrid = cv2.bitwise_not(maze)
        #Make sure entry point of the maze is on top
        self.maze_og = cv2.rotate(maze_OccupencyGrid, cv2.ROTATE_90_COUNTERCLOCKWISE)

        #Store Crop and Rot params
        self.update_frameofreference_parameters(X, Y, W, H, 90)

        if(config.debug and config.debug_localization):
            #cv2.imshow('1a. RegionofInterest_mask', regOfIntrst_mask)
            #cv2.imshow('1b. Frame_Car_Removed', frame_car_remvd)
            #cv2.imshow('1c. Ground_Replica', Ground_replica)
            #cv2.imshow('1d. Background_Model', self.bg_model)
            #cv2.imshow('2. Maze_Occupancy_Grid', self.maze_og)
            pass

    @staticmethod
    def get_centroid(cnt):
        M = cv2.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        return (cy, cx)
    
    def get_car_loc(self, car_cnt, car_mask):
        bot_cntr = self.get_centroid(car_cnt)
        # Convert from point to array (To apply transforms)
        bot_cntr_arr = np.array([bot_cntr[1], bot_cntr[0]])
        bot_cntr_translated = np.zeros_like(bot_cntr_arr)
        bot_cntr_translated[0] = bot_cntr_arr[0] - self.orig_X
        bot_cntr_translated[1] = bot_cntr_arr[1] - self.orig_Y   #Translate row

        bot_on_maze = (self.rot_mat @ bot_cntr_translated.T).T
        #Translate origin if necessary (To get complete image)
        rot_cols = self.orig_cols
        rot_rows = self.orig_rows
        bot_on_maze[0] = bot_on_maze[0] + (rot_cols * (bot_on_maze[0]<0))
        bot_on_maze[1] = bot_on_maze[1] + (rot_cols * (bot_on_maze[1]<0))
        #Update the placeholder for relative location of car
        self.loc_car = (int(bot_on_maze[0]), int(bot_on_maze[1]))

    def localize_bot(self, curr_frame, frame_disp):

        if not self.is_bg_extracted:
            self.extract_bg(curr_frame)
            self.is_bg_extracted = True

        #Foreground extraction
        change = cv2.absdiff(curr_frame, self.bg_model)
        change_gray = cv2.cvtColor(change, cv2.COLOR_BGR2GRAY)
        change_mask = cv2.threshold(change_gray, 15, 255, cv2.THRESH_BINARY)[1]
        #car will be the largest object in the change mask
        car_mask, car_cnt = ret_largest_obj(change_mask)

        self.get_car_loc(car_cnt, car_mask)

        #Draw bounding circle around detected car
        center, radii = cv2.minEnclosingCircle(car_cnt)
        car_circular_mask = cv2.circle(car_mask.copy(), (int(center[0]), int(center[1])), int(radii+(radii*0.4)), 255, 3)
        car_circular_mask = cv2.bitwise_xor(car_circular_mask, car_mask)
        frame_disp[car_mask>0] = frame_disp[car_mask>0] + (0, 64, 0)
        frame_disp[car_circular_mask>0] = (0,0,255)

        if (config.debug and config.debug_localization):
            cv2.imshow("Background_Model",self.bg_model)
            cv2.imshow("Maze_OcupancyGrid",self.maze_og)            
            cv2.imshow("Car_Localized", frame_disp)
            cv2.imshow("Change_Mask(Noise Visible)", change_mask) 
            cv2.imshow("Detected_Car", car_mask)
