#Shishir Khanal
#1/11/2025
#Script to determine the location and pose of bot

import cv2
import numpy as np
from numpy import interp
from math import pow, atan2, sqrt, degrees, asin
from . import config
import os
import pygame

pygame.mixer.init()
pygame.mixer.music.load(os.path.abspath('Documents/GitHub/Ros2_Path_Planning/Path_Planning_ws/src/maze_bot/resource/Mini_Goal_Reached.wav'))

class bot_motionplanner():
    def __init__(self):
        #counter to move the car forward in few iterations
        self.count = 0
        # Keep track whether initial pose is extracted
        self.pt_i_taken = False
        #store initial car location
        self.init_loc = 0
        #Angle relation computed
        self.angle_relation_computed = False

        #Bot angle image
        self.bot_angle = 0
        #Bot Angle simulation
        self.bot_angle_s = 0
        #Angle betweeen Image and sim
        self.bot_angle_rel = 0
        #Maze exit not reached
        self.goal_not_reached_flag = True
        #Current mini goal (x,y)
        self.goal_pose_x = 0
        self.goal_pose_y = 0
        #Current mini goal iteration
        self.path_iter = 0

        #Previous iteration case
        self.prev_angle_to_turn = 0
        self.Prev_distance_to_goal = 0
        self.prev_path_iter = 0

        #Angle or distance or mini-goal
        self.prev_angle_to_turn = 0
        self.Prev_distance_to_goal = 0
        self.prev_path_iter = 0
        
        #Variables to keeps track of loops passed since last pose change
        self.angle_not_changed = 0
        self.dist_not_changed = 0
        self.goal_not_changed = 0
        self.goal_not_changed_long = 0
        self.backpedaling  = 0

        self.trigger_backpedaling = False
        self.trigger_nxtpt = False

    #common w/ gotogoal
    @staticmethod
    def euler_from_quaternion(x, y, z, w):
        t0 = 2.0 * (w * x + y * z)
        t1 = 1.0 - 2.0 * (x * x + y * y)
        roll_x = atan2(t0, t1)

        t2 = 2.0 * (w * y - z * x)
        t2 = 1.0 if t2 > 1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = asin(t2)

        t3 = 2.0 * (w * z + x * y)
        t4 = 1.0 - 2.0 * (y * y + z * z)
        yaw_z = atan2(t3, t4)

        return roll_x, pitch_y, yaw_z

    #common with gotogoal
    def get_pose(self, data):
        quaternions = data.pose.pose.orientation
        (roll, pitch, yaw) = self.euler_from_quaternion(quaternions.x, quaternions.y, quaternions.z, quaternions.w)
        yaw_deg = degrees(yaw)
        #Convert yaw meas from [-180, 180]->  [0, 360]
        if(yaw_deg>0):
            self.bot_angle_s = yaw_deg
        else:
            self.bot_angle_s = yaw_deg + 360

    @staticmethod
    def bck_to_orig(pt,transform_arr,rot_mat):

        st_col = transform_arr[0] # cols X
        st_row = transform_arr[1] # rows Y
        tot_cols = transform_arr[2] # total_cols / width W
        tot_rows = transform_arr[3] # total_rows / height H
        
        # point --> (col(x),row(y)) XY-Convention For Rotation And Translated To MazeCrop (Origin)
        #pt_array = np.array( [pt[0]+st_col, pt[1]+st_row] )
        pt_array = np.array( [pt[0], pt[1]] )
        
        # Rot Matrix (For Normal XY Convention Around Z axis = [cos0 -sin0])  Image convention [ cos0 sin0]
        #                                                      [sin0  cos0]                    [-sin0 cos0]
        rot_center = (rot_mat @ pt_array.T).T# [x,y]
        
        # Translating Origin If neccasary (To get whole image)
        rot_cols = tot_cols#tot_rows
        rot_rows = tot_rows#tot_cols
        rot_center[0] = rot_center[0] + (rot_cols * (rot_center[0]<0) ) + st_col  
        rot_center[1] = rot_center[1] + (rot_rows * (rot_center[1]<0) ) + st_row 
        return rot_center

    def display_control_mechanism_in_action(self, bot_loc, path, img_shortest_path, bot_localizer, frame_disp):
        Doing_pt = 0
        Done_pt = 0

        path_i = self.path_iter
        
        # Circle to represent car current location
        img_shortest_path = cv2.circle(img_shortest_path, bot_loc, 3, (0,0,255))

        if ( (type(path)!=int) and ( path_i!=(len(path)-1) ) ):
            curr_goal = path[path_i]
            # Mini Goal Completed
            if path_i!=0:
                img_shortest_path = cv2.circle(img_shortest_path, path[path_i-1], 3, (0,255,0),2)
                Done_pt = path[path_i-1]
            # Mini Goal Completing   
            img_shortest_path = cv2.circle(img_shortest_path, curr_goal, 3, (0,140,255),2)
            Doing_pt = curr_goal
        else:
            # Only Display Final Goal completed
            img_shortest_path = cv2.circle(img_shortest_path, path[path_i], 10, (0,255,0))
            Done_pt = path[path_i]

        if Doing_pt!=0:
            Doing_pt = self.bck_to_orig(Doing_pt, bot_localizer.transform_arr, bot_localizer.rot_mat_rev)
            frame_disp = cv2.circle(frame_disp, (int(Doing_pt[0]),int(Doing_pt[1])), 3, (0,140,255),2)   
         
        if Done_pt!=0:
            Done_pt = self.bck_to_orig(Done_pt, bot_localizer.transform_arr, bot_localizer.rot_mat_rev)
            if ( (type(path)!=int) and ( path_i!=(len(path)-1) ) ):
                pass
                #frame_disp = cv2.circle(frame_disp, (int(Done_pt[0]),int(Done_pt[1])) , 3, (0,255,0),2)   
            else:
                frame_disp = cv2.circle(frame_disp, (int(Done_pt[0]),int(Done_pt[1])) , 10, (0,255,0))  

        st = "Total Mini Goals = ( {} ) , Current Mini Goal = ( {} )".format(len(path),self.path_iter)        
        
        frame_disp = cv2.putText(frame_disp, st, (bot_localizer.orig_X+50,bot_localizer.orig_Y-30), cv2.FONT_HERSHEY_PLAIN, 1.2, (0,0,255))
        if config.debug and config.debug_motionplanning:
            cv2.imshow("Planned_Path+Localized Car+Target_and_Achieved_Vertices",img_shortest_path)
        else:
            try:
                cv2.destroyWindow("Planned_Path+Localized Car+Target_and_Achieved_Vertices")
            except:
                pass

    @staticmethod
    def angle_n_dist(pt_a, pt_b):
        #Sim: Front Y, right x pos
        #Image: Back Y, left x pos
        #For consistency, subtract first point Y axis with second point Y axis
        error_x = pt_b[0] - pt_a[0]
        error_y = pt_a[1] - pt_b[1]
        #Distance and angle for image
        distance = sqrt(pow((error_x), 2) + pow((error_y), 2))

        #Angle between two points
        angle = atan2(error_y, error_x)
        #rad to deg
        angle_deg = degrees(angle)
        if(angle_deg>0):
            return (angle_deg),distance
        else:
            return (angle_deg + 360),distance

    def check_gtg_status(self, angle_to_turn, distance_to_goal):

        #change in angle over last iteration
        change_angle_to_turn = abs(angle_to_turn-self.prev_angle_to_turn)
        if((abs(angle_to_turn)>5) and (change_angle_to_turn<0.4) and (not self.trigger_backpedaling)):
            self.angle_not_changed += 1
            #if angle is not changed for a while, trigger backpedaling
            if(self.angle_not_changed>200):
                self.trigger_backpedaling = True
        else:
            self.angle_not_changed = 0
        print("[prev,change,not_changed_iter,self.trigger_backpedaling] = [{:.1f},{:.1f},{},{}]"
        .format(self.prev_angle_to_turn,change_angle_to_turn,self.angle_not_changed,self.trigger_backpedaling))
        self.prev_angle_to_turn = angle_to_turn

        change_dist = abs(distance_to_goal - self.Prev_distance_to_goal)

        if((abs(distance_to_goal)>5) and (change_dist<1.2) and (not self.trigger_backpedaling)):
            self.dist_not_changed += 1
            #For a significant time dist is not changed
            if (self.dist_not_changed>100):
                self.trigger_backpedaling = True
        else:
            self.dist_not_changed = 0
        print("[prev_d,change_d,not_changed_iter,self.trigger_backpedaling] = [{:.1f},{:.1f},{},{}] "
        .format(self.Prev_distance_to_goal,change_dist,self.dist_not_changed,self.trigger_backpedaling))
        self.Prev_distance_to_goal = distance_to_goal  

        change_goal = self.prev_path_iter - self.path_iter

        if ((change_goal==0) and (distance_to_goal<30)):
            self.goal_not_changed += 1
            #for significant time goal not changed, trigger nxtpt
            if(self.goal_not_changed>500):
                self.trigger_nxtpt = True
        #If mini goal is not changing for a long time
        else:
            self.goal_not_changed_long = 0
            self.goal_not_changed = 0
        print("[prev_g,change_g,not_changed_iter] = [{:.1f},{:.1f},{}] "
        .format(self.prev_path_iter,change_goal,self.goal_not_changed))
        self.prev_path_iter = self.path_iter

    @staticmethod
    def dist(pt_a, pt_b):
        error_x = pt_b[0] - pt_a[0]
        error_y = pt_a[1] - pt_b[1]
        return (sqrt(pow((error_x),2) + pow((error_y),2)))
    
    def get_suitable_nxtpt(self, car_loc, path):
        extra_i = 1
        test_goal = path[self.path_iter+extra_i]

        while(self.dist(car_loc, test_goal)<20):
            extra_i+=1
            test_goal = path[self.path_iter+extra_i]
        print("Loading {} pt".format(extra_i))
        self.path_iter = self.path_iter + extra_i - 1

    def go_to_goal(self, bot_loc, path, velocity, velocity_publisher):
        angle_to_goal,distance_to_goal = self.angle_n_dist(bot_loc, (self.goal_pose_x,self.goal_pose_y))
        angle_to_turn = angle_to_goal - self.bot_angle

        speed = interp(distance_to_goal, [0,100], [0.2,1.5])
        self.curr_speed = speed
        angle = interp(angle_to_turn,[-360,360], [-4,4])
        self.curr_angle = angle

        print("Angle_2_Goal: {} Angle_2_Turn: {} Angle[Sim]: {}".format(angle_to_goal,angle_to_turn,abs(angle)))
        print("Distance_2_Goal: ",distance_to_goal)

        if self.goal_not_reached_flag:
            self.check_gtg_status(angle_to_turn, distance_to_goal)

        #If car is far away, turn towards goal
        if (distance_to_goal >= 2):
            velocity.angular.z = angle

        #Slow down the car if the turn angle is high
        if abs(angle) < 0.4:
            velocity.linear.x = speed
        elif (abs(angle) < 0.8):
            velocity.linear.x = 0.02
        else:
            velocity.linear.x = 0.0

        #Backpedal trigger
        if self.trigger_backpedaling:
            print("------- Backpedaling: ", self.backpedaling, " -----------")
            if self.backpedaling == 0:
                self.trigger_nxtpt = True
            velocity.linear.x = -0.16
            velocity.angular.z = angle
            self.backpedaling += 1
            #Stop after 100 iterations
            if self.backpedaling == 100:
                self.trigger_backpedaling = False
                self.backpedaling = 0
                print("------- Backpedaling Done -----------")

        #Keep publishing vel until reaching end
        if(self.goal_not_reached_flag) or (distance_to_goal<=1):
            velocity_publisher.publish(velocity)
        
        #print("Total Vertices = ( {} ) , Current Vertex Target = ( {} )".format(len(path), self.path_iter))

        #If car is within reasonable distance of mini goal
        if((distance_to_goal <= 8) or self.trigger_nxtpt):
            if self.trigger_nxtpt:
                if self.backpedaling:
                    #Look for appropriate mini-goal
                    self.get_suitable_nxtpt(bot_loc, path)
                self.trigger_nxtpt = False

            velocity.linear.x = 0.0
            velocity.angular.z = 0.0
            if self.goal_not_reached_flag:
                velocity_publisher.publish(velocity)

            #Reached the end
            if self.path_iter == (len(path)-1):
                if self.goal_not_reached_flag:
                    self.goal_not_reached_flag = False
                    #Play music
                    pygame.mixer.music.load(os.path.abspath('Documents/GitHub/Ros2_Path_Planning/Path_Planning_ws/src/maze_bot/resource/Goal_reached.wav'))
                    pygame.mixer.music.play()
            else:
                    # Iterate over the next mini-goal
                    self.path_iter += 1
                    self.goal_pose_x = path[self.path_iter][0]
                    self.goal_pose_y = path[self.path_iter][1]
                    print("Current Goal (x,y): ( {} , {} )".format(path[self.path_iter][0],path[self.path_iter][1]))
                    
                    if pygame.mixer.music.get_busy() == False:
                        pygame.mixer.music.play()
    

    def navigate_path(self, bot_loc, path, velocity, velocity_publisher):
        # If valid path found
        if (type(path)!=int):
            # Trying to reach first mini-goal
            if (self.path_iter == 0):
                self.goal_pose_x = path[self.path_iter][0]
                self.goal_pose_y = path[self.path_iter][1]
        
        if (self.count > 20):
            if not self.angle_relation_computed:
                #Stop car to compute angle
                velocity.linear.x = 0.0
                velocity_publisher.publish(velocity)

                self.bot_angle, _= self.angle_n_dist(self.init_loc, bot_loc)
                self.bot_angle_init = self.bot_angle

                self.bot_angle_rel = self.bot_angle_s - self.bot_angle
                self.angle_relation_computed = True
            
            else:
                #Bot angle in image
                self.bot_angle = self.bot_angle_s - self.bot_angle_rel 

                print("\n\nCar_angle(Image_Frm_Relation): {} Imgage-Simulation_Relation: {} Car_Angle_(Simulation): {}".format(self.bot_angle,self.bot_angle_rel,self.bot_angle_s))
                print("Car_Angle_Initial_(Image): ",self.bot_angle_init)
                print("Car_Location: {}".format(bot_loc))

                ##Traverse through path
                self.go_to_goal(bot_loc, path, velocity, velocity_publisher)
        #Start of motion planner
        else:
            if not self.pt_i_taken:
                self.init_loc = bot_loc
                self.pt_i_taken = True
            #Jog car
            velocity.linear.x = 1.0
            velocity_publisher.publish(velocity) 

            self.count += 1 
