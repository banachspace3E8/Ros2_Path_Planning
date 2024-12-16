#Shishir Khanal
#12/12/2014
#Script to map the maze

import cv2
import numpy as np

draw_intrstpts = True
debug_mapping = False

class Graph():
    def __init__(self):
         self.graph = {}
         self.start = 0
         self.end = 0
    
    def add_vertex(self, vertex, neighbour=None, case=None, cost=None):
        if vertex in self.graph.keys():
            self.graph[vertex][neighbour] = {}
            self.graph[vertex][neighbour]["case"] = case
            self.graph[vertex][neighbour]["cost"] = cost
        else:
            self.graph[vertex] = {}
            self.graph[vertex]["case"] = case

    def displaygraph(self):
        for key,value in self.graph.items():
            print("key {} has value {}".format(key, value))

class bot_mapper():

    def __init__(self):
        self.graphified = False
        #Cropping Control for removing maze boundary
        self.crp_amt = 5

        self.Graph = Graph()
        #State variales to define the connection status of each vertex
        self.connected_left = False
        self.connected_upleft = False
        self.connected_up = False
        self.connected_upright = False
        #Colored Maze for displaying connection between nodes
        self.maze_connect = []

    #Connect curr_node to its neighoburs in immediate left(left -> topright) region
    def connect_neighbours(self, maze, node_row, node_col, case, step_l=1, step_up=0, totl_cnncted=0):
        curr_node = (node_row, node_col)
        if (maze[node_row - step_up][node_col - step_l] > 0):
            neighbor_node = (node_row,node_col)
            if neighbor_node in self.Graph.graph.keys():
                neighbors_case = self.Graph.graph[neighbor_node]["case"]
                cost = max(abs(step_l),abs(step_up))
                totl_cnncted += 1

                self.Graph.add_vertex(curr_node, neighbor_node, neighbors_case, cost)
                self.Graph.add_vertex(neighbor_node, curr_node, case, cost)
                print("\nConnected {} to {} with Case [step_l, step_up] = [ {} , {} ] & Cost -> {}".format(curr_node, neighbor_node, step_l, step_up, cost))

                if not self.connected_left:
                    self.display_connected_nodes(curr_node, neighbor_node, "LEFT", (0, 0, 255))
                    self.connected_left = True
                    step_l = 1
                    step_up = 1
                    self.connect_neighbours(maze, node_row, node_col, case, step_l, step_up, totl_cnncted)
                if not self.connected_upleft:
                    self.display_connected_nodes(curr_node, neighbor_node, "UPLEFT", (0, 128, 255))
                    #Vertex has connedted to its upleft neighbor
                    self.connected_upleft = True
                    step_l = 0
                    step_up = 1
                    self.connect_neighbours(maze, node_row, node_col, case, step_l, step_up, totl_cnncted)
                if not self.connected_up:
                    self.display_connected_nodes(curr_node, neighbor_node, "UP", (0, 255, 0))
                    #Vertex has connected to its up neighbor
                    self.connected_up = True
                    #Check top-right route now
                    step_l = -1
                    step_up = 1
                    self.connect_neighbours(maze, node_row, node_col, case, step_l, step_up, totl_cnncted)
                if not self.connected_upright:
                    self.display_connected_nodes(curr_node, neighbor_node, "UPRIGHT", (255, 0, 0))
                    #Vertex has connected to its up-right neighbor
                    self.connected_upright = True
                    #no need to call connect neighbours as the cycle is completed an connections are done
            if not self.connected_upright:
                if not self.connected_left:
                    step_l += 1
                elif not self.connected_upleft:
                    #Look a little more diagonally upleft
                    step_l += 1
                    step_up += 1
                elif not self.connected_up:
                    step_up += 1
                elif not self.connected_upright:
                    step_l -= 1
                    step_up -= 1
                self.connect_neighbours(maze, node_row, node_col, case, step_l, step_up, totl_cnncted)                  
        else:
            #No path in the direction, cycle to the next direction
            if not self.connected_left:
                #Basically there is a wall on the left so start looking up left
                self.connected_left = True
                #Looking upleft now
                step_l = 1
                step_up = 1
                self.connect_neighbours(maze, node_row, node_col, case, step_l, step_up, totl_cnncted)
            elif not self.connected_upleft:
                #There is a wall upleft so just start looking up
                self.connected_upleft = True
                step_l = 0
                step_up = 1
                self.connect_neighbours(maze, node_row, node_col, case, step_l, step_up, totl_cnncted)
            elif not self.connected_up:
                #There is a wall upleft so just start looking up right
                self.connected_upleft = True
                step_l = -1
                step_up = 1
                self.connect_neighbours(maze, node_row, node_col, case, step_l, step_up, totl_cnncted)
            elif not self.connected_upright:
                #There is a wall upleft so just start looking up right
                self.connected_upleft = True
                step_l = 0
                step_up = 0    
                return 

    #Draw triangle around a point
    @staticmethod
    def triangle(image, ctr_pt, radius, colour=(0,255,255), thickness = 2):
        #Polygon corner points
        pts = np.array([ [ctr_pt[0]         ,  ctr_pt[1]-radius],
                        [ctr_pt[0]-radius   ,  ctr_pt[1]+radius],
                        [ctr_pt[0]+radius  ,   ctr_pt[1]+radius]
                        ],np.int32
                    )
        pts = pts.reshape((-1, 1, 2))

        image = cv2.polylines(image, [pts], True, colour, thickness)
        return image

    @staticmethod
    def get_surround_pixel_intensities(maze, curr_row, curr_col):
        maze = cv2.threshold(maze, 1, 1, cv2.THRESH_BINARY)[1]

        rows = maze.shape[0]
        cols = maze.shape[1]
        #State vars, if our point is a boundary condition
        top_row = False
        btm_row = False
        lft_col = False
        rgt_col = False
        
        if (curr_row == 0):
            top_row = True
        if (curr_row == (rows-1)):
            #Botom row -> Row below not accessible
            btm_row = True
        if (curr_col == 0):
            #left col -> Col to left not accessible
            lft_col = True
        if (curr_col == (cols-1)):
            #Right col -> col to right not accessible
            rgt_col = True
        
        if (top_row or lft_col):
            top_left = 0
        else:
            top_left = maze[curr_row - 1][curr_col - 1]

        if (top_row or rgt_col):
            top_rgt = 0
        else:
            top_rgt = maze[curr_row - 1][curr_col + 1]

        if (btm_row or lft_col):
            btm_left = 0
        else:
            btm_left = maze[curr_row + 1][curr_col - 1]

        if (btm_row or rgt_col):
            btm_rgt = 0
        else:
            btm_rgt = maze[curr_row + 1][curr_col + 1]
        
        if(top_row):
            top = 0
        else:
            top = maze[curr_row - 1][curr_col]

        if(rgt_col):
            rgt = 0
        else:
            rgt = maze[curr_row][curr_col + 1]
        
        if(btm_row):
            btm = 0
        else:
            btm = maze[curr_row + 1][curr_col]
        
        if(lft_col):
            lft = 0
        else:
            lft = maze[curr_row][curr_col - 1]

        no_of_pathways = ( top_left     + top       + top_rgt +
                            lft         + 0         + rgt     +
                            btm_left    + btm       + btm_rgt
                            )
        if no_of_pathways > 2:
            print("  [ top_left , top      , top_rgt  ,lft    , rgt      , btm_left , btm      , btm_rgt   ] \n [ ",
                  str(top_left)," , ",str(top)," , ",str(top_rgt)," ,\n   ",str(lft)," , ","-"," , ",str(rgt)," ,\n   ",
                  str(btm_left)," , ",str(btm)," , ",str(btm_rgt)," ] ")
            print("\nno_of_pathways [row,col]= [ ",curr_row," , ",curr_col," ] ",no_of_pathways) 

        return top_left, top, top_rgt, rgt, btm_rgt, btm, btm_left, lft, no_of_pathways

    # Reset state parameters of each vertex connection
    def reset_connct_paramtrs(self):
        # Reseting member variables to False initially when looking for nodes to connect
        self.connected_left = False
        self.connected_upleft = False
        self.connected_up = False
        self.connected_upright = False

    def one_pass(self, maze):

        self.Graph.clear()

        #Initialize maze_connect with colored maze
        self.maze_connect = cv2.cvtColor(maze, cv2.COLOR_GRAY2BGR)
        cv2.namedWindow("Nodes Connected", cv2.WINDOW_FREERATIO)

        #Initialize counts of Interest Points
        turns = 0
        junc_3 = 0
        junc_4 = 0 
        #Converting maze to Colored for Identifying discovered Interest Points
        maze_bgr = cv2.cnvtColor(maze, cv2.COLOR_GRAY2BGR)
        #Create a window to display detected interest points
        cv2.namedWindow("Maze (Interest Points)", cv2.WINDOW_FREERATIO)
        rows = maze.shape[0]
        cols = maze.shape[1]

        for row in range(rows):
            for col in range(cols):
                if(maze[row][col] == 255):
                    if debug_mapping:
                        #Reinitialize maze connect with colored maze
                        self.maze_connect = cv2.cvtColor(maze, cv2.COLOR_GRAY2BGR)
                    #Probable IP => Find surrounding pixel intensities
                    top_left, top, top_rgt, rgt, btm_rgt, btm, btm_left, lft, paths = self.get_surround_pixel_intensities(maze.copy(), row, col)

                    if ((row == 0) or (row == (rows - 1)) or (col == 0) or (col == (cols - 1))):
                        if (row == 0):
                            #Start
                            maze_bgr[row][col] = (0, 128, 255)
                            cv2.imshow("Maze (Interest Points)", maze_bgr)
                            self.Graph.add_vertex((row,col), case="_Start_")
                            self.Graph.start = (row, col)
                        else:
                            #End (Maze Exit)
                            maze_bgr[row][col] = (0, 255, 0)
                            cv2.imshow("Maze (Interest Points)", maze_bgr)
                            self.Graph.add_vertex((row,col), case="_End_")
                            self.Graph.end = (row, col)
                            self.reset_connct_paramtrs()
                            self.connect_neighbours(maze, row, col, "_End_")
                    #Check if it is a dead point
                    elif(paths == 1):
                        crop = maze[row-1:row+2, col-1:col+2]
                        print(" ** [Dead End] ** \n", crop)
                        maze_bgr[row][col] = (0, 0, 255) #Red Color
                        if draw_intrstpts:
                            maze_bgr = cv2.circle(maze_bgr, (col, row), 10, (0,0,255), 2)
                        cv2.imshow("Maze (Interest Points)", maze_bgr)
                        self.Graph.add_vertex((row,col),case="_DeadEnd_")
                        self.reset_connct_paramtrs()
                        self.connect_neighbours(maze, row, col, "_DeadEnd_")
                    #Check if it's a Turn or an ordinary path
                    elif(paths == 2):
                        crop = maze[row-1:row+2,col-1:col+2]
                        nzero_loc = np.nonzero(crop > 0)
                        nzero_ptA = (nzero_loc[0][0],nzero_loc[1][0])
                        nzero_ptB = (nzero_loc[0][2], nzero_loc[1][2])
                        if not (((2 - nzero_ptA[0]) == nzero_ptB[0]) and
                                ((2 - nzero_ptA[1]) == nzero_ptB[1])):
                            maze_bgr[row][col] = (255, 0, 0)
                            #if draw_intrstpts:
                                #maze_bgr = cv2.circle(maze_bgr, (col, row), 10, (255, 0, 0))
                            cv2.imshow("Maze (Interest Points)", maze_bgr)
                            #Adding found vertex to graph
                            self.Graph.add_vertex((row,col),case="_Turn_")
                            #Connect vertex to its neighbour if any
                            self.reset_connct_paramtrs()
                            self.connect_neighbours(maze, row, col, "_Turn_")
                            turns += 1
                    #Check if it is a 3 junction or 4 junction
                    elif(paths > 2):
                        if(paths == 3):
                            maze_bgr[row][col] = (255, 244, 128)
                            if draw_intrstpts:
                                maze_bgr =self.triangle(maze_bgr, (col, row), 10, (144, 244, 128))
                            cv2.imshow("Maze (Interest Points)", maze_bgr)
                            #Adding found vertex to graph
                            self.Graph.add_vertex((row,col),case="_3-Junc_")
                            #Connect vertex to its neighbour if any
                            self.reset_connct_paramtrs()
                            self.connect_neighbours(maze, row, col, "_3-Junc_")
                            junc_3 += 1
                        else:
                            maze_bgr[row][col] = (128, 0, 128)
                            if draw_intrstpts:
                                maze_bgr =self.rectangle(maze_bgr, (col - 10, row - 10), (col + 10, row + 10), (255, 140, 144), 2)
                            cv2.imshow("Maze (Interest Points)", maze_bgr)
                            #Adding found vertex to graph
                            self.Graph.add_vertex((row,col),case="_4-Junc_")
                            #Connect vertex to its neighbour if any
                            self.reset_connct_paramtrs()
                            self.connect_neighbours(maze, row, col, "_4-Junc_")
                            junc_4 += 1
        print("\nInterest Points !!! \n[ Turns , 3_Junc , 4_Junc ] [ ",turns," , ",junc_3," , ",junc_4," ] \n")

    def graphify(self, extracted_maze):
        
        if not self.graphified:
            cv2.imshow("Extracted_Maze [MazeConverter]", extracted_maze)
            thinned = cv2.ximgproc.thinnig(extracted_maze)
            cv2.imshow('Maze (thinned)', thinned)

            #Dilate and thin again to minimize unnecessary interest points (i.e. turns)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
            thinned_dilated = cv2.morphologyEx(thinned, cv2.MORPH_DILATE, kernel)
            _, bw2 = cv2.threshold(thinned_dilated, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            thinned = cv2.ximgproc.thinning(bw2)
            cv2.imshow('Maze (thinned*2)', thinned)

            thinned_cropped = thinned[self.crp_amt:thinned.shape[0] - self.crp_amt,
                                        self.crp_amt:thinned.shape[1] - self.crp_amt]
            cv2.imshow('Maze (thinned*2)(Cropped)', thinned_cropped)
            #Overlay found path on Maze Occupancy grid
            extracted_maze_cropped = extracted_maze[self.crp_amt:extracted_maze.shape[0] - self.crp_amt,
                                                self.crp_amt:extracted_maze.shape[1] - self.crp_amt]
            extracted_maze_cropped = cv2.cvtColor(extracted_maze_cropped, cv2.COLOR_GRAY2BGR)
            extracted_maze_cropped[thinned_cropped>0] = (0, 255, 255)
            cv2.imshow('Maze (thinned*2)(Cropped)(Path_Overlayed)', extracted_maze_cropped)
            #Extract interest points from maze
            self.one_pass(thinned_cropped)