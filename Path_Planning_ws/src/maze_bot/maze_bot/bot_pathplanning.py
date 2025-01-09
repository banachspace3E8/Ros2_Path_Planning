#Shishir Khanal
#12/12/2014
#Script to serch a feasible path in the map
import cv2
from numpy import sqrt
import numpy as np
from . import config

class bot_pathplanner():
    def __init__(self):
        self.DFS = DFS()
        self.dijkstra = Dijkstra()
        self.astar = a_star()

    @staticmethod
    def cords_to_pts(cords):
        return [cord[::-1] for cord in cords]

    def draw_path_on_maze(self, maze, shortest_path_pts, method):
        maze_bgr = cv2.cvtColor(maze, cv2.COLOR_GRAY2BGR)
        self.chosen_route = np.zeros_like(maze_bgr)

        rang = list(range(0,254,25))

        depth = maze.shape[0]
        for i in range(len(shortest_path_pts) - 1):
            per_depth = (shortest_path_pts[i][1])/depth
            color = (
                int(255 * (abs(per_depth+(-1*(per_depth>0.5)))*2)),
                int(255 * per_depth),
                int(255 * (1-per_depth))
            )
            cv2.line(maze_bgr, shortest_path_pts[i], shortest_path_pts[i+1], color)
            cv2.line(self.chosen_route, shortest_path_pts[i], shortest_path_pts[i+1], color, 3)

        img_str = "maze (Found Path) [" + method + "]"
        if config.debug and config.debug_pathplanning:
            cv2.namedWindow(img_str, cv2.WINDOW_FREERATIO)
            cv2.imshow(img_str, maze_bgr)
        
        if method == "Dijkstra":
            self.dijkstra.shortest_path_overlayed = maze_bgr
        elif method == "a_star":
            self.astar.shortest_path_overlayed = maze_bgr
            
        self.img_shortest_path = maze_bgr.copy()
    
    def find_and_display_path(self, graph, start, end, maze, method="DFS"):
        Path_str = "Path"
        
        if method == "DFS":
            paths = self.DFS.get_paths(graph, start, end)
            path_to_display = paths[0]
        elif method == "DFS_Shortest":
            paths_and_costs = self.DFS.get_paths_cost(graph, start, end)
            paths = paths_and_costs[0]
            costs = paths_and_costs[1]
            min_cost = min(costs)
            path_to_display = paths[costs.index(min_cost)]
        elif method == "Dijkstra":
            if not self.dijkstra.shortestpath_found:
                print("Finding Shortest Routes using Dijkstra")
                self.dijkstra.find_best_routes(graph, start, end)
            path_to_display = self.dijkstra.shortest_path
            Path_str = "Shortest "+ Path_str
        elif method == "A_star":
            if not self.astar.shortestpath_found:
                print("Finding Shortest Routes using A*")
                self.astar.find_best_routes(graph, start, end)
            path_to_display = self.astar.shortest_path
            Path_str = "Shortest " + Path_str

        pathpts_to_display = self.cords_to_pts(path_to_display)
        #print("Found path pts = {}".format(pathpts_to_display))
        self.draw_path_on_maze(maze, pathpts_to_display, method)
        #cv2.waitKey(0)
    


class DFS():
    #Recursive approach

    @staticmethod
    def get_paths(graph, start, end, path=[]):
        #Paths that leads us to goal node
        path = path + [start]
        # Define base case
        if (start == end):
            return [path]
        #start not part of graph
        if start not in graph.keys():
            return []
        #List to store all possible paths from start to end
        paths = []

        # Break down complex probem into simpler problems
        for node in graph[start].keys():
            #Once encountered base condition -> Roll back answer to solver subproblem
            #Cheking if not already traversed and not a "case" key
            if (node not in path) and (node !="case"):
                new_paths = DFS.get_paths(graph, node, end, path)
                for p in new_paths:
                    paths.append(p)

        return paths
    
    #Find all feasible paths and their respective cost
    @staticmethod
    def get_paths_cost(graph, start, end, path=[], cost=0, trav_cost=0):
        path = path + [start]
        cost = cost + trav_cost

        #base case
        if start == end:
            return [path],[cost]
        if start not in graph.keys():
            return [],0
        
        #Store all possible paths from start to end and their costs
        paths = []
        costs = []

        #Retrieve connections for each node
        for node in graph[start].keys():
            #check not already traversed and not a "case" key
            if ((node not in path) and (node != "case")):
                new_paths, new_costs = DFS.get_paths_cost(graph, node, end, path, cost, graph[start][node]['cost'])
                
                for p in new_paths:
                    paths.append(p)
                for c in new_costs:
                    costs.append(c)

        return paths, costs


class Heap():
    def __init__(self):
        self.array = []
        self.size = 0
        self.size = 0
        self.posOfVertices = []

    def new_minHeap_node(self, v, dist):
        return ([v, dist])
    
    def swap_nodes(self, a, b):
        temp = self.array[a]
        self.array[a] = self.array[b]
        self.array[b] = temp

    def minHeapify(self,node_idx):
        smallest = node_idx
        left = (node_idx*2) + 1
        right = (node_idx*2) + 2

        if ((left<self.size) and (self.array[left][1]<self.array[smallest][1])):
            smallest = left
        if ((right<self.size) and (self.array[right][1]<self.array[smallest][1])):
            smallest = right

        if (smallest != node_idx):
            self.posOfVertices[self.array[node_idx][0]] = smallest
            self.posOfVertices[self.array[smallest][0]] = node_idx

            self.swap_nodes(node_idx, smallest)

            self.minHeapify(smallest)
    
    def extractmin(self):
        if self.size == 0:
            return
        root = self.array[0]

        lastNode = self.array[self.size - 1]
        self.array[0] = lastNode

        self.posOfVertices[root[0]] = self.size-1
        self.posOfVertices[lastNode[0]] = 0

        self.size -= 1
        self.minHeapify(0)
        return root

    def decreaseKey(self, vertex, dist):
        vertexidx = self.posOfVertices[vertex]
        self.array[vertexidx][1] = dist
        
        #Travel up while complete heap is not complete
        # While idx is valid and (Updated_key_idx < Parent_key_dist)
        while((vertexidx>0) and (self.array[vertexidx][1]<self.array[(vertexidx-1)//2][1])):
            #Update position and parent and curr node
            self.posOfVertices[self.array[vertexidx][0]] = (vertexidx-1)//2
            self.posOfVertices[self.array[(vertexidx-1)//2][0]] = vertexidx

            #swap curr_node w parent
            self.swap_nodes(vertexidx, (vertexidx-1)//2)
            # Navigate to parent and start process again
            vertexidx = (vertexidx-1)//2

    def isInMinHeap(self, v):
        if self.posOfVertices[v] < self.size:
            return True
        return False
    
class Dijkstra():
    def __init__(self):
        self.shortestpath_found = False
        self.shortest_path = []
        self.minHeap = Heap()
        self.dijiktra_nodes_visited = 0

        #save relationship between vertex and indices
        self.idxs2vrtxs = {}
        self.vrtxs2idxs = {}

    def ret_shortestroute(self, parent, start, end, route):
        route.append(self.idxs2vrtxs[end])
        if (end == start):
            return
        end = parent[end]
        self.ret_shortestroute(parent, start, end, route)
    
    def find_best_routes(self, graph, start, end):
        
        start_idx = [idx for idx,key in enumerate(graph.items()) if key[0]==start][0]
        #Store dist of each node
        dist = []
        #Found shortest subpath
        parent = []
        #Size of minHeap -> # of keys in graph
        self.minHeap.size = len(graph.keys())
        print(self.minHeap.size)
        for idx,v in enumerate(graph.keys()):
            #Init dist for all vertices to inf
            dist.append(1e7)

            self.minHeap.array.append(self.minHeap.new_minHeap_node(idx, dist[idx]))
            self.minHeap.posOfVertices.append(idx)

            #Initializing parent_nodes_list with -1 for all indices
            parent.append(-1)

            #Update dictionaries of vertices and their positions
            self.vrtxs2idxs[v] = idx
            self.idxs2vrtxs[idx] = v

        dist[start_idx] = 0
        self.minHeap.decreaseKey(start_idx, dist[start_idx])

        while (self.minHeap.size != 0):
            self.dijiktra_nodes_visited += 1
            curr_top = self.minHeap.extractmin()
            u_idx = curr_top[0]
            u = self.idxs2vrtxs[u_idx]

            for v in graph[u]:
                if v != "case":
                    v_idx = self.vrtxs2idxs[v]
                    
                    #if we have not fond shortest distance to v + new found dist < known dist -> Update dist
                    if (self.minHeap.isInMinHeap(v_idx) and (dist[u_idx] != 1e7) and
                    ((graph[u][v]["cost"] + dist[u_idx]) < dist[v_idx])):
                        dist[v_idx] = graph[u][v]["cost"] + dist[u_idx]
                        self.minHeap.decreaseKey(v_idx, dist[v_idx])
                        parent[v_idx] = u_idx
            #When end goal has already been visited, end the loop
            if (u == end):
                break
        
        shortest_path = []
        self.ret_shortestroute(parent, start_idx, self.vrtxs2idxs[end], shortest_path)

        #return path from start to end
        self.shortest_path = shortest_path[::-1]
        self.shortestpath_found = True


class a_star(Dijkstra):
    def __init__(self):
        #Use the same initialization from the parent class
        super().__init__()
        #Counter totrack total nodesvisited to reach goal
        self.astar_nodes_visited = 0

    @staticmethod
    def euc_d(a, b):
        return sqrt(pow(a[0]-b[0],2) + pow(a[1]-b[1],2))
    
    #Function overriding
    def find_best_routes(self, graph, start, end):
        #list created by list comphrension, tale first item
        start_idx = [idx for idx, key in enumerate(graph.items()) if key[0]==start][0]
        print("Index of search key: {}".format(start_idx))
        #Cost to reaching the node
        cost2node = []
        dist = []
        parent = []
        self.minHeap.size = len(graph.keys())

        for idx,v in enumerate(graph.keys()):
            dist.append(1e7)
            cost2node.append(1e7)
            self.minHeap.array.append(self.minHeap.new_minHeap_node(idx, dist[idx]))
            self.minHeap.posOfVertices.append(idx)
            parent.append(-1)
            self.vrtxs2idxs[v] = idx
            self.idxs2vrtxs[idx] = v
        cost2node[start_idx] = 0
        dist[start_idx] = cost2node[start_idx] + self.euc_d(start, end)
        self.minHeap.decreaseKey(start_idx, dist[start_idx])
        while(self.minHeap.size != 0):
            self.astar_nodes_visited += 1
            curr_top = self.minHeap.extractmin()
            u_idx = curr_top[0]
            u = self.idxs2vrtxs[u_idx]
            
            for v in graph[u]:
                if v!="case":
                    #print("Vertex adjacent to {} is {}".format(u,v))
                    v_idx = self.vrtxs2idxs[v]
                    if(self.minHeap.isInMinHeap(v_idx) and (dist[u_idx] != 1e7) and
                        ((graph[u][v]["cost"] + cost2node[u_idx]) < cost2node[v_idx])):
                        cost2node[v_idx] = graph[u][v]["cost"] + dist[u_idx]
                        dist[v_idx] = cost2node[v_idx] + self.euc_d(v, end)
                        self.minHeap.decreaseKey(v_idx, dist[v_idx])
                        parent[v_idx] = u_idx
            if(u == end):
                break

        shortest_path = []
        self.ret_shortestroute(parent, start_idx, self.vrtxs2idxs[end], shortest_path)
        self.shortest_path = shortest_path[::-1]
        self.shortestpath_found = True

