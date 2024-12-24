#Shishir Khanal
#12/12/2014
#Script to serch a feasible path in the map
import cv2
import numpy as np

class DFS():
    #Recursive approach

    @staticmethod
    def get_paths(graph, start, end, path=[]):
        path = path + [start]
        # Define base case
        if (start == end):
            return [path]
        
        #Handle boundary case [start not part of graph]
        if start not in graph.keys():
            return []

        #List to store all possible paths from start to end
        paths = []

        # Break down complex probem into simpler problems
        for node in graph[start].keys():
            #Once encountered base condition -> Roll back answer to solver subproblem
            #Cheking if not already traversed and not a "case" key
            new_paths = DFS.get_paths(graph, node, end, path)
            for p in new_paths:
                paths.append(p)

