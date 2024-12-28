import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
#import os
#os.chdir(os.path.join(os.getcwd(), "Lecture1"))


#%% Data structures
class RandomList(list):
    '''
    Parent class for Stack and Queue data structures.
    When appending multiple items at once (or instantiating a non-empty
    RandomList) order of items is randomized.
    '''
    
    def __init__(self, elements: list = []):
        np.random.shuffle(elements)
        super().__init__(elements)
        
    def __add__(self, elements: list = []):
        np.random.shuffle(elements)
        for e in elements:
            self.append(e)
        return self
    
    def __iadd__(self, elements: list = []):
        np.random.shuffle(elements)
        for e in elements:
            self.append(e)
        return self
        
   
class Stack(RandomList):
    
    def __init__(self, elements: list = []):
        super().__init__(elements)
    
    def out(self):
        '''Extracts most recently added element.'''
        return self.pop()


class Queue(RandomList):
    
    def __init__(self, elements: list = []):
        super().__init__(elements)
    
    def out(self):
        '''Extracts oldest element.'''
        return self.pop(0)


#%% Maze class
class MazeGraph:
    '''Builds a maze graph out of df containing maze edges.'''
    
    def __init__(self, df, layout_seed: int = 0,
                 start: str = 'START', goal: str = 'GOAL'):
        self.graph = nx.Graph()    
        for _, row in df.iterrows():
            self.graph.add_edge(row.tile0, row.tile1)
        self.pos = nx.spring_layout(self.graph, seed=layout_seed)
        
        self.colormap = []
        for node in self.graph:
            if node == start:
                self.colormap.append('green')
                continue
            if node == goal:
                self.colormap.append('red')
                continue
            self.colormap.append('orange')
        
        
    def show(self):
        nx.draw(self.graph, node_color=self.colormap,
                with_labels=True, pos=self.pos, font_size=15,
                width=2, node_size=500)
        plt.show()


#%% Search classes
class Node():
    '''Single node containing current node state, parent and connections.'''
    
    def __init__(self, state: str, parent: str, connections: list):
        self.state = state
        self.parent = parent
        self.connections = connections
        
    def __repr__(self):
        return f'Node={self.state}'
    
    def info(self):
        return f'Node class with state: {self.state}, parent: {self.parent} '\
               f'and connections: {self.connections}.'


class Search():
    '''
    Search algorithm. Variant (DSF, BSF) is determined by data structure used.
    '''
    
    @staticmethod
    def find_nodes(node_name: str) -> list:
        '''Finds all nodes connected to node of name node_name.'''
        nodes = list(df[df.tile0 == node_name].tile1)
        # Check for edges both ways
        nodes += list(df[df.tile1 == node_name].tile0)
        # Avoid repetitions
        nodes = list(set(nodes))
        return nodes
    
    def __init__(self, df, dstructure=Stack,
                 start: str = 'START', goal: str = 'GOAL'):
        '''Get connections to start node.'''
        start_nodes = Search.find_nodes(start)
        
        self.frontier = dstructure([Node(start, None, start_nodes)])
        self.dstructure = dstructure
        self.explored = []
        self.edges = df
        
        self.start = start
        self.goal = goal
    
    def search(self, limit: int = 100):
        '''
        Full search algorithm. Will search until self.explore finds a solution
        or limit of loops is breached. If solution is found, returns number of
        explorations, length of solution and the solution
        (a tuple containing the steps towards goal).
        A RecursionError is raised if there's no solution (nodes are exhausted
        without a goal) or limit of explorations is breached.
        '''
        i = 1
        solution = self.explore()
        while solution == None:
            if i > limit:
                raise RecursionError("Limit of explorations "
                                    f"breached ({i} > {limit}).")
            if not self.frontier:
                raise RecursionError(f"DEAD END: Explored {i} nodes without "
                                      "finding a solution.")
                
            solution = self.explore()
            i += 1
        
        return i, len(solution), solution
            
    def explore(self):
        '''
        Explores a frontier node. If node is goal returns solution, else None.
        '''
        node = self.frontier.out()
        if node.state == self.goal:
            return self.solve(node)
        
        self.explored.append(node.state)
        self.add_frontier(node.connections, parent=node)
        
        return None
    
    def add_frontier(self, connections: list, parent: Node):
        '''Adds parent connections to frontier as new nodes.'''
        children = []
        # Avoids repetitions on connections
        connections = list(set(connections))
        # Avoids repetitions on frontier
        frontier_states = [f.state for f in self.frontier]
        for child in connections:
            if child in self.explored or child in frontier_states:
                continue
            edges = Search.find_nodes(child)
            children.append(Node(child, parent, list(edges)))
        
        self.frontier += children
    
    def solve(self, node: Node):
        '''
        Having found goal node returns tuple containing steps for reaching
        this node.
        '''
        solution = []
        solution.append(node)
        
        while node.parent != None:
            node = node.parent
            solution.append(node)
        
        return tuple(reversed(solution))
    
    def __repr__(self):
        return f'Search with {self.dstructure}.'


#%% Maze (maze2, layout_seed=8)
filename = 'maze2.csv'
df = pd.read_csv(filename)
start = 'M'
goal = 'A'

maze = MazeGraph(df, layout_seed=8, start=start, goal=goal)
maze.show()


#%% Solve maze
np.random.seed(137)

filename = 'maze2.csv'
df = pd.read_csv(filename)

maze_solver = Search(df, dstructure=Stack, start=start, goal=goal)
solution = maze_solver.search(limit=100)
print("Solution:", solution)
