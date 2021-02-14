from .board import BoardType

import networkx as nx
import matplotlib.pyplot as plt



class HexBoardGraph:
    def __init__(self, neighbour_dict, board, board_type):
        self.graph = nx.Graph()
        self.pos = {}
        self.create_graph(neighbour_dict, board, board_type)

    def create_graph(self, neighbour_dict, board, board_type):
        edges = self.get_edges(neighbour_dict, board)
        self.graph.add_edges_from(edges)
        self.get_node_pos(board_type)

    def get_edges(self, neighbour_dict, board):
        """
        neighbour_dict is a dictionary for hexagonal neighbours for a board element
            key = (row of element, col of element)
            value = [(row of neighbour 1, col of neighbour 1), ..., (row of neighbour n, col of neighbour n)]

        board is a list of lists structure, where each element is a tuple of the coordinates of the element. 
        So for a triangle board of size 3, the resulting board will look like this:
            [
                [(0,0)],
                [(1,0), (1,1)],
                [(2,0), (2,1), (2,2)]
            ] 
        """
        edges = []

        for r, row in enumerate(board):
            for c, coordinates in enumerate(row):

                # Get list of neighbour coordinates on the form (row, col)
                neighbours = neighbour_dict[(r, c)]
                for direction, neighbour in neighbours.items():
                    base_node = (r, c)
                    edges.append((base_node, neighbour))
        return edges
    
    def get_node_pos(self, board_type):
        """
        Generate dictionary of positions of nodes, so graph visualisation is consistent and okayish pretty...
        Triangle-boards look like triangles, and diamond-boards look like diamonds at least.
        """

        for node in self.graph:
            r, c = node
            if board_type == BoardType.TRIANGLE:
                x_pos = c - 0.5 * r
                y_pos = 1 - 0.1 * r
            elif board_type == BoardType.DIAMOND:
                x_pos = c - r
                y_pos = 1 - 0.05 * r - 0.05 * c
            self.pos[node] = (x_pos, y_pos)
    
