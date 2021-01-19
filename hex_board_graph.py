import networkx as nx
import matplotlib.pyplot as plt


class HexBoardGraph:
    def __init__(self, neighbour_dict, board):
        self.graph = nx.Graph()
        self.create_graph(neighbour_dict, board)

    def create_graph(self, neighbour_dict, board):
        edges = self.get_edges(neighbour_dict, board)
        print(edges)
        self.graph.add_edges_from(edges)
        nx.draw(self.graph)
        plt.show()

    def get_edges(self, neighbour_dict, board):
        """
        neighbour_dict is a dictionary for hexagonal neighbours for a board element
            key = (row of element, col of element)
            value = [(row of neighbour 1, col of neighbour 1), ..., (row of neighbour n, col of neighbour n)]

        board is a list of lists structure, where each element is a tuple of an element id (id is needed for graph visualising)
        and the coordinates of the element. So for a triangle board of size 3, the resulting board will look like this:
            [
                [(0, (0,0))],
                [(1, (1,0)), (2, (1,1))],
                [(3, (2,0)), (4, (2,1)), (5, (2,2))]
            ] 
        """
        edges = []

        for r, row in enumerate(board):
            for c, elem in enumerate(row):

                # Get list of neighbour coordinates on the form (row, col)
                neighbours = neighbour_dict[elem[1]]
                for n in neighbours:
                    # Get element id of base_node
                    base_node = board[r][c][0]
                    neighbour_node = board[n[0]][n[1]][0]
                    edges.append((base_node, neighbour_node))

        return edges
