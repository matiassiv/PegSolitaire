from hex_board_graph import HexBoardGraph


class BoardType:
    TRIANGLE = 1
    DIAMOND = 2


class Board:
    # Some arbitrary default val, would add to config file instead
    def __init__(self, board_type=BoardType.TRIANGLE, size=4):
        self.board_type = board_type
        self.size = size
        self.board = []
        self.neighbour_dict = {}
        self.generate_board()
        self.generate_neighbours()
        print(self.neighbour_dict)

    def generate_board(self):
        """
        The game board is a list of lists structure, where each element is a tuple of an element id (kan muligens sløyfes)
        and the coordinates of the element. So for a triangle board of size 3, the resulting board will look like this:

        [
            [(0, (0,0))],
            [(1, (1,0)), (2, (1,1))],
            [(3, (2,0)), (4, (2,1)), (5, (2,2))]
        ] 
        """

        if self.board_type == BoardType.TRIANGLE:
            for i in range(self.size):
                # Kind of obscure indexing, but this sums all numbers from 0 to i, which gives the first number for
                # the next row of the triangle board
                id_offset = (i*(i+1))//2
                row = []
                for j in range(i+1):
                    row.append((j+id_offset, (i, j)))
                self.board.append(row)

        else:
            for i in range(self.size):
                self.board.append([(j + i*self.size, (i, j))
                                   for j in range(self.size)])

    def generate_neighbours(self):

        if self.board_type == BoardType.TRIANGLE:

            for row in range(self.size):
                for col in range(row+1):
                    triangle_neighbour_indices = [
                        (row - 1, col),
                        (row, col + 1),
                        (row + 1, col + 1),
                        (row + 1, col),
                        (row, col - 1),
                        (row - 1, col - 1)
                    ]
                    valid_neighbours = []
                    for index in triangle_neighbour_indices:
                        if ((index[0] >= 0 and index[0] < self.size)
                                and (index[1] >= 0 and index[1] <= index[0])):
                            valid_neighbours.append(index)

                    self.neighbour_dict[(row, col)] = valid_neighbours

        else:

            for row in range(self.size):
                for col in range(self.size):
                    diamond_neighbour_indices = [
                        (row - 1, col),
                        (row - 1, col + 1),
                        (row, col + 1),
                        (row + 1, col),
                        (row + 1, col - 1),
                        (row, col - 1)
                    ]

                    valid_neighbours = []
                    for index in diamond_neighbour_indices:
                        if ((index[0] >= 0 and index[0] < self.size)
                                and (index[1] >= 0 and index[1] < self.size)):
                            valid_neighbours.append(index)
                    self.neighbour_dict[(row, col)] = valid_neighbours

    def drawGraph(self):
        self.graph = HexBoardGraph(self.neighbour_dict, self.board)


test1 = Board(BoardType.TRIANGLE, 6)
test1.drawGraph()
