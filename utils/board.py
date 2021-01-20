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

    def generate_board(self):
        """
        The game board is a list of lists structure, where each element is a tuple of the coordinates of the element. 
        So for a triangle board of size 3, the resulting board will look like this:
            [
                [(0,0)],
                [(1,0), (1,1)],
                [(2,0), (2,1), (2,2)]
            ] 
        """

        if self.board_type == BoardType.TRIANGLE:
            for r in range(self.size):
                self.board.append([(r, c) for c in range(r + 1)])

        else:
            for r in range(self.size):
                self.board.append([(r, c) for c in range(self.size)])

    def generate_neighbours(self):

        if self.board_type == BoardType.TRIANGLE:

            for row in range(self.size):
                for col in range(row+1):
                    triangle_neighbour_indices = [
                        (row - 1, col),             #UP
                        (row, col + 1),             #RIGHT
                        (row + 1, col + 1),         #DOWN_RIGHT
                        (row + 1, col),             #DOWN
                        (row, col - 1),             #LEFT
                        (row - 1, col - 1)          #UP_LEFT
                    ]
                    valid_neighbours = {}
                    for i, index in enumerate(triangle_neighbour_indices):
                        if ((index[0] >= 0 and index[0] < self.size)
                                and (index[1] >= 0 and index[1] <= index[0])):
                            valid_neighbours[i] = index

                    self.neighbour_dict[(row, col)] = valid_neighbours

        else: #BoardType.DIAMOND

            for row in range(self.size):
                for col in range(self.size):
                    diamond_neighbour_indices = [
                        (row - 1, col),             #UP
                        (row - 1, col + 1),         #UP_RIGHT
                        (row, col + 1),             #RIGHT
                        (row + 1, col),             #DOWN
                        (row + 1, col - 1),         #DOWN_LEFT
                        (row, col - 1)              #LEFT   
                    ]

                    valid_neighbours = {}
                    for i, index in enumerate(diamond_neighbour_indices):
                        if ((index[0] >= 0 and index[0] < self.size)
                                and (index[1] >= 0 and index[1] < self.size)):
                            valid_neighbours[i] = index

                    self.neighbour_dict[(row, col)] = valid_neighbours
