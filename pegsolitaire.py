from utils.board import Board, BoardType
from utils.hex_board_graph import HexBoardGraph
import networkx as nx
import matplotlib.pyplot as plt


class Peg:
    EMPTY = 1
    PEG = 2
    SELECTED = 3  # Only for visualisation
    JUMPED_OVER = 4  # Only for visualisation


class PegBoard(Board):
    def __init__(self, board_type, size, empty_start_pegs=[(0, 0)], graphing_freq=1, display_game=False):
        super().__init__(board_type, size)
        self.graph = HexBoardGraph(
            self.neighbour_dict, self.board, self.board_type)

        self.peghole_status = {
            coord: Peg.PEG for row in self.board for coord in row}
        for peg in empty_start_pegs:
            self.peghole_status[peg] = Peg.EMPTY

        self.graphing_freq = graphing_freq
        self.display_game = display_game

        if self.display_game:
            self.generate_legal_moves()
            self.init_graph()
            self.display_graph()
            # self.display_board_state()
        # self.legal_moves
        # generate_legal_moves

    def generate_legal_moves(self):
        """
        This function generates legal moves for the current board state and returns them as list of tuples of the form:
            legal_moves = [((peg_coordinates), (jumped_over_coordinates), (landing_pegholde_coordinates)), ...]
        """
        legal_moves = []

        # Iterate through each peghole on the board
        for r, row in enumerate(self.board):
            for c, col in enumerate(row):

                # Check if peghole contains a peg
                if self.peghole_status[(r, c)] == Peg.PEG:

                    # Iterate through neighbours to find possible jump-over pegs
                    for direction, neighbour in self.neighbour_dict[(r, c)].items():
                        """
                        A legal jump:
                            - the neighbour is a peg 
                            - single direction jump (direction of neighbour == direction of neighbour's neighbour)
                            - the jump is within bounds (all neighbour checks are within bounds in this case)
                            - the landing peghole is empty
                        """
                        if self.peghole_status[neighbour] == Peg.PEG and direction in self.neighbour_dict[neighbour]:
                            landing_peghole = self.neighbour_dict[neighbour][direction]
                            if self.peghole_status[landing_peghole] == Peg.EMPTY:
                                legal_moves.append(
                                    ((r, c), neighbour, landing_peghole))

        # For easy display, lets us print possible moves to terminal if a human wants to play
        if self.display_game:
            self.legal_moves = legal_moves
        return legal_moves

    def make_move(self, move):
        # Visualise selected move
        if self.display_game:
            self.peghole_status[move[0]] = Peg.SELECTED
            self.peghole_status[move[1]] = Peg.JUMPED_OVER
            self.update_graph()

        # Actually make the move
        self.peghole_status[move[0]] = Peg.EMPTY
        self.peghole_status[move[1]] = Peg.EMPTY
        self.peghole_status[move[2]] = Peg.PEG

        #self.legal_moves = self.generate_legal_moves()
        if self.display_game:
            # self.display_board_state()
            self.update_graph()

    def get_remaining_pegs(self):
        # Count remaining pegs after game end
        remaining = 0
        for peghole in self.peghole_status:
            if self.peghole_status[peghole] == Peg.PEG:
                remaining += 1

        return remaining

    def get_reinforcement(self):
        # Get reinforcement to return to RL-model
        reinforcement = 0
        legal_moves = self.generate_legal_moves()
        if len(legal_moves) == 0:
            remaining_pegs = self.get_remaining_pegs()
            # Can give extra reinforcement for wins
            if remaining_pegs == 1:
                reinforcement += 1
            else:
                reinforcement -= 1

        return reinforcement

    def get_board_state(self):
        return self.peghole_status

    """Display methods and visualisation"""

    def init_graph(self):
        plt.clf()
        plt.ion()
        self.display_graph()
        plt.show()
        plt.pause(self.graphing_freq)

    def update_graph(self):
        plt.clf()
        self.display_graph()
        plt.pause(self.graphing_freq)

    def display_board_state(self):
        # print legal moves for debugging purposes
        for i, move in enumerate(self.legal_moves):
            print(i, move)

    def display_graph(self):
        nx.draw(
            self.graph.graph,
            pos=self.graph.pos,
            node_color=self.get_node_colours(),
            node_size=self.get_node_sizes()
        )

    def get_node_sizes(self):
        sizes = []
        for node in self.graph.graph:
            # Make holes slightly smaller than pegs, to better simulate real life peg solitaire
            if self.peghole_status[node] == Peg.EMPTY:
                size = 120
            else:
                size = 200
            sizes.append(size)
        return sizes

    def get_node_colours(self):
        colours = []
        for node in self.graph.graph:
            if self.peghole_status[node] == Peg.EMPTY:
                colour = "#0a0a0a"
            elif self.peghole_status[node] == Peg.PEG:
                colour = "#09388f"
            elif self.peghole_status[node] == Peg.SELECTED:
                colour = "#1cb514"
            elif self.peghole_status[node] == Peg.JUMPED_OVER:
                colour = "#c9221c"
            colours.append(colour)
        return colours


if __name__ == "__main__":
    test1 = PegBoard(BoardType.TRIANGLE, 5, display_game=True)

    while True:
        move = input("Select your move: ")
        if move == 'q':
            break
        test1.make_move(test1.legal_moves[int(move)])
        test1.generate_legal_moves()
