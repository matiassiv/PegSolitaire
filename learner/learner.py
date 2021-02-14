# Hacky python imports, cuz Python imports are hard
import os
import sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

try:
    from pegsolitaire import PegBoard, Peg
    from utils.board import BoardType
    from critic import Critic
    from actor import Actor
except:
    print("Couldn't handle imports")


class Learner:
    def __init__(
            self,
            num_episodes=1,
            game_settings={"board_type": BoardType.TRIANGLE, "size": 5,
                           "empty_start_pegs": [(0, 0)], "graphing_freq": 1}
    ):
        self.num_episodes = num_episodes
        self.game_settings = game_settings
        self.critic = Critic()
        self.actor = Actor(num_episodes=num_episodes)

    def train(self):

        # Get remaining pegs after each run, to plot model performance
        remaining = []

        # Iterate over predefined number of episodes
        for episode in self.num_episodes:

            # Initialise game, and reset eligibilities of actor/critic to 0
            curr_game, curr_state, legal_moves = self.init_game()
            self.actor.reset_eligibilities()

            # Add SAP to actor policy and state to critic
            self.actor.handle_state(curr_state, legal_moves)
            self.critic.handle_state(curr_state)

            # Record SAPs performed by the model, which is used to update eligibility trace
            SAP_trace = []

            # Run game until no more legal moves
            while len(legal_moves) > 0:
                move = self.actor.get_move(curr_state, legal_moves)
                new_state, reinforcement, legal_moves = self.perform_move(
                    curr_state, curr_game, move)

                SAP_trace.append((curr_state, move))

                # Critic must calculate temporal difference
                temporal_difference = self.critic.calculate_temp_diff(
                    new_state, curr_state, reinforcement)

                # Update eligibility trace, then update critic value and actor policy
                self.critic.update_value_and_eligibility(SAP_trace, temporal_difference)
                self.actor.update_policy_and_eligibility(SAP_trace, temporal_difference)

                # Shift curr_state to the new_state and add necessary data structures to
                # actor and critic if it is an unseen board state.
                curr_state = new_state
                self.actor.handle_state(curr_state, legal_moves)
                self.critic.handle_state(curr_state)

            remaining.append(curr_game.get_remaining_pegs())
            self.actor.update_greediness()

    def perform_move(self, current_state, current_game, selected_move):
        """
        Performs move on the board, and returns the new state, reinforcement and new legal moves
        """

        # Make selected move
        current_game.make_move()

        # Generate new state and transform to the internal representation of the learner
        new_state = self.generate_internal_board_rep(
            current_game.get_board_state)

        # Generate new legal moves and get the reward of the state
        new_legal_moves = current_game.generate_legal_moves()
        reinforcement = current_game.get_reinforcement()

        return new_state, reinforcement, new_legal_moves

    def generate_internal_board_rep(self, board_state):
        """
        Current implementation bases itself on using board state as key in 
        the policy and value dictionaries for critic and actor. Thus, we need the
        board state to be immutable. We can accomplish this by converting the board
        state to a bitstring, where "1"s represent pegs and "0"s represent holes. The bitstring
        is constructed in top-down fashion, so coordinate (0,0) is the first bit, (1,0) is the 
        second, (1,1) is the third and so on... The parameter board_state is a dictionary of the
        current status for each peghole. Additionally, a bitstring is easy to use as input for an NN.
        """
        board_rep = ""
        for peghole in sorted(board_state):
            if board_state[peghole] == Peg.PEG:
                board_rep += "1"
            else:
                board_rep += "0"
        return board_rep

    def init_game(self):
        """
        Initialises the game for each episode for the learner.
        Game settings are read from a dictionary.
        An immutable board representation is generated, so actor/critic
        can keep track of states.
        Finally the initial legal moves are generated.
        """

        game = PegBoard(
            board_type=self.game_settings["board_type"],
            size=self.game_settings["size"],
            empty_start_pegs=self.game_settings["empty_start_pegs"],
            graphing_freq=self.game_settings["graphing_freq"]
        )

        board_rep = self.generate_internal_board_rep(game.get_board_state())
        legal_moves = game.generate_legal_moves()

        return game, board_rep, legal_moves


if __name__ == "__main__":
    print("hi")
