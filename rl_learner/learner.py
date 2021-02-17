# Hacky python imports, cuz Python imports are hard
import os
import sys
import random
import matplotlib.pyplot as plt
import numpy as np
import time

random.seed(1)
np.random.seed(1)

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)


try:
    from pegsolitaire import PegBoard, Peg
    from utils.board import BoardType
except:
    print("Cannot import from out of folder")
try:
    from critic import Critic
except:
    print("bleh")

try:
    from actor import Actor
except:
    print("Couldn't handle imports from same folder")

"""
print(sys.path)

print(dir())
"""


class Learner:
    def __init__(
            self,
            num_episodes=2000,
            game_settings={"board_type": BoardType.TRIANGLE, "size": 6,
                           "empty_start_pegs": [(0, 0)], "graphing_freq": 0.4, "display_game": False},
            critic_settings={"critic_type": "table", "learning_rate": 0.03,
                             "discount_factor": 0.95, "trace_decay": 0.8, "layer_sizes": [20, 10]},
            actor_settings={"learning_rate": 0.03, "e_greedy": 0.5,
                            "trace_decay": 0.8, "discount_factor": 0.95
                            }
    ):
        self.num_episodes = num_episodes
        self.game_settings = game_settings

        # Number of input nodes to network must be number of pegholes
        if game_settings["board_type"] == BoardType.TRIANGLE:
            size = game_settings["size"]
            input_nodes = size * (size + 1) // 2
        else:
            size = game_settings["size"]
            input_nodes = size * size

        self.critic = Critic(
            critic_type=critic_settings["critic_type"],
            learning_rate=critic_settings["learning_rate"],
            discount_factor=critic_settings["discount_factor"],
            trace_decay=critic_settings["trace_decay"],
            input_nodes=input_nodes,
            layer_sizes=critic_settings["layer_sizes"]
        )
        self.actor = Actor(
            learning_rate=actor_settings["learning_rate"],
            discount_factor=actor_settings["discount_factor"],
            e_greedy=actor_settings["e_greedy"],
            trace_decay=actor_settings["trace_decay"],
            num_episodes=num_episodes
        )

    def train(self):

        # Get remaining pegs after each run, to plot model performance
        remaining = []

        # Iterate over predefined number of episodes
        for episode in range(self.num_episodes):
            # start_time = time.time()
            # Initialise game, and reset eligibilities of actor/critic to 0
            curr_game, curr_state, legal_moves = self.init_game()
            self.actor.reset_eligibilities()
            self.critic.reset_eligibilities()

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
                # print("Temp_diff:", temporal_difference)
                # Update eligibility trace, then update critic value and actor policy
                self.critic.update_value_and_eligibility(
                    SAP_trace, temporal_difference)
                self.actor.update_policy_and_eligibility(
                    SAP_trace, temporal_difference)

                # Shift curr_state to the new_state and add necessary data structures to
                # actor and critic if it is an unseen board state.
                curr_state = new_state
                self.actor.handle_state(curr_state, legal_moves)
                self.critic.handle_state(curr_state)

            remaining.append(curr_game.get_remaining_pegs())
            self.actor.update_greediness()
            # end_time = time.time()
            # print("TIME episode " + str(episode)+":", end_time - start_time)
        return remaining

    def perform_move(self, current_state, current_game, selected_move):
        """
        Performs move on the board, and returns the new state, reinforcement and new legal moves
        """

        # Make selected move
        current_game.make_move(selected_move)

        # Generate new state and transform to the internal representation of the learner
        new_state = self.generate_internal_board_rep(
            current_game.get_board_state())

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

    def init_game(self, display_game=False):
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
            graphing_freq=self.game_settings["graphing_freq"],
            display_game=display_game
        )

        board_rep = self.generate_internal_board_rep(game.get_board_state())
        legal_moves = game.generate_legal_moves()

        return game, board_rep, legal_moves

    def test(self):
        curr_game, curr_state, legal_moves = self.init_game(display_game=True)
        self.actor.e_greedy = 0
        self.actor.handle_state(curr_state, legal_moves)

        # Record SAPs performed by the model, which is used to update eligibility trace
        SAP_trace = []

        # Run game until no more legal moves
        while len(legal_moves) > 0:
            move = self.actor.get_move(curr_state, legal_moves)
            new_state, reinforcement, legal_moves = self.perform_move(
                curr_state, curr_game, move)

            SAP_trace.append((curr_state, move))

            curr_state = new_state

        return curr_game.get_remaining_pegs(), SAP_trace


if __name__ == "__main__":
    # Settings for Triangle board size 5, critic_table
    settings_triangle_table = {
        "num_episodes": 600,
        "game_settings": {
            "board_type": BoardType.TRIANGLE,
            "size": 5,
            "empty_start_pegs": [(2, 1)],
            "graphing_freq": 0.4,
            "display_game": False
        },
        "critic_settings": {
            "critic_type": "table",
            "learning_rate": 0.01,
            "discount_factor": 0.9,
            "trace_decay": 0.8,
            "layer_sizes": [10]
        },
        "actor_settings": {
            "learning_rate": 0.05,
            "e_greedy": 0.5,
            "trace_decay": 0.8,
            "discount_factor": 0.9
        }
    }

    # Settings for Triangle board size 5, critic_ANN
    # For critic_ANN we need to set learning rate much lower than for actor
    settings_triangle_ann = {
        "num_episodes": 600,
        "game_settings": {
            "board_type": BoardType.TRIANGLE,
            "size": 5,
            "empty_start_pegs": [(2, 1)],
            "graphing_freq": 0.4,
            "display_game": False
        },
        "critic_settings": {
            "critic_type": "ann",
            "learning_rate": 0.0008,
            "discount_factor": 0.9,
            "trace_decay": 0.8,
            "layer_sizes": [15, 8]
        },
        "actor_settings": {
            "learning_rate": 0.03,
            "e_greedy": 0.5,
            "trace_decay": 0.8,
            "discount_factor": 0.9
        }
    }

    # Settings for Diamond board size 4, critic_table
    settings_diamond_table = {
        "num_episodes": 600,
        "game_settings": {
            "board_type": BoardType.DIAMOND,
            "size": 4,
            "empty_start_pegs": [(2, 1)],
            "graphing_freq": 0.4,
            "display_game": False
        },
        "critic_settings": {
            "critic_type": "table",
            "learning_rate": 0.01,
            "discount_factor": 0.9,
            "trace_decay": 0.8,
            "layer_sizes": [10]
        },
        "actor_settings": {
            "learning_rate": 0.05,
            "e_greedy": 0.5,
            "trace_decay": 0.8,
            "discount_factor": 0.9
        }
    }

    # Settings for Diamond board size 4, critic_ann
    settings_diamond_ann = {
        "num_episodes": 600,
        "game_settings": {
            "board_type": BoardType.DIAMOND,
            "size": 4,
            "empty_start_pegs": [(2, 1)],
            "graphing_freq": 0.4,
            "display_game": False
        },
        "critic_settings": {
            "critic_type": "ann",
            "learning_rate": 0.001,
            "discount_factor": 0.9,
            "trace_decay": 0.8,
            "layer_sizes": [15, 8]
        },
        "actor_settings": {
            "learning_rate": 0.05,
            "e_greedy": 0.5,
            "trace_decay": 0.8,
            "discount_factor": 0.9
        }
    }

    # Settings reviewer can adjust
    settings_reasonable_behaviour = {
        "num_episodes": 500,
        "game_settings": {
            "board_type": BoardType.TRIANGLE,
            "size": 5,
            "empty_start_pegs": [(2, 1)],
            "graphing_freq": 0.4,
            "display_game": False
        },
        "critic_settings": {
            "critic_type": "table",
            "learning_rate": 0.01,
            "discount_factor": 0.88,
            "trace_decay": 0.8,
            "layer_sizes": [15, 20, 30, 5, 1]  # Sizes of hidden layers
        },
        "actor_settings": {
            "learning_rate": 0.05,
            "e_greedy": 0.5,  # Decays with a flat rate proportional to num_episodes
            "trace_decay": 0.8,
            "discount_factor": 0.88}
    }

    # Select which settings to use
    settings_tuple = (settings_triangle_table, settings_triangle_ann,
                      settings_diamond_table, settings_diamond_ann)

    # Uncomment below to run all 4 cases
    """
    for i, settings in enumerate(settings_tuple):
        start_time = time.time()
        l = Learner(settings["num_episodes"], settings["game_settings"],
                    settings["critic_settings"], settings["actor_settings"])
        performance = l.train()
        x_vals = np.arange(len(performance))

        plt.plot(x_vals, performance)
        plt.ylabel("Remaining pegs")
        plt.xlabel("Episodes")
        plt.title(settings["critic_settings"]["critic_type"])
        plt.savefig(str(i)+".png")
        plt.clf()

        end_time = time.time()
        print("Time to train", settings["num_episodes"],
              "episodes:", end_time - start_time)
        
        remaining, SAP_trace = l.test()
        plt.clf()
        print("Remaining pegs:", remaining)
        for SAP in SAP_trace:
            print(SAP)
        """
