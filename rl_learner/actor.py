import numpy as np
import random

random.seed(1)


class Actor:
    def __init__(self, learning_rate=0.05, e_greedy=0.5, trace_decay=0.8, discount_factor=0.95,  num_episodes=50):

        self.learning_rate = learning_rate
        self.e_greedy = e_greedy
        self.trace_decay = trace_decay
        self.discount_factor = discount_factor
        # We use num_episodes to facilitate decay in greediness
        # In the final episode we reach greediness = 0, thus always selecting the greedy choice
        self.e_greedy_decay = e_greedy/num_episodes

        self.policy = {}
        self.eligibilities = {}

    def get_move(self, state, legal_moves):

        # If we do a run outside of training and the model encounters
        # a state it has not yet seen
        if state not in self.policy:
            print("Random")
            return random.choice(legal_moves)

        if np.random.rand() >= self.e_greedy:
            # If random number is bigger than e_greedy, we return the best policy move
            return max(self.policy[state], key=(lambda action: self.policy[state][action]))

        # Else return random move
        return random.choice(list(self.policy[state]))

    def reset_eligibilities(self):

        for state in self.eligibilities:
            for action in self.eligibilities[state]:
                self.eligibilities[state][action] = 0

    # Add a state-action pair to policy
    def add_SAP(self, state, action):

        if state not in self.policy:
            self.policy[state] = {}

        self.policy[state][action] = 0

    # Add possible actions to policy for a given state
    def handle_state(self, state, legal_moves):
        if state not in self.policy:
            for action in legal_moves:
                self.add_SAP(state, action)

    def update_greediness(self):

        self.e_greedy -= self.e_greedy_decay

    def update_policy(self, state, action, temporal_difference):
        self.policy[state][action] += \
            self.learning_rate * temporal_difference * \
            self.eligibilities[state][action]

    def update_eligibility(self, state, action, decay_version=False):
        if state not in self.eligibilities:
            self.eligibilities[state] = {}

        # If we're updating the last SAP, we set eligibility to 1
        # else we decay already set eligibilities
        if not decay_version:
            self.eligibilities[state][action] = 1
        else:
            self.eligibilities[state][action] = self.eligibilities[state][action] * \
                self.discount_factor * self.trace_decay

    def update_policy_and_eligibility(self, SAP_trace, temporal_difference):
        """
        Takes in the trace of states of actions of an active game and
        the temporal difference computed by the critic from the current state
        and the previous one.

        Updates the eligibility of the current SAP
        Then, all states in policy and eligibilities for this episode are updated
        """
        last_state, last_action = SAP_trace[-1]
        self.update_eligibility(last_state, last_action)

        for (state, action) in SAP_trace:
            self.update_policy(state, action, temporal_difference)
            self.update_eligibility(state, action)
