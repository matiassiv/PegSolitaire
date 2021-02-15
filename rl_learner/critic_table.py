import random

random.seed(1)


class CriticTable:
    def __init__(self, learning_rate=0.05, discount_factor=0.95, trace_decay=0.8):

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.trace_decay = trace_decay

        self.values = {}
        self.eligibilities = {}

    def reset_eligibilities(self):
        for state in self.eligibilities:
            self.eligibilities[state] = 0

    def handle_state(self, state):
        # Initialise state with small random value if not encountered before
        if state not in self.values:
            self.values[state] = random.uniform(0.0, 0.1)

    def calculate_temp_diff(self, new_state, curr_state, reinforcement):
        self.handle_state(new_state)
        return reinforcement + (self.discount_factor * self.values[new_state] - self.values[curr_state])

    def update_value(self, state, temporal_difference):
        self.values[state] += self.learning_rate * \
            temporal_difference * self.eligibilities[state]

    def update_eligibility(self, state, decay_version=False):
        if not decay_version:
            self.eligibilities[state] = 1
        else:
            self.eligibilities[state] = self.discount_factor * \
                self.trace_decay * self.eligibilities[state]

    def update_value_and_eligibility(self, SAP_trace, temporal_difference):
        last_state, last_action = SAP_trace[-1]
        self.update_eligibility(last_state)

        for (state, action) in SAP_trace:
            self.update_value(state, temporal_difference)
            self.update_eligibility(state, True)
