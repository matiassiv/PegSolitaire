

class Critic:
    def __init__(self, critic="table", learning_rate=0.02, discount_factor=0.95, trace_decay=0.8):
        if critic =

    def reset_eligibilities(self):
        pass

    def handle_state(self, state):
        pass

    def calculate_temp_diff(self, new_state, curr_state, reinforcement):
        pass
    
    def update_value_and_eligibility(self, SAP_trace, temporal_difference)