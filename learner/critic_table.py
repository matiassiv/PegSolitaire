

class CriticTable:
    def __init__(self, learning_rate=0.02, discount_factor=0.95, trace_decay=0.8):
        
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.trace_decay = trace_decay
        
        self.values = {}
        self.eligibilities = {}
        
    def reset_eligibilities(self):
        pass
    
    def handle_state(self, state):
        pass

    def calculate_temp_diff(self, new_state, curr_state, reinforcement):
        pass
    
    def update_value_and_eligibility(self, SAP_trace, temporal_difference):
        pass