

class CriticTable:
    def __init__(self, learning_rate=0.02, discount_factor=0.95, trace_decay=0.8):
        
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.trace_decay = trace_decay
        
        self.values = {}
        self.eligibilities = {}