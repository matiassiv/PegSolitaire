try:
    from critic_table import CriticTable
    from critic_ann import CriticANN
except:
    print("error in critic import")


class Critic:
    def __init__(
        self,
        critic_type="table",
        learning_rate=0.02,
        discount_factor=0.95,
        trace_decay=0.8,
        input_nodes=15,
        layer_sizes=[20, 10]
    ):
        if critic_type == "table":
            self.critic = CriticTable(
                learning_rate, discount_factor, trace_decay)
        elif critic_type == "ann":
            self.critic = CriticANN(
                learning_rate=learning_rate,
                discount_factor=discount_factor,
                trace_decay=trace_decay,
                input_nodes=input_nodes,
                layer_sizes=layer_sizes
            )

    def reset_eligibilities(self):
        self.critic.reset_eligibilities()

    def handle_state(self, state):
        self.critic.handle_state(state)

    def calculate_temp_diff(self, new_state, curr_state, reinforcement):
        return self.critic.calculate_temp_diff(new_state, curr_state, reinforcement)

    def update_value_and_eligibility(self, SAP_trace, temporal_difference):
        self.critic.update_value_and_eligibility(
            SAP_trace, temporal_difference)
