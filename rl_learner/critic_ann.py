from tensorflow import keras
import tensorflow as tf
import time
import numpy as np
import os
# As we do one action at a time, we don't get any benefit from using the GPU
# Setting the env var to "-1" makes tf not recognize the GPU on my computer
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
np.random.seed(1)


class CriticANN:
    def __init__(
        self,
        learning_rate=0.02,
        discount_factor=0.95,
        trace_decay=0.8,
        input_nodes=15,
        layer_sizes=[20, 10],
    ):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.trace_decay = trace_decay
        self.input_nodes = input_nodes
        self.layer_sizes = layer_sizes
        self.eligibilities = {}

        self.create_ANN()

    def create_ANN(self):
        """
        Here we create the actual NN model to calculate V(s)
        Input layer is so that number of nodes matches
        number of pegholes on the board. The input layer
        takes in some state s. Next up are n numbers of
        fully connected hidden layers. The last layer
        is the output layer, which is just a single node
        and representes V(s) for the passed-in state s.
        """
        # Input layer has num_nodes == num_pegholes
        inputs = keras.Input(self.input_nodes)
        x = inputs
        for i, layer in enumerate(self.layer_sizes):
            x = keras.layers.Dense(layer, activation="relu")(x)

        # Output has only one node, as output is just the value of a state, which is a scalar
        output = keras.layers.Dense(1, activation="linear")(x)
        self.model = keras.Model(inputs, output)
        self.model.summary()

        opt = keras.optimizers.SGD(learning_rate=self.learning_rate)
        self.model.compile(optimizer=opt, loss="mse")

    def reset_eligibilities(self):
        # Eligibilities have same shape as gradient/weights
        # Every layer except input has one set of incoming weights
        # and one set of outgoing weights.
        for i in range(len(self.layer_sizes)+1):
            self.eligibilities[2*i] = tf.zeros_like(
                self.model.layers[i+1].get_weights()[0])
            self.eligibilities[2*i + 1] = tf.zeros_like(
                self.model.layers[i+1].get_weights()[1])

    def handle_state(self, state):
        # Not needed
        pass

    def convert_state_to_tensor(self, state):
        return tf.constant([[int(i) for i in state]],
                           dtype=tf.float64)

    def calculate_temp_diff(self, new_state, curr_state, reinforcement):
        """
        diff = r + (discount * nn.forward(s') - nn.forward(s))

        We use temp_diff as input to our loss function for ANN-critic
        """

        # Tensor operations seem to much faster, so using tensor to calc temp_diff
        # then converting back to a regular scalar
        self.new_state = self.convert_state_to_tensor(new_state)
        self.curr_state = self.convert_state_to_tensor(curr_state)
        self.reinforcement = reinforcement
        temp_diff = reinforcement + self.discount_factor * self.model(self.new_state) \
            - self.model(self.curr_state)

        return temp_diff.numpy()[0, 0]

    def update_eligibility(self, gradients, decay_version=False):
        if not decay_version:
            for i, grad in enumerate(gradients):
                self.eligibilities[i] += grad
        else:
            for i in self.eligibilities:
                self.eligibilities[i] = self.discount_factor * \
                    self.trace_decay * self.eligibilities[i]

    # @tf.function
    def update_value_and_eligibility(self, SAP_trace, temporal_difference):
        """
        In the context of an ANN, the value of a state can be thought of as the 
        output from the ANN. Generally, we want to minimise loss to improve ANN performance.
        The value of a state, V(s) = r + (discount_factor * V(s')), where r is the reinforcement
        from going from state s to state s'. Thus the target for our ANN is therefore to make
        ANN(s) ~ r + (discount_factor * ANN(s')), or conversely, using MSE as a loss function:
        we want to minimise ((r + (discount_factor * ANN(s'))) -  ANN(s)) ** 2, or, put simply,
        (temp_diff) ** 2. It can be shown that the derivative of the loss function wrt w_i is simply 
        -2 * temp_diff * dV(s)/dw_i, giving the weight update rule: 
        w_i += learning_rate * (-2) * temp_diff * dV(s)/dw_i.

        However, we are also using eligibility traces in this context, which means that we only want
        to update the gradients according to their eligibility. This is accomplished by essentially
        scaling the gradients according to their eligibility. Eligibililty of layer i, e_i, is updated
        with the gradients its layer: e_i += dV(s)/dw_i. Weight updates are turned into a new update rule
        w_i += learning_rate * (-2) * temp_diff * e_i. After the weight_update, we decay the eligibility trace
        as per usual. The consequence of this small change to the weight update rule, means that recent
        changes to the weights are given more emphasis.
        """

        # To allow us to adjust the gradients before applying them in the backpropagation,
        # we can make use of gradient tape. The tape watches variables involved in tensor operations
        # and has a method called gradient which computes the gradient for all watched variables
        # wrt to some function. In our case we want to compute the gradient of the loss function wrt
        # the weights of the model. Ideally we would be able to just use the temporal_difference which
        # is already calculated, but tape requires that the variables to differentiate needs to be referenced
        # in the indented block - therefore we need to recalculate V(s) and V(s')
        with tf.GradientTape() as tape:
            # We start by computing the target, which is the discounted value of the next state
            # with the added reinforcement
            y = self.reinforcement + self.discount_factor * \
                self.model(self.new_state)

            # Then we compute the predicted value of the current state
            y_pred = self.model(self.curr_state)
            # Now the tape watches the model's weights, as they were involved in computing the predictions
            # The loss is computed, i.e. the MSE of y_pred and y.
            loss = self.model.compiled_loss(y, y_pred)

        # We get all the trainable_variables, i.e. the weights of all layers of the model,
        # and we calculate the gradient of the weights
        trainable_vars = self.model.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # The gradients correspond to the partial derivative of the loss function
        # wrt the weights, that is dJ/dw_i. However, for the eligibility update we
        # are interested in only dV(s)/dw_i. From the actor-critic pdf we know that
        # dJ/dw_i = -2 * temp_diff * dV(s)/dw_i, so we can simply divide by (-2) * temp_diff
        # to find the gradients to update the eligibility with.
        gradients /= (-2) * temporal_difference
        self.update_eligibility(gradients)

        weight_grads = []
        for i in self.eligibilities:
            # Calculate the weight_gradients for the backpropagation
            weight_grads.append(
                self.eligibilities[i] * (-2) * temporal_difference)

            # Decay eligibilities
            self.update_eligibility(gradients, decay_version=True)

        # Apply the weight gradients to finish backpropagation
        self.model.optimizer.apply_gradients(zip(weight_grads, trainable_vars))


if __name__ == "__main__":
    ann = CriticANN()
    print(len(ann.model.layers[3].get_weights()))
