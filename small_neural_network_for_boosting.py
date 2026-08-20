import numpy as np


class verylittle_neuralnetwork:

    def __init__(
        self,
        input_features_numbers,
        hidden_layers_neurons_numbers,
        output_layer_neurons_numbers
    ):

        # Small random weights for the input → hidden layer
        self.w1 = (
            np.random.randn(
                input_features_numbers,
                hidden_layers_neurons_numbers
            ) * 0.01
        )

        # Start biases at zero
        self.b1 = np.zeros(
            hidden_layers_neurons_numbers
        )

        # Small random weights for the hidden → output layer
        self.w2 = (
            np.random.randn(
                hidden_layers_neurons_numbers,
                output_layer_neurons_numbers
            ) * 0.01
        )

        # Start output bias at zero
        self.b2 = np.zeros(
            output_layer_neurons_numbers
        )

    def relu_activation_function(self, x):

        return np.maximum(0, x)

    def forward_pass(self, x):

        # Input → hidden layer
        self.z1 = x @ self.w1 + self.b1

        # ReLU activation
        self.a1 = self.relu_activation_function(
            self.z1
        )

        # Hidden → output layer
        self.z2 = self.a1 @ self.w2 + self.b2

        # Linear output
        # No sigmoid because this is regression
        self.a2 = self.z2

        return self.a2

    def backpropagation(
        self,
        x,
        y,
        learning_rate
    ):

        # --------------------------------
        # Forward-pass loss
        # --------------------------------

        self.cost_function = (
            self.a2 - y
        ) ** 2

        # dC / da2
        self.der_costfunction_rsp_a2 = (
            2 * (self.a2 - y)
        )

        # Since:
        #
        # a2 = z2
        #
        # da2 / dz2 = 1

        self.der_a2_rsp_z2 = 1

        # dC / dz2
        self.der_costfunction_rsp_z2 = (
            self.der_costfunction_rsp_a2
            * self.der_a2_rsp_z2
        )

        # --------------------------------
        # Gradients for w2
        # --------------------------------

        # dz2 / dw2
        self.der_z2_rsp_w2 = self.a1

        # dC / dw2
        self.der_cost_function_rsp_w2 = np.outer(
            self.der_z2_rsp_w2,
            self.der_costfunction_rsp_z2
        )

        # --------------------------------
        # Gradient for b2
        # --------------------------------

        # dz2 / db2 = 1
        self.der_z2_rsp_b2 = 1

        # dC / db2
        self.cost_function_rsp_b2 = (
            self.der_costfunction_rsp_z2
        )

        # --------------------------------
        # Gradient flowing backwards
        # --------------------------------

        # dz2 / da1
        self.der_z2_rsp_a1 = self.w2

        # dC / da1
        self.der_costfunction_rsp_a1 = (
            self.der_costfunction_rsp_z2
            @ self.der_z2_rsp_a1.T
        )

        # --------------------------------
        # ReLU derivative
        # --------------------------------

        # da1 / dz1
        self.der_a1_rsp_z1 = (
            self.z1 > 0
        ).astype(float)

        # dC / dz1
        self.der_costfunction_rsp_z1 = (
            self.der_costfunction_rsp_a1
            * self.der_a1_rsp_z1
        )

        # --------------------------------
        # Gradient for b1
        # --------------------------------

        # dz1 / db1 = 1
        self.der_z1_rsp_b1 = 1

        # dC / db1
        self.cost_function_rsp_b1 = (
            self.der_costfunction_rsp_z1
        )

        # --------------------------------
        # Gradient for w1
        # --------------------------------

        # dz1 / dw1 = x
        self.der_z1_rsp_w1 = x

        # dC / dw1
        self.cost_function_rsp_w1 = np.outer(
            self.der_z1_rsp_w1,
            self.der_costfunction_rsp_z1
        )

        # --------------------------------
        # Gradient descent
        # --------------------------------

        self.w1 -= (
            learning_rate
            * self.cost_function_rsp_w1
        )

        self.b1 -= (
            learning_rate
            * self.cost_function_rsp_b1
        )

        self.w2 -= (
            learning_rate
            * self.der_cost_function_rsp_w2
        )

        self.b2 -= (
            learning_rate
            * self.cost_function_rsp_b2
        )

        return self.cost_function

    def fit(
        self,
        x,
        y,
        epochs,
        learning_rate
    ):

        for epoch in range(epochs):

            total_loss = 0

            for i in range(len(x)):

                # Forward pass for one training example
                self.forward_pass(x[i])

                # Backpropagation and weight update
                loss = self.backpropagation(
                    x[i],
                    y[i],
                    learning_rate
                )

                total_loss += loss

            # Average loss over the entire dataset
            average_loss = (
                total_loss / len(x)
            )

            print(
                f"Epoch: {epoch}, Loss: {average_loss}"
            )

    def predict(self, x):

        predictions = []

        for i in range(len(x)):

            prediction = self.forward_pass(
                x[i]
            )

            predictions.append(prediction)

        return np.array(predictions)