import numpy as np


class verylittle_neuralnetwork:

    def __init__(
        self, 
        input_features_numbers,
        hidden_layers_neurons_numbers,
        output_layer_neurons_numbers
    ):

        self.w1 = np.random.rand(
            input_features_numbers,
            hidden_layers_neurons_numbers
        )

        self.b1 = np.random.rand(
            hidden_layers_neurons_numbers
        )

        self.w2 = np.random.rand(
            hidden_layers_neurons_numbers,
            output_layer_neurons_numbers
        )

        self.b2 = np.random.rand(
            output_layer_neurons_numbers
        )

    def relu_activation_function(self, x):
        return np.maximum(0, x)

    def forward_pass(self, x):

        self.z1 = x @ self.w1 + self.b1

        self.a1 = self.relu_activation_function(
            self.z1
        )

        self.z2 = self.a1 @ self.w2 + self.b2

        self.a2 = self.z2

        return self.a2

    def backpropogation(self, x, y, learning_rate):

        self.cost_function = (
            self.a2 - y
        ) ** 2

        self.der_costfunction_rsp_a2 = (
            2 * (self.a2 - y)
        )

        self.der_a2_rsp_z2 = 1

        self.der_costfunction_rsp_z2 = (
            self.der_costfunction_rsp_a2
            * self.der_a2_rsp_z2
        )

        self.der_z2_rsp_w2 = self.a1

        self.der_cost_function_rsp_w2 = np.outer(
            self.der_z2_rsp_w2,
            self.der_costfunction_rsp_z2
        )

        self.der_z2_rsp_b2 = 1

        self.cost_function_rsp_b2 = (
            self.der_costfunction_rsp_z2
        )

        self.der_z2_rsp_a1 = self.w2

        self.der_costfunction_rsp_a1 = (
            self.der_costfunction_rsp_z2
            @ self.der_z2_rsp_a1.T
        )

        self.der_a1_rsp_z1 = (
            self.z1 > 0
        ).astype(float)

        self.der_costfunction_rsp_z1 = (
            self.der_costfunction_rsp_a1
            * self.der_a1_rsp_z1
        )

        self.der_z1_rsp_b1 = 1

        self.cost_function_rsp_b1 = (
            self.der_costfunction_rsp_z1
        )

        self.der_z1_rsp_w1 = x

        self.cost_function_rsp_w1 = np.outer(
            self.der_z1_rsp_w1,
            self.der_costfunction_rsp_z1
        )

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

    def fit(self, x, y, epochs, learning_rate):

        for epoch in range(epochs):

            total_loss = 0

            for i in range(len(x)):

                self.forward_pass(x[i])

                loss = self.backpropogation(
                    x[i],
                    y[i],
                    learning_rate
                )

                total_loss += loss

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