import numpy as np
import copy


class Boosting:

    def __init__(
        self,
        model,
        num_model,
        nn_learning_rate,
        boosting_learning_rate,
        epochs
    ):

        # Base neural network
        self.model = model

        # Number of boosting models
        self.num_models = num_model

        # Learning rate used INSIDE each neural network
        self.nn_learning_rate = nn_learning_rate

        # Learning rate used when adding
        # each model's correction to the ensemble
        self.boosting_learning_rate = (
            boosting_learning_rate
        )

        # Number of epochs for every neural network
        self.epochs = epochs

        # Store all trained models
        self.models = []

        # Initial prediction
        self.initial_prediction = None

        # Store residual MSE after every
        # boosting round
        self.residual_history = []

    def fit(self, X, Y):

        # Clear previous models and history
        # in case fit() is called again
        self.models = []
        self.residual_history = []

        # --------------------------------
        # Initial prediction
        # --------------------------------

        # Start by predicting the mean
        # of the target values
        self.initial_prediction = np.mean(Y)

        # Keep predictions in exactly
        # the same shape as Y
        predictions = np.full(
            Y.shape,
            self.initial_prediction
        )

        # --------------------------------
        # Boosting loop
        # --------------------------------

        for i in range(self.num_models):

            # Calculate the remaining error
            #
            # residual = actual - current prediction

            residual = Y - predictions

            # Calculate residual MSE
            residual_mse = np.mean(
                residual ** 2
            )

            # Save residual error
            self.residual_history.append(
                residual_mse
            )

            print(
                f"Boosting Round {i + 1}, "
                f"Residual MSE: {residual_mse}"
            )

            # --------------------------------
            # Create a new neural network
            # --------------------------------

            new_model = copy.deepcopy(
                self.model
            )

            # --------------------------------
            # Train neural network
            # --------------------------------

            # IMPORTANT:
            # Use the neural-network learning rate
            # here, NOT the boosting learning rate.

            new_model.fit(
                X,
                residual,
                self.epochs,
                self.nn_learning_rate
            )

            # --------------------------------
            # Predict the correction
            # --------------------------------

            correction = new_model.predict(X)

            # Convert correction to
            # (number_of_samples, 1)

            correction = (
                correction
                .ravel()
                .reshape(-1, 1)
            )

            # --------------------------------
            # Update ensemble prediction
            # --------------------------------

            # IMPORTANT:
            # Use the BOOSTING learning rate here.

            predictions += (
                self.boosting_learning_rate
                * correction
            )

            # --------------------------------
            # Store trained model
            # --------------------------------

            self.models.append(
                new_model
            )

        return self

    def predict(self, X):

        # Start with the initial prediction

        predictions = np.full(
            (len(X), 1),
            self.initial_prediction
        )

        # Add the correction from
        # every boosting model

        for model in self.models:

            correction = model.predict(X)

            # Make shape consistent
            correction = (
                correction
                .ravel()
                .reshape(-1, 1)
            )

            # Add weighted correction

            predictions += (
                self.boosting_learning_rate
                * correction
            )

        return predictions