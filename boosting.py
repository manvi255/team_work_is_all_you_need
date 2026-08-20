import numpy as np
import copy


class Boosting:

    def __init__(
        self,
        model,
        num_model,
        learning_rate,
        epochs
    ):

        self.model = model
        self.num_models = num_model
        self.boosting_learning_rate = learning_rate
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.models = []
        self.initial_prediction = None

    def fit(self, X, Y):

        self.initial_prediction = np.mean(Y)

        predictions = np.full(
            len(Y),
            self.initial_prediction
        )

        for i in range(self.num_models):

            residual = Y - predictions

            new_model = copy.deepcopy(
                self.model
            )

            new_model.fit(
                X,
                residual,
                self.epochs,
                self.learning_rate
            )

            correction = new_model.predict(X)
            correction = correction.ravel()

            predictions += (
                self.boosting_learning_rate
                * correction
            )

            self.models.append(
                new_model
            )

        return self

    def predict(self, X):

        predictions = np.full(
            len(X),
            self.initial_prediction
        )

        for model in self.models:

            correction = model.predict(X)
            correction = correction.ravel()

            predictions += (
                self.boosting_learning_rate
                * correction
            )

        return predictions