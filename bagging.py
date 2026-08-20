from ensemble import ensemble
import numpy as np


class Bagging(ensemble):

    def bootstrapping(self, X, Y):

        num_samples = X.shape[0]

        bootstrap_index = np.random.choice(
            num_samples,
            size=num_samples,
            replace=True
        )

        X_bootstrap = X[bootstrap_index]
        Y_bootstrap = Y[bootstrap_index]

        return X_bootstrap, Y_bootstrap

    def fit(self, X, Y):

        for model in self.models:

            X_bootstrap, Y_bootstrap = self.bootstrapping(
                X, Y
            )

            model.fit(
                X_bootstrap,
                Y_bootstrap
            )

    def predict(self, X):

        predictions = []

        for model in self.models:

            prediction = model.predict(X)

            predictions.append(prediction)

        predictions = np.array(predictions)

        final_prediction = np.mean(
            predictions,
            axis=0
        )

        return final_prediction