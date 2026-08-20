from ensemble import ensemble
import numpy as np
import copy


class stacking(ensemble):

    def __init__(
        self,
        model,
        num_model,
        meta_model,
        *model_args,
        **model_kwargs
    ):

        super().__init__(
            model,
            num_model,
            *model_args,
            **model_kwargs
        )

        self.model = model
        self.model_args = model_args
        self.model_kwargs = model_kwargs
        self.meta_model = meta_model

    def fit(
        self,
        X,
        Y,
        n_folds=5,
        epochs=100,
        learning_rate=0.01
    ):

        if n_folds > len(X):
            n_folds = len(X)

        self.models = [
            self.model(
                *self.model_args,
                **self.model_kwargs
            )
            for _ in range(self.num_models)
        ]

        meta_X = np.zeros(
            (len(X), self.num_models)
        )

        indices = np.arange(len(X))
        np.random.shuffle(indices)

        folds = np.array_split(
            indices,
            n_folds
        )

        for i, model in enumerate(self.models):

            oof_predictions = np.zeros(
                len(X)
            )

            for fold in range(n_folds):

                validation_indices = folds[fold]

                train_indices = np.concatenate(
                    [
                        folds[j]
                        for j in range(n_folds)
                        if j != fold
                    ]
                )

                X_train = X[train_indices]
                Y_train = Y[train_indices]

                X_validation = X[
                    validation_indices
                ]

                fold_model = copy.deepcopy(
                    model
                )

                fold_model.fit(
                    X_train,
                    Y_train,
                    epochs,
                    learning_rate
                )

                predictions = fold_model.predict(
                    X_validation
                )

                oof_predictions[
                    validation_indices
                ] = predictions.ravel()

            meta_X[:, i] = oof_predictions

            model.fit(
                X,
                Y,
                epochs,
                learning_rate
            )

        self.meta_model.fit(
            meta_X,
            Y,
            epochs,
            learning_rate
        )

        return self

    def predict(self, X):

        meta_X = np.zeros(
            (len(X), self.num_models)
        )

        for i, model in enumerate(
            self.models
        ):

            predictions = model.predict(X)

            meta_X[:, i] = predictions.ravel()

        return self.meta_model.predict(
            meta_X
        )