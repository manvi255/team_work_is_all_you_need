import numpy as np 
class ensemble:

    def __init__(self, model, num_model, *model_args, **model_kwargs):

        self.num_models = num_model
        self.weights_model = np.ones(num_model) / num_model
        self.models = []

        for _ in range(num_model):
            all_models = model(*model_args, **model_kwargs)
            self.models.append(all_models)
            