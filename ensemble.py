import tensorflow as tf


class Ensemble(tf.keras.Model):

    def __init__(self, model_class, config, num_models, num_classes):
        super(Ensemble, self).__init__()
        self.models = []

        for _ in range(num_models):
            model = model_class(config, num_classes=num_classes)
            self.models.append(model)

    def call(self, features):
        output = None
        for model in self.models:
            if output is None:
                output = model(features)
            else:
                output += model(features)
        return output/len(self.models)
