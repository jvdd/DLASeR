import numpy as np

from src.dl_framework import DLFramework
from src.utils import create_simple_ANN_model, operator_to_math, model_name_to_loss, model_name_to_metrics


class DLASeR(DLFramework):

    def __init__(self, qualities, input_size):
        """ Creates the DLASeR framework.

        :param qualities: the qualities for which goals are defined.
        :param input_size: the length of the feature vector.
        """
        super().__init__(qualities)
        self.models = {}
        self.scalers = {}
        self.input_size = input_size

    def add_threshold_goal(self, quality, operator, threshold_value, scaler, layers):
        """ Add a threshold goal to the DLASeR framework.

        :param quality: the quality for which the threshold goal is defined.
        :param operator: the operator of the threshold goal (below = Min, above = Max).
        :param threshold_value: the threshold value of the threshold goal.
        :param scaler: the scaler accompanying the deep learning model.
        :param layers: array specifying the width of the layers of the deep learning model.
        """
        super().add_threshold_goal(quality, operator, threshold_value)
        name = self._get_threshold_name(quality, threshold_value)
        self.scalers[name] = scaler
        self.models[name] = create_simple_ANN_model(self.input_size, layers, output_activation='sigmoid')

    def add_setpoint_goal(self, quality, setpoint_value, epsilon):
        raise NotImplementedError  # DLASeR does not support threshold goals

    def add_optimization_goal(self, quality, operator, scaler, layers):
        """ Add an optimization goal to the DLASeR framework.

        :param quality: the quality for which the optimization goal is defined.
        :param operator: the operator of the optimization goal (minimization = Min, maximization = Max).
        :param scaler: the scaler accompanying the deep learning model.
        :param layers: array specifying the width of the layers of the deep learning model.
        """
        super().add_optimization_goal(quality, operator)
        name = self._get_optimization_name(quality)
        self.scalers[name] = scaler
        self.models[name] = create_simple_ANN_model(self.input_size, layers, output_activation='linear')

    def _y_dict_to_outputs(self, y_dict):
        """ Generates a mapping of given dictionary (quality, values) to another dictionary (model_name, values).
        This mapping also processes the values of the accompanying quality for the goal(s) on that quality.

        :param y_dict: dictionary with as key the quality and as value the quality its values.
        :return: dictionary with as key the model name and as value the processed values of the
                    accompanying quality according to the goal of the model.
        """
        for quality in y_dict.keys():
            assert quality in self.qualities  # Make sure the quality is known
        outputs = {}
        # Threshold goals
        for name, threshold_goal in self.threshold_goals.items():
            quality = threshold_goal.quality
            print(name, quality)
            quality_vals = y_dict[quality]
            operator_math = operator_to_math(threshold_goal.operator)
            y = np.array([operator_math(val, threshold_goal.value) for val in quality_vals]).astype(np.int)
            outputs[name] = y
        # Set-point goals
        # => Not supported by DLASeR
        # Optimization goals
        for name, optimization_goal in self.optimization_goals.items():
            quality = optimization_goal.quality
            y = y_dict[quality]
            outputs[name] = y
        return outputs

    # The online train method
    def train(self, x, y_dict, batch_size, epochs, verbose=True):
        """ The train method that is used in the online (training) cycles.

        :param x: np.array of train feature vectors.
        :param y_dict: dictionary with as key the quality and as value the quality its values for the training data.
        :param batch_size: the batch size.
        :param epochs: the number of epochs.
        :param verbose: boolean indicating whether training info should be printed.
        :return: dictionary with as key the model name and as value the History objects that recorded the training process.
        """
        outputs = self._y_dict_to_outputs(y_dict)
        histories = {}
        for model_name in self.get_model_names():
            # self.scalers[model_name].partial_fit(x)
            histories[model_name] = self.models[model_name].fit(self.scalers[model_name].transform(x), outputs[model_name],
                                                                batch_size=batch_size, epochs=epochs,
                                                                callbacks=self.callback,
                                                                validation_split=0.1,
                                                                verbose=verbose)
        return histories

    # The fit method
    def fit(self, x_train, x_val, y_dict_train, y_dict_val, batch_size, epochs, verbose=True):
        """ A regular fit method that utilizes validation data.

        :param x_train: np.array of training feature vectors.
        :param x_val: np.array of validation feature vectors.
        :param y_dict_train: dictionary with as key the quality and as value the quality its values for the training data.
        :param y_dict_val: dictionary with as key the quality and as value the quality its values for the validation data.
        :param batch_size: the batch size.
        :param epochs: the number of epochs.
        :param verbose: boolean indicating whether training info should be printed.
        :return: dictionary with as key the model name and as value the History objects that recorded the training process.
        """
        outputs_train = self._y_dict_to_outputs(y_dict_train)
        outputs_val = self._y_dict_to_outputs(y_dict_val)
        histories = {}
        for model_name in self.get_model_names():
            self.scalers[model_name].fit(x_train)
            histories[model_name] = self.models[model_name].fit(self.scalers[model_name].transform(x_train), outputs_train[model_name],
                                                                batch_size=batch_size, epochs=epochs,
                                                                validation_data=[self.scalers[model_name].transform(x_val), outputs_val[model_name]],
                                                                callbacks=self.callback,
                                                                verbose=verbose)
        return histories

    # The offline fit method (used for the grid search)
    # The callback is a specific talos callback
    def fit_gridsearch(self, x_train, x_val, y_dict_train, y_dict_val, batch_size, epochs, callback, verbose=True):
        """ Fit method that is used in the grid search.

        :param x_train: np.array of training feature vectors.
        :param x_val: np.array of validation feature vectors.
        :param y_dict_train: dictionary with as key the quality and as value the quality its values for the training data.
        :param y_dict_val: dictionary with as key the quality and as value the quality its values for the validation data.
        :param batch_size: the batch size.
        :param epochs: the number of epochs.
        :param callback: the talos specific callback.
        :param verbose: boolean indicating whether training info should be printed.
        :return: the History object that recorded the training process.
        """
        outputs_train = self._y_dict_to_outputs(y_dict_train)
        outputs_val = self._y_dict_to_outputs(y_dict_val)
        histories = {}
        for model_name in self.get_model_names():
            histories[model_name] = self.models[model_name].fit(self.scalers[model_name].transform(x_train), outputs_train[model_name],
                                                                batch_size=batch_size, epochs=epochs,
                                                                validation_data=[self.scalers[model_name].transform(x_val), outputs_val[model_name]],
                                                                callbacks=callback,
                                                                verbose=verbose)
        return histories

    # Necessary to compile DLASeR before training or fitting
    def compile_model(self, model_name, optimizer, lr, extensive_metrics=False):
        """ Compiles the deep learning model(s)

        :param model_name: the name of the model to compile.
                           (the model names can be fetched by the get_model_names() method)
        :param optimizer: the optimizer that the deep learning model should use.
        :param lr: the learning rate of the optimizer.
        :param extensive_metrics: boolean indicating whether additional metrics should be added
                (i.e., accuracy, f1-score, precision and recall)
        """
        assert model_name in self.get_model_names()  # Check if valid model name
        self.models[model_name].compile(optimizer=optimizer(lr=lr),
                                        loss=model_name_to_loss(model_name),
                                        metrics=model_name_to_metrics(model_name, extensive=extensive_metrics))

    # Necessary to compile DLASeR before training or fitting
    def compile(self, optimizers, lrs, extensive_metrics=False):
        """ Compiles all the deep learning models.

        :param optimizers: dictionary with as key the model name and as value the optimizer that the corresponding model should use.
        :param lrs: dictionary with as key the model name and as value the learning rate that the optimizer of the corresponding
                   model should use.
        :param extensive_metrics: boolean indicating whether additional metrics should be added
                (i.e., accuracy, f1-score, precision and recall)
        """
        for model_name in self.get_model_names():
            assert model_name in optimizers.keys() and model_name in lrs.keys()  # Check if all model names present
        for model_name in self.get_model_names():
            self.models[model_name].compile(optimizer=optimizers[model_name](lr=lrs[model_name]),
                                            loss=model_name_to_loss(model_name),
                                            metrics=model_name_to_metrics(model_name, extensive=extensive_metrics))

    def predict(self, x):
        """ Predicts for the given feature vectors.

        :param x: np.array containing feature vectors.
        :return: a tuple;
                    -> raw predictions of the deep learning models
                    -> the names of the associated model (contains info on the goal and quality)
        """
        pred = []
        model_names = self.get_model_names()
        for model_name in model_names:
            pred.append(self.models[model_name].predict(self.scalers[model_name].transform(x), use_multiprocessing=True))
        return pred, model_names

    def save_models(self, filename):
        """ Save the models at the given filename.

        :param filename: the filename where the models should be stored at.
        """
        for model_name in self.get_model_names():
            self.models[model_name].save(filename + model_name + '.h5')

    def load_models(self, filename):
        """ Load the deep learning models from the given filename.

        :param filename: the filename where the models are stored.
        """
        for model_name in self.get_model_names():
            self.models[model_name].load_weights(filename + model_name + '.h5')

    def get_model_names(self):
        """ Retrieve the list of model names.

        :return: array containing the model names.
        """
        return list(self.models.keys())

    def get_models(self):
        """ Gets the deep learning models.

        :return: dictionary with as key the model name and as value the keras deep learning model.
        """
        return self.models

    def print_models(self):
        """ Prints a structured overview of the deep learning models.
        """
        for model_name in self.get_model_names():
            print('\n' + '~'*25 + '~' + model_name)
            print('Scaler: ' + self.scalers[model_name].__class__.__name__)
            print(self.models[model_name].summary())
