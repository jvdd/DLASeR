import numpy as np

from src.dl_framework import DLFramework
from src.utils import build_core_model, add_classification_head, add_regression_head, operator_to_math, \
    layer_names_to_loss, layer_names_to_metrics, equal_regr_and_class_weights


class DLASeRPlus(DLFramework):

    def __init__(self, qualities, input_size, scaler, core_layers):
        """ Creates the DLASeR+ framework.

        :param qualities: the qualities for which goals are defined.
        :param input_size: the length of the feature vector.
        :param scaler: the scaler accompanying the deep learning model.
        :param core_layers: array specifying the width of the core layers.
        """
        super().__init__(qualities)
        self.scaler = scaler
        self.model = build_core_model(input_size, core_layers)

    def add_threshold_goal(self, quality, operator, threshold_value, class_layers):
        """ Add a threshold goal to the DLASeR+ framework.

        :param quality: the quality for which the threshold goal is defined.
        :param operator: the operator of the threshold goal (below = Min, above = Max).
        :param threshold_value: the threshold value of the threshold goal.
        :param class_layers: array specifying the width of the layers in the classification head.
        """
        super().add_threshold_goal(quality, operator, threshold_value)
        name = self._get_threshold_name(quality, threshold_value)
        self.model = add_classification_head(self.model, name, class_layers)

    def add_setpoint_goal(self, quality, setpoint_value, epsilon, class_layers):
        """ Add a set-point goal to the DLASeR+ framework.

        :param quality: the quality for which the set-point goal is defined.
        :param setpoint_value: the set-point value of the set-point goal.
        :param epsilon: the error (or margin) that defines the set-point goal.
        :param class_layers: array specifying the width of the layers in the classification head.
        """
        super().add_setpoint_goal(quality, setpoint_value, epsilon)
        name = self._get_setpoint_name(quality, setpoint_value)
        self.model = add_classification_head(self.model, name, class_layers)

    def add_optimization_goal(self, quality, operator, regr_layers):
        """ Add an optimization goal to the DLASeR+ framework.

        :param quality: the quality for which the optimization goal is defined.
        :param operator: the operator of the optimization goal (minimization = Min, maximization = Max).
        :param regr_layers: array specifying the width of the layers in the regression head.
        """
        super().add_optimization_goal(quality, operator)
        name = self._get_optimization_name(quality)
        self.model = add_regression_head(self.model, name, regr_layers)

    def _y_dict_to_output(self, y_dict):
        """ Generates a mapping of given dictionary (quality, values) to another dictionary (output_layer_name, values).
        This mapping also processes the values of the accompanying quality for the goal(s) on that quality.

        :param y_dict: dictionary with as key the quality and as value the quality its values.
        :return: dictionary with as key the output layer name and as value the processed values of the
                    accompanying quality according to the goal of the output layer.
        """
        for quality in y_dict.keys():
            assert quality in self.qualities  # Make sure the quality is known
        outputs = {}
        # Threshold goals
        for name, threshold_goal in self.threshold_goals.items():
            quality = threshold_goal.quality
            quality_vals = y_dict[quality]
            operator_math = operator_to_math(threshold_goal.operator)
            y = np.array([operator_math(val, threshold_goal.value) for val in quality_vals]).astype(np.int)
            outputs['output_class_' + name] = y
        # Set-point goals
        for name, setpoint_goal in self.setpoint_goals.items():
            quality = setpoint_goal.quality
            quality_vals = y_dict[quality]
            setpoint = setpoint_goal.value
            epsilon = setpoint_goal.epsilon
            y = np.array([setpoint - epsilon < val < setpoint + epsilon for val in quality_vals]).astype(np.int)
            outputs['output_class_' + name] = y
        # Optimization goals
        for name, optimization_goal in self.optimization_goals.items():
            quality = optimization_goal.quality
            y = y_dict[quality]
            outputs['output_regr_' + name] = y
        return outputs

    # The online train method
    def train(self, x, y_dict, batch_size, epochs, verbose=True):
        """ The train method that is used in the online (training) cycles.

        :param x: np.array of train feature vectors.
        :param y_dict: dictionary with as key the quality and as value the quality its values for the training data.
        :param batch_size: the batch size.
        :param epochs: the number of epochs.
        :param verbose: boolean indicating whether training info should be printed.
        :return: the History object that recorded the training process.
        """
        # self.scaler.partial_fit(x)
        return self.model.fit({'main_input': self.scaler.transform(x)},
                              self._y_dict_to_output(y_dict),
                              batch_size=batch_size, epochs=epochs,
                              callbacks=self.callback,
                              validation_split=0.1,
                              verbose=verbose)

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
        :return: the History object that recorded the training process.
        """
        self.scaler.fit(x_train)
        return self.model.fit({'main_input': self.scaler.transform(x_train)}, self._y_dict_to_output(y_dict_train),
                              batch_size=batch_size, epochs=epochs,
                              validation_data=[{'main_input': self.scaler.transform(x_val)}, self._y_dict_to_output(y_dict_val)],
                              callbacks=self.callback,
                              verbose=verbose)

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
        return self.model.fit({'main_input': self.scaler.transform(x_train)}, self._y_dict_to_output(y_dict_train),
                              batch_size=batch_size, epochs=epochs,
                              validation_data=[{'main_input': self.scaler.transform(x_val)}, self._y_dict_to_output(y_dict_val)],
                              callbacks=[callback],
                              verbose=verbose)

    # Necessary to compile DLASeR+ before training or fitting
    def compile(self, optimizer, lr, extensive_metrics=False):
        """ Compiles the deep learning model.

        :param optimizer: the optimizer that the deep learning model should use.
        :param lr: the learning rate of the optimizer.
        :param extensive_metrics: boolean indicating whether additional metrics should be added
                (i.e., accuracy, f1-score, precision and recall). Default = False.
        """
        outputs_names = []
        if self.model.layers[-1].name.startswith(
                'output_'):  # In the case that the model contains already some classification / regression heads
            idx = -1
            while self.model.layers[idx].name.startswith('output_'):
                outputs_names.append(self.model.layers[idx].name)
                idx -= 1

        self.model.compile(optimizer=optimizer(lr=lr),
                           loss=layer_names_to_loss(outputs_names),
                           metrics=layer_names_to_metrics(outputs_names, extensive=extensive_metrics),
                           loss_weights=equal_regr_and_class_weights(outputs_names))

    def predict(self, x):
        """ Predicts for the given feature vectors.

        :param x: np.array containing feature vectors.
        :return: a tuple;
                    -> raw predictions of the deep learning model (the classification and regression heads)
                    -> the names of the associated layer (contains info on the goal and quality)
        """
        pred = self.model.predict(self.scaler.transform(x), use_multiprocessing=True)
        names = []
        idx = -1
        while self.model.layers[idx].name.startswith('output_'):
            names.append(self.model.layers[idx].name)
            idx -= 1
        return pred, names[::-1]  # Apparently keras predicts in the reverse order

    def save_model(self, filename):
        """ Save the model at the given filename.

        :param filename: the filename where the model should be stored at.
        """
        self.model.save(filename + '.h5')

    def load_model(self, filename):
        """ Load the deep learning model from the given filename.

        :param filename: the filename where the model is stored.
        """
        self.model.load_weights(filename + '.h5')

    def get_model(self):
        """ Gets the deep learning model.

        :return: the keras deep learning model.
        """
        return self.model

    def print_model(self):
        """ Prints a structured overview of the deep learning model.
        """
        print('Scaler: ' + self.scaler.__class__.__name__)
        self.model.summary()