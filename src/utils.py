
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, Dropout, BatchNormalization
from keras.regularizers import l2, l1

### Operator

class Enum(tuple): __getattr__ = tuple.index
Operator = Enum(['Min', 'Max'])

def operator_to_thresh_str(operator):
    if operator == Operator.Min:
        return '<'
    elif operator == Operator.Max:
        return '>'
    raise ValueError('Given operator is not valid: ' + operator)

def operator_to_opt_str(operator):
    if operator == Operator.Min:
        return 'Minimize'
    elif operator == Operator.Max:
        return 'Maximize'
    raise ValueError('Given operator is not valid: ' + operator)

def operator_to_math(operator):
    if operator == Operator.Min:
        return lambda x, y: x < y
    elif operator == Operator.Max:
        return lambda x, y: x > y
    raise ValueError('Given operator is not valid: ' + operator)



### Keras metrics

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))



### DLASeR utils

def create_simple_ANN_model(input_size, layers, output_activation, output_dim=1):
    inp = Input(shape=(input_size,), dtype='float32')
    x = Dense(layers[0], activation='relu', kernel_regularizer=l1(0.0005))(inp)
    # x = Dropout(0.1)(x)
    for layer in layers[1:]:
        x = BatchNormalization()(x)
        x = Dense(layer, activation='relu', kernel_regularizer=l2(0.0005))(x)
    x = BatchNormalization()(x)
    x = Dense(layers[-1], activation='relu', kernel_regularizer=l2(0.0005))(x)
    outp = Dense(output_dim, activation=output_activation)(x)

    model = Model(inputs=[inp], outputs=[outp])

    return model


### DLASeR+ utils

# The core layers shared by all the specific tasks
def build_core_model(input_size, layers):
    inp = Input(shape=(input_size,), dtype='float32', name='main_input')
    x = Dense(layers[0], activation='relu', kernel_regularizer=l1(0.0005))(inp)
    # x = Dropout(0.1)(x)
    for layer in layers[1:-1]:
        x = BatchNormalization()(x)
        x = Dense(layer, activation='relu', kernel_regularizer=l2(0.0005))(x)
    x = BatchNormalization()(x)
    outp = Dense(layers[-1], activation='relu', kernel_regularizer=l2(0.0005), name='embedding')(x)

    model = Model(inputs=[inp], outputs=[outp])

    return model


## The adaptation goal heads

CLASSIFICATION_LOSS = 'binary_crossentropy'
CLASSIFICATION_METRICS = ['acc']
CLASSIFICATION_METRICS_EXTENSIVE = ['acc', f1_m, precision_m, recall_m]
REGRESSION_LOSS = 'mse'
REGRESSION_METRICS = []


def layer_names_to_loss(layer_names):
    # layer_names are the output layers
    loss = {}
    for layer_name in layer_names:
        if '_class_' in layer_name:  # In case of classification
            loss[layer_name] = CLASSIFICATION_LOSS
        elif '_regr_' in layer_name:  # In case of regression
            loss[layer_name] = REGRESSION_LOSS
        else:
            raise ValueError('Illegal layer name given: ' + layer_name)
    return loss


def layer_names_to_metrics(layer_names, extensive):
    # layer_names are the output layers
    metrics = {}
    for layer_name in layer_names:
        if '_class_' in layer_name:  # In case of classification
            if extensive:
                metrics[layer_name] = CLASSIFICATION_METRICS_EXTENSIVE
            else:
                metrics[layer_name] = CLASSIFICATION_METRICS
        elif '_regr_' in layer_name:  # In case of regression
            metrics[layer_name] = REGRESSION_METRICS
        else:
            raise ValueError('Illegal layer name given: ' + layer_name)
    return metrics


def equal_regr_and_class_weights(layer_names):
    weights = {}
    regr_count = sum([1 if '_regr_' in name else 0 for name in layer_names])
    class_count = len(layer_names) - regr_count
    if regr_count != 0 and class_count != 0:
        for name in layer_names:
            if '_regr_' in name:
                weights[name] = 1 / regr_count
            elif '_class_' in name:
                weights[name] = 1 / class_count
            else:
                raise ValueError('Illegal layer name given: ' + name)
        return weights
    else:  # Only of 1 type
        return [1] * (len(layer_names))


# Add classification head to core model (or a model with already several heads)
def add_classification_head(core_model, quality, layers):
    inp = core_model.get_layer('main_input').output
    embedding_layer = core_model.get_layer('embedding').output

    x = Dense(layers[0], activation='relu', kernel_regularizer=l1(0.0005))(embedding_layer)
    # x = Dropout(0.1)(x)
    for layer in layers[1:]:
        x = BatchNormalization()(x)
        x = Dense(layer, activation='relu', kernel_regularizer=l2(0.0005))(x)
    outp = Dense(1, activation='sigmoid', kernel_regularizer=l2(0.0005), name='output_class_' + quality)(x)

    outputs = []
    if core_model.layers[-1].name.startswith(
            'output_'):  # In the case that the model contains already some classification heads
        idx = -1
        while core_model.layers[idx].name.startswith('output_'):
            outputs.append(core_model.layers[idx].output)
            idx -= 1

    outputs.append(outp)

    model = Model(inputs=[inp], outputs=outputs)

    return model


# Add regression head to core model (or a model with already several heads)
def add_regression_head(core_model, quality, layers):
    inp = core_model.get_layer('main_input').output
    embedding_layer = core_model.get_layer('embedding').output

    x = Dense(layers[0], activation='relu', kernel_regularizer=l1(0.0005))(embedding_layer)
    # x = Dropout(0.1)(x)
    for layer in layers[1:]:
        x = BatchNormalization()(x)
        x = Dense(layer, activation='relu', kernel_regularizer=l2(0.0005))(x)
    outp = Dense(1, kernel_regularizer=l2(0.0005), name='output_regr_' + quality)(x)

    outputs = []
    if core_model.layers[-1].name.startswith(
            'output_'):  # In the case that the model contains already some classification heads
        idx = -1
        while core_model.layers[idx].name.startswith('output_'):
            outputs.append(core_model.layers[idx].output)
            idx -= 1

    outputs.append(outp)

    model = Model(inputs=[inp], outputs=outputs)

    return model
