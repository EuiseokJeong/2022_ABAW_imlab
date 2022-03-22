import tensorflow as tf
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Activation, Dense, BatchNormalization, Dropout, LSTM, Concatenate, Lambda, LeakyReLU, Softmax, Input
from tensorflow.keras.activations import swish, sigmoid, tanh, softmax
import numpy as np
# from config import configs
from tensorflow.keras.utils import get_custom_objects
def get_model(configs):
    FE_layers = configs['feature_extractor_layers']
    lstm_num = configs['lstm_num']
    classifier_layers = configs['classifier_layers']
    dropout_rate = configs['dropout_rate']
    domain_layers = configs['domain_layers']
    adaptation_factor = configs['adaptation_factor']

    class dense_module(tf.keras.Model):
        def __init__(self, layer_num, activation, name, drop_rate=0):
            super(dense_module, self).__init__()
            self.dense = Dense(layer_num, name=f"{name}_dense")
            self.bn = BatchNormalization(name=f"{name}_batch")
            self.activation = activation
            self.dropout = Dropout(drop_rate,name=f"{name}_dropout")

        def call(self, x):
            h = self.dense(x)
            h = self.bn(h)
            h = self.activation(h)
            h = self.dropout(h)
            return h

    @tf.custom_gradient
    def grad_reverse(x):
        y = tf.identity(x)
        def custom_grad(dy):
            return -dy
        return y, custom_grad
    class GradReverse(tf.keras.layers.Layer):
        def __init__(self):
            super().__init__()
        def call(self, x):
            return adaptation_factor * grad_reverse(x)

    img_input = Input(shape=(6, 1, 512), name='img_in')
    audio_input = Input(shape=(1000), name='audio_in')

    # feature extractor
    img_feature = Reshape((6, 512))(img_input)
    img_feature = LSTM(lstm_num, activation='relu', dtype='float32', name=f'img_lstm_{lstm_num}')(img_feature)
    h = Concatenate()([img_feature, audio_input])

    for i in FE_layers:
        h = dense_module(i, swish, drop_rate=dropout_rate, name=f"FE_{i}")(h)

    # domain classifier
    domain_h = GradReverse()(h)
    for i in domain_layers:
        domain_h = dense_module(i, swish, drop_rate=dropout_rate, name=f"DC_{i}")(domain_h)
    domain_h = Dense(3)(domain_h)
    domain_out = softmax(domain_h)

    # va classifier
    va_h = h
    for i in classifier_layers:
        va_h = dense_module(i, swish, name=f"VA_{i}")(va_h)
    va_h = Dense(2)(va_h)
    va_out = tanh(va_h)

    # expr classifier
    erxpr_h = h
    for i in classifier_layers:
        erxpr_h = dense_module(i, swish, name=f"EXPR_{i}")(erxpr_h)
    expr_h = Dense(8)(erxpr_h)

    # au classifier
    au_h = h
    for i in classifier_layers:
        au_h = dense_module(i, swish, name=f"AU_{i}")(au_h)
    au_h = Dense(12)(au_h)
    au_out = sigmoid(au_h)

    input_list = [img_input, audio_input]
    output_list = [domain_out,va_out, expr_h, au_out]

    model = tf.keras.Model(
        inputs=input_list,
        outputs=output_list
    )
    return model
