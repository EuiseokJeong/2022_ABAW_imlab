import tensorflow as tf
from tensorflow.keras.layers import Activation, Dense, BatchNormalization, Dropout, LSTM, Concatenate, Lambda, LeakyReLU, Softmax, Input
from tensorflow.keras.activations import swish
import numpy as np
from config import configs
from tensorflow.keras.utils import get_custom_objects
def get_model(configs):
    FE_layers = configs['feature_extractor_layers']
    lstm_layers = configs['latm_layers']
    classifier_layers = configs['classifier_layers']
    dropout_rate = configs['dropout_rate']
    class classifier(tf.keras.Model):
        def __init__(self, task):
            super(classifier, self).__init__()
            self.classifier_layers = [[Dense(i), BatchNormalization(), swish, Dropout(dropout_rate)] for i in classifier_layers]
            # self.dense_1 = Dense(128, activation=LeakyReLU(alpha=0.3))
            # # self.dense_1 = Dense(128, activation='relu')
            # self.bn1 = BatchNormalization()
            # # self.t_softmax = softmax_with_temperature(temp)
            if task == 'VA':
                self.dense = Dense(2, activation='tanh')
            elif task == 'EXPR':
                self.dense = Dense(8)
            elif task == 'AU':
                self.dense = Dense(12, activation='sigmoid')
            else:
                raise ValueError(f"Task {task} is not valid!")
        def call(self, input):
            h = input
            for layers in self.classifier_layers:
                for layer in layers:
                    h = layer(h)

            # h = self.dense_1(input)
            # h = self.bn1(h)
            output = self.dense(h)
            return output

    class model(tf.keras.Model):
        def __init__(self):
            super(model, self).__init__()
            self.FE = feature_extractor()
            self.va_classifier = classifier('VA')
            self.expr_classifier = classifier('EXPR')
            self.au_classifier = classifier('AU')
            # self.softmax_temp = Lambda(self.softmax_with_temperature)

        def call(self, img_feature, audio_feature):
            h = self.FE(img_feature, audio_feature)
            va = self.va_classifier(h)
            expr = self.expr_classifier(h)
            au = self.au_classifier(h)
            # au = self.softmax_temp(au)
            return h, va, expr, au
        def build_graph(self):
            img_input = tf.keras.Input(shape=(6, 1, 512))
            audio_input = tf.keras.Input(shape=(1000))
            x = [img_input, audio_input]
            return tf.keras.Model(inputs=[img_input, audio_input], outputs=self.call(img_input, audio_input))


    class feature_extractor(tf.keras.Model):
        def __init__(self):
            super(feature_extractor, self).__init__()
            self.img_reshape = tf.keras.layers.Reshape((6,512))
            self.lstm_1 = LSTM(512, activation='relu', dtype='float32')
            # self.dense_1 = Dense(1024, activation=LeakyReLU(alpha=0.3), dtype='float32')
            # # self.dense_1 = Dense(1024, activation='relu')
            # self.bn_1 = BatchNormalization()
            # self.dense_2 = Dense(512, activation=LeakyReLU(alpha=0.3))
            # # self.dense_2 = Dense(512, activation='relu')
            # self.bn_2 = BatchNormalization()
            # self.dense_3 = Dense(256, activation=LeakyReLU(alpha=0.3))
            # # self.dense_3 = Dense(256, activation='relu')
            # self.bn_3 = BatchNormalization()
            self.FE_layers = [[Dense(i), BatchNormalization(), swish, Dropout(dropout_rate)] for i in FE_layers]

        def call(self, img_feature, audio_feature):
            img_feature = self.img_reshape(img_feature)
            img_feature = self.lstm_1(img_feature)
            feature = Concatenate()([img_feature, audio_feature])
            for layers in self.FE_layers:
                for layer in layers:
                    feature = layer(feature)
                # dense, bn, swish, dropout = layers
                # feature = dense(feature)
                # feature = bn(feature)
                # feature = swish(feature)
                # feature = dropout(feature)
            # h = self.dense_1(feature)
            # h = self.bn_1(h)
            # h = self.dense_2(h)
            # h = self.bn_2(h)
            # output = self.dense_3(h)
            return feature





    # img_input = tf.keras.Input(shape=(6,1,512))
    # img_input = tf.keras.layers.Reshape((6,512))(img_input)
    #
    # audio_input = tf.keras.Input(shape=(512))
    #
    # img_feature = LSTM(512, activation='relu')(img_input)
    # # img_feature = tf.squeeze(img_feature)
    # feature = Concatenate()([img_feature, audio_input])
    #
    # # feature extractor
    # # feature = feature_extractor()(img_input, audio_input)
    # feature = Dense(1024, activation=LeakyReLU(alpha=0.3))(feature)
    # feature = Dense(512, activation=LeakyReLU(alpha=0.3))(feature)
    #
    #
    # va = classifier('VA')(feature)
    # expr = classifier('EXPR')(feature)
    # au = classifier('AU')(feature)
    # model = tf.keras.Model(
    #     inputs = [img_input,audio_input],
    #     outputs=[va, expr, au]
    # )
    #
    # tf.keras.utils.plot_model(
    #     model, to_file='./model.png', show_shapes=False, show_dtype=False,
    #     show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96,
    #     layer_range=None, show_layer_activations=False
    # )
    model = model()
    return model
