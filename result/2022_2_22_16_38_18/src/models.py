import tensorflow as tf
from tensorflow.keras.layers import Activation, Dense, BatchNormalization, Dropout, LSTM, Concatenate, Lambda, LeakyReLU, Softmax
import numpy as np
from tensorflow.keras.utils import get_custom_objects
def get_model(configs, teacher):
    FE_layers = configs['feature_extractor_layers']
    lstm_layers = configs['latm_layers']
    classifier_layers = configs['classifier_layers']
    dropout_rate = configs['dropout_rate']
    if teacher:
        temp = 1
    else:
        temp =configs['temperature']

    # class t_softmax(Activation):
    #     def __init__(self, activation, **kwargs):
    #         super(t_softmax, self).__init__(activation, **kwargs)
    #         self.__name__='t_softmax'
    # def t_softmax(x):
    #     return np.exp(x/temp) / np.sum(np.exp(x/temp))
    # get_custom_objects().update({'t_softmax':t_softmax(t_softmax)})


    class feature_extractor(tf.keras.Model):
        def __init__(self):
            super(feature_extractor, self).__init__()
            self.lstm_1 = LSTM(512, activation='relu', dropout=dropout_rate)
            self.dense_1 = Dense(1024, activation=LeakyReLU(alpha=0.3))
            # self.dense_1 = Dense(1024, activation='relu')
            self.bn_1 = BatchNormalization()
            self.dense_2 = Dense(512, activation=LeakyReLU(alpha=0.3))
            # self.dense_2 = Dense(512, activation='relu')
            self.bn_2 = BatchNormalization()
            self.dense_3 = Dense(256, activation=LeakyReLU(alpha=0.3))
            # self.dense_3 = Dense(256, activation='relu')
            self.bn_3 = BatchNormalization()
            # self.FE_layers = [(Dense(i, activation='relu'), Dropout(dropout_rate), BatchNormalization()) for i in FE_layers]

        def call(self, img_feature, audio_feature):
            img_feature = self.lstm_1(img_feature)
            feature = Concatenate()([img_feature, audio_feature])
            h = self.dense_1(feature)
            h = self.bn_1(h)
            h = self.dense_2(h)
            h = self.bn_2(h)
            output = self.dense_3(h)
            return output
    # class softmax_with_temperature():
    #     def __init__(self, temperature):
    #         super(softmax_with_temperature, self).__init__()
    #         self.t = temperature
    #         self.softmax = tf.keras.layers.Softmax()
    #     def call(self, input):
    #         input = input / self.t
    #         return self.softmax(input)

    class classifier(tf.keras.Model):
        def __init__(self, task):
            super(classifier, self).__init__()
            self.dense_1 = Dense(128, activation=LeakyReLU(alpha=0.3))
            # self.dense_1 = Dense(128, activation='relu')
            self.bn1 = BatchNormalization()
            # self.t_softmax = softmax_with_temperature(temp)
            if task == 'VA':
                self.dense_2 = Dense(2, activation='tanh')
            elif task == 'EXPR':
                self.dense_2 = Dense(8)
                # self.dense_2 = Dense(8)
            elif task == 'AU':
                self.dense_2 = Dense(12, activation='sigmoid')
            else:
                raise ValueError(f"Task {task} is not valid!")
        def call(self, input):
            h = self.dense_1(input)
            h = self.bn1(h)
            output = self.dense_2(h)
            return output
        def softmax_with_temperature(self, x):
            logits = x / configs['temperature']
            return np.exp(logits) / np.sum(np.exp(logits))
        # def temp_softmax(self, x):


    # def custom_relu(x):
    #     return K.maximum(0.0, x)
    #
    # model = tf.keras.models.Sequential([
    #     tf.keras.layers.Flatten(input_shape=(128, 128)),
    #     tf.keras.layers.Dense(512),
    #     tf.keras.layers.Lambda(custom_relu),
    class model(tf.keras.Model):
        def __init__(self):
            super(model, self).__init__()
            self.FE = feature_extractor()
            self.va_classifier = classifier('VA')
            self.expr_classifier = classifier('EXPR')
            self.au_classifier = classifier('AU')
            # self.softmax_temp = Lambda(self.softmax_with_temperature)

        def call(self, img_feature, audio_feature):
            img_feature = tf.squeeze(img_feature)
            h = self.FE(img_feature, audio_feature)
            va = self.va_classifier(h)
            expr = self.expr_classifier(h)
            au = self.au_classifier(h)
            # au = self.softmax_temp(au)

            return h, va, expr, au



    model = model()
    # model_sans_softmax = tf.keras.Model(inputs=model.input, outputs=model.output)
    return model