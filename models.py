import tensorflow as tf
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Activation, Dense, BatchNormalization, Dropout, LSTM, Concatenate, Lambda, LeakyReLU, Softmax, Input
from tensorflow.keras.activations import swish, sigmoid, tanh
import numpy as np
# from config import configs
from tensorflow.keras.utils import get_custom_objects
def get_model(configs):
    FE_layers = configs['feature_extractor_layers']
    lstm_num = configs['lstm_num']
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
            # img_feature, audio_feature = x
            h = self.FE(img_feature, audio_feature)
            va = self.va_classifier(h)
            expr = self.expr_classifier(h)
            au = self.au_classifier(h)
            # au = self.softmax_temp(au)
            return h, va, expr, au
        def build_graph(self):
            img_input = tf.keras.Input(shape=(6, 1, 512))
            audio_input = tf.keras.Input(shape=(1000))
            x = (img_input, audio_input)
            return tf.keras.Model(inputs=x, outputs=self.call(x))


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
            # try:
            #     img_feature = tf.squeeze(img_feature, [2])
            img_feature = self.img_reshape(img_feature)
            #     # img_feature = tf.squeeze(img_feature)
            #     # print(img_feature.shape)
            # except:
            #     # print(img_feature.shape)
            #     print()
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

    # img_feature = Input(shape=(6,1,512), name='img_in')
    # audio_feature = Input(shape=(1000), name='audio_in')
    # list_input = [img_feature, audio_feature]
    # # feature extractor
    # img_feature = LSTM(lstm_num, activation='relu', dtype='float32')(img_feature)
    #
    # h = Concatenate([img_feature, audio_feature])
    # for i in FE_layers:
    #     h = Dense(i)(h)
    #     h = BatchNormalization()(h)
    #     h = swish(h)
    #     h = Dropout(dropout_rate)(h)
    # feature_out = h
    # # va classifier
    # va_h = h
    # for i in classifier_layers:
    #     va_h = Dense(i)(va_h)
    #     va_h = BatchNormalization()(va_h)
    #     va_h = swish(va_h)
    #     # h = Dropout(dropout_rate)(h)
    # va_out =Dense(2, activation='tanh')(va_h)
    # # expr classifier
    # erxpr_h = h
    # for i in classifier_layers:
    #     erxpr_h = Dense(i)(erxpr_h)
    #     erxpr_h = BatchNormalization()(erxpr_h)
    #     erxpr_h = swish(erxpr_h)
    #     # h = Dropout(dropout_rate)(h)
    # expr_out = Dense(8)(erxpr_h)
    # # au classifier
    # au_h = h
    # for i in classifier_layers:
    #     au_h = Dense(i)(au_h)
    #     au_h = BatchNormalization()(au_h)
    #     au_h = swish(au_h)
    #     # h = Dropout(dropout_rate)(h)
    # au_out = Dense(12, activation='sigmoid')(au_h)
    # output_list = [feature_out, va_out, expr_out, au_out]

    # lstm_num = 512
    # FE_layers = [1024]
    # classifier_layers = [512, 256, 128]
    # dropout_rate = 0.2

    class dense_module(tf.keras.Model):
        def __init__(self, layer_num, activation, drop_rate=0):
            super(dense_module, self).__init__()
            self.dense = Dense(layer_num)
            self.bn = BatchNormalization()
            self.activation = activation
            self.dropout = Dropout(drop_rate)

        def call(self, x):
            h = self.dense(x)
            h = self.bn(h)
            h = self.activation(h)
            h = self.dropout(h)
            return h

    img_input = Input(shape=(6, 1, 512), name='img_in')
    audio_input = Input(shape=(1000), name='audio_in')

    # feature extractor
    img_feature = Reshape((6, 512))(img_input)
    img_feature = LSTM(lstm_num, activation='relu', dtype='float32', name='img_lstm')(img_feature)
    h = Concatenate()([img_feature, audio_input])

    for i in FE_layers:
        h = dense_module(i, swish, drop_rate=configs['dropout_rate'])(h)
    feature_out = h

    # va classifier
    va_h = h
    for i in classifier_layers:
        va_h = dense_module(i, swish)(va_h)
    va_h = Dense(2)(va_h)
    va_out = tanh(va_h)

    # expr classifier
    erxpr_h = h
    for i in classifier_layers:
        erxpr_h = dense_module(i, swish)(erxpr_h)
    expr_out = Dense(8)(erxpr_h)

    # au classifier
    au_h = h
    for i in classifier_layers:
        au_h = dense_module(i, swish)(au_h)
    au_h = Dense(12)(au_h)
    au_out = sigmoid(au_h)

    input_list = [img_input, audio_input]
    output_list = [feature_out, va_out, expr_out, au_out]

    model = tf.keras.Model(
        inputs = input_list,
        outputs= output_list
    )


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
    # model = model()
    # img_input = tf.random.uniform((1,6,1,512))
    # audio_input = tf.random.uniform((1,1000))
    # model = model.build_graph()
    # tmp = model(img_input, audio_input)
    # model.save("/home/euiseokjeong/Desktop/IMLAB/ABAW/result/keep/teacher_test/test_model")
    return model
