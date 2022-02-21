import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Multiply, Add
from tensorflow.keras import Model
from models.FER_model.resnext_block import GroupConv2D

class ResidualBlock34(Model) :
    def __init__(self, block_type = None, n_filters = None, cardinality = None, se = None) :
        super(ResidualBlock34, self).__init__()
        self.n_filters = n_filters
        self.se = se
        if block_type == 'identity' :
            self.strides = 1
        elif block_type == 'conv' :
            self.strides = 2
            self.conv_shortcut = Conv2D(filters = self.n_filters, 
                                        kernel_size = 1, 
                                        padding = 'same',
                                        strides = self.strides,
                                        kernel_initializer = 'he_normal')
            self.bn_shortcut = BatchNormalization(momentum = 0.9)
        self.conv_1 = Conv2D(filters = self.n_filters,
                             kernel_size = 3, 
                             padding = 'same',
                             strides = self.strides,
                             kernel_initializer = 'he_normal')
        self.bn_1 = BatchNormalization(momentum = 0.9)
        self.relu_1 = ReLU()
        
        if not cardinality : 
            self.conv_2 = Conv2D(filters = self.n_filters,
                                 kernel_size = 3,
                                 padding = 'same',
                                 kernel_initializer = 'he_normal')
        else : 
            self.conv_2 = GroupConv2D(input_channels=self.n_filters,
                                      output_channels=self.n_filters,
                                      kernel_size=(3, 3),
                                      #strides=strides,
                                      padding="same",
                                      groups=cardinality)
            
        self.bn_2 = BatchNormalization(momentum = 0.9)
        self.relu_2 = ReLU()
        self.CA = CA(self.n_filters)
        self.SA = SA()
        self.Mul = Multiply()
        self.Add = Add()

    def call(self, input) :
        shortcut = input
        if self.strides == 2 :
            shortcut = self.conv_shortcut(shortcut)
            shortcut = self.bn_shortcut(shortcut)
        x = self.conv_1(input)
        x = self.bn_1(x)
        x = self.relu_1(x)
        x = self.conv_2(x)
        x = self.bn_2(x)
        x = tf.add(shortcut, x)
        x = self.relu_2(x)
        
        if self.se == 'CA' :
            x = self.CA(x)
        elif self.se == 'SA' :
            x = self.SA(x)
        elif self.se == 'serial_CA_SA' :
            x = self.CA(x)
            x = self.SA(x)
        elif self.se == 'serial_SA_CA' :
            x = self.SA(x)
            x = self.CA(x)
        elif self.se == 'parallel_mul' :
            ca = self.CA(x)
            sa = self.SA(x)
            x = self.Mul([ca, sa])
        elif self.se == 'parallel_add' :
            ca = self.CA(x)
            sa = self.SA(x)
            x = self.Add([ca, sa])

        return x


class CA(Model) :
    def __init__(self, x, r=4) :
        super(CA, self).__init__()
        self.ch = x
        self.ch_r = x/r
        self.GAP = GlobalAveragePooling2D()
        self.FC1 = Dense(self.ch_r)
        self.FC2 = Dense(self.ch, activation = 'sigmoid')
        self.relu = ReLU()
        self.Mul = Multiply()

        
    def call(self, x, r=4) :
        squeeze = self.GAP(x)
        excitation = self.FC1(squeeze)
        excitation = self.relu(excitation)
        excitation = self.FC2(excitation)

        output = self.Mul([x, excitation])

        return output

class SA(Model) : 
    def __init__(self) :
        super(SA, self).__init__()
        self.conv = Conv2D(1, kernel_size = (3, 3), strides = (1, 1), padding = 'same', activation = 'sigmoid')
        self.Mul = Multiply()

    def call(self, x) :
        conv = self.conv(x)
        output = self.Mul([x, conv])

        return output
