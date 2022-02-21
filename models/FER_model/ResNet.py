from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, ReLU, MaxPool2D, GlobalAveragePooling2D
from tensorflow.keras import Model, Sequential
from models.FER_model.ResidualBlock import ResidualBlock34
# from ResidualBlock import ResidualBlock34

class ResNet34(Model) :
    def __init__(self, cardinality = None, se = None, n_classes = 2) :
        super(ResNet34, self).__init__()
        self.n_classes = n_classes
        self.cardinality = cardinality
        self.se = se
        self.conv_1 = Conv2D(filters = 64,
                             kernel_size = 7,
                             padding = 'same',
                             strides = 2,
                             kernel_initializer = 'he_normal',
                             name = 'ResNet_1')
        self.bn_1 = BatchNormalization(momentum = 0.9, name = 'ResNet_1_BN')
        self.relu_1 = ReLU(name = 'ResNet_1_Act')
        self.maxpool = MaxPool2D(3, 2,
                                    padding = 'same', 
                                    name = 'ResNet_1_Pool')
        self.residual_blocks = Sequential()
        for n_filters, reps, downscale in zip([64, 128, 256, 512],
                                              [3, 4, 6, 3],
                                              [False, True, True, True]) :
            for i in range(reps) :
                if i == 0 and downscale :
                    self.residual_blocks.add(ResidualBlock34(block_type = 'conv', 
                                                 n_filters = n_filters, cardinality = self.cardinality, se = self.se))
                else : 
                    self.residual_blocks.add(ResidualBlock34(block_type = 'identity',
                                             n_filters = n_filters, cardinality = self.cardinality, se = self.se))
        self.GAP = GlobalAveragePooling2D()
        self.FC = Dense(units = self.n_classes, activation = 'tanh')


    def call(self, input, training = False) :
        x = self.conv_1(input)
        x = self.bn_1(x)
        x = self.relu_1(x)
        x = self.maxpool(x)
        x = self.residual_blocks(x)
        x = self.GAP(x)
        x = self.FC(x)

        return x
