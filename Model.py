from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dropout, BatchNormalization


class Model():

    def __init__(self, img_width, img_height, img_depth, num_classes, X_train, X_valid, optims):
        self.img_width = img_width
        self.img_height = img_height
        self.img_depth = img_depth
        self.num_classes = num_classes
        self.X_train = X_train
        self.X_valid = X_valid
        self.optims = optims

    def build_net(self):

        ##
        print("Model    -- start --")
        ##

        # create a sequencial model
        net = Sequential(name='DCNN')

        # add layers to the model
        net.add(
            Conv2D(
                filters=32,
                kernel_size=(5, 5),
                input_shape=(self.img_width, self.img_height, self.img_depth),
                activation='elu',
                padding='same',
                kernel_initializer='he_normal',
                name='conv2d_1'
            )
        )
        net.add(BatchNormalization(name='batchnorm_1'))
        net.add(
            Conv2D(
                filters=64,
                kernel_size=(5, 5),
                activation='elu',
                padding='same',
                kernel_initializer='he_normal',
                name='conv2d_2'
            )
        )
        net.add(BatchNormalization(name='batchnorm_2'))
        net.add(
            Conv2D(
                filters=128,
                kernel_size=(5, 5),
                activation='elu',
                padding='same',
                kernel_initializer='he_normal',
                name='conv2d_3'
            )
        )
        net.add(BatchNormalization(name='batchnorm_3'))
        net.add(MaxPooling2D(pool_size=(2, 2), name='maxpool2d_1'))
        net.add(Dropout(0.2, name='dropout_1'))

        net.add(Flatten(name='flatten'))
        net.add(
            Dense(
                128,
                activation='elu',
                kernel_initializer='he_normal',
                name='dense_1'
            )
        )
        net.add(BatchNormalization(name='batchnorm_5'))
        net.add(Dropout(0.4, name='dropout_2'))

        net.add(
            Dense(
                self.num_classes,
                activation='softmax',
                name='out_layer'
            )
        )

        # compile the model
        net.compile(
            loss='categorical_crossentropy',
            optimizer=self.optims[0],
            metrics=['accuracy']
        )

        # get a summary of the model
        net.summary()

        ##
        print("Model    -- end --")
        ##

        return net


