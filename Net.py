from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense,\
    Conv2D, MaxPooling2D, Dropout, BatchNormalization
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class Net():
    def __init__(self, X_train, X_valid, y_train, y_valid):
        # Normalizing results, as neural networks are very sensitive to unnormalized data. ### WHY not Y ?
        self.X_train = X_train / 255
        self.X_valid = X_valid / 255
        self.y_train = y_train
        self.y_valid = y_valid

        self.img_width = X_train.shape[1]
        self.img_height = X_train.shape[2]
        self.img_depth = X_train.shape[3]
        self.num_classes = y_train.shape[1]

    def build_net(self, optim):

        """
        This is a Deep Convolutional Neural Network (DCNN). For generalization purpose I used dropouts in regular intervals.
        I used `ELU` as the activation because it avoids dying relu problem but also performed well as compared to LeakyRelu
        atleast in this case. `he_normal` kernel initializer is used as it suits ELU. BatchNormalization is also used for better
        results.
        """

        img_width = self.img_width
        img_height = self.img_height
        img_depth = self.img_depth
        num_classes = self.num_classes

        net = Sequential(name='DCNN')

        net.add(
            Conv2D(
                filters=64,
                kernel_size=(5, 5),
                input_shape=(img_width, img_height, img_depth),
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

        net.add(MaxPooling2D(pool_size=(2, 2), name='maxpool2d_1'))
        net.add(Dropout(0.4, name='dropout_1'))

        net.add(
            Conv2D(
                filters=128,
                kernel_size=(3, 3),
                activation='elu',
                padding='same',
                kernel_initializer='he_normal',
                name='conv2d_3'
            )
        )
        net.add(BatchNormalization(name='batchnorm_3'))
        net.add(
            Conv2D(
                filters=128,
                kernel_size=(3, 3),
                activation='elu',
                padding='same',
                kernel_initializer='he_normal',
                name='conv2d_4'
            )
        )
        net.add(BatchNormalization(name='batchnorm_4'))

        net.add(MaxPooling2D(pool_size=(2, 2), name='maxpool2d_2'))
        net.add(Dropout(0.4, name='dropout_2'))

        net.add(
            Conv2D(
                filters=256,
                kernel_size=(3, 3),
                activation='elu',
                padding='same',
                kernel_initializer='he_normal',
                name='conv2d_5'
            )
        )
        net.add(BatchNormalization(name='batchnorm_5'))
        net.add(
            Conv2D(
                filters=256,
                kernel_size=(3, 3),
                activation='elu',
                padding='same',
                kernel_initializer='he_normal',
                name='conv2d_6'
            )
        )
        net.add(BatchNormalization(name='batchnorm_6'))

        net.add(MaxPooling2D(pool_size=(2, 2), name='maxpool2d_3'))
        net.add(Dropout(0.5, name='dropout_3'))

        net.add(Flatten(name='flatten'))

        net.add(
            Dense(
                128,
                activation='elu',
                kernel_initializer='he_normal',
                name='dense_1'
            )
        )
        net.add(BatchNormalization(name='batchnorm_7'))

        net.add(Dropout(0.6, name='dropout_4'))

        net.add(
            Dense(
                num_classes,
                activation='softmax',
                name='out_layer'
            )
        )

        net.compile(
            loss='categorical_crossentropy',
            optimizer=optim,
            metrics=['accuracy']
        )

        net.summary()

        return net



    def runnetwork(self):

        X_train = self.X_train
        X_valid = self.X_valid
        y_train = self.y_train
        y_valid = self.y_valid


        """
                I used two callbacks one is `early stopping` for avoiding overfitting training data
                and other `ReduceLROnPlateau` for learning rate.
                """

        early_stopping = EarlyStopping(
            monitor='val_accuracy',
            min_delta=0.00005,
            patience=11,
            verbose=1,
            restore_best_weights=True,
        )

        lr_scheduler = ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.5,
            patience=7,
            min_lr=1e-7,
            verbose=1,
        )

        callbacks = [
            early_stopping,
            lr_scheduler,
        ]

        # As the data in hand is less as compared to the task so ImageDataGenerator is good to go.
        train_datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.15,
            height_shift_range=0.15,
            shear_range=0.15,
            zoom_range=0.15,
            horizontal_flip=True,
        )
        train_datagen.fit(X_train)

        batch_size = 32  # batch size of 32 performs the best.
        epochs = 100
        optims = [
            optimizers.Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name='Nadam'),
            optimizers.Adam(0.001),
        ]

        # I tried both `Nadam` and `Adam`, the difference in results is not different but I finally went with Nadam as it is more popular.
        model = Net.build_net(self, optims[1])
        history = model.fit_generator(
            train_datagen.flow(X_train, y_train, batch_size=batch_size),
            validation_data=(X_valid, y_valid),
            steps_per_epoch=len(X_train) / batch_size,
            epochs=epochs,
            callbacks=callbacks,
            use_multiprocessing=True
        )
        