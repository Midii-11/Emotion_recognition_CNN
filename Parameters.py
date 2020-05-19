from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class Parameters():
    def __init__(self, X_train):
        self.X_train = X_train


    def parameters(self):

        ##
        print("Parameters    -- start --")
        ##

        # Initialize callbacks
        early_stopping = EarlyStopping(monitor='val_accuracy', min_delta=0.00005, patience=11,
            verbose=1, restore_best_weights=True)

        lr_scheduler = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5,
            patience=7, min_lr=1e-7, verbose=1)

        tensorboard_callback = TensorBoard(log_dir="logs/fit/" + "One", histogram_freq=1)

        # store callbacks to array (array is expected by Keras)
        callbacks = [
            early_stopping,
            lr_scheduler,
            tensorboard_callback
        ]

        # Generates new input data from the input-set (increasing the diversity of the set)
        train_datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.15,
            height_shift_range=0.15,
            shear_range=0.15,
            zoom_range=0.15,
            horizontal_flip=True,
        )
        train_datagen.fit(self.X_train)

        # setup model parameters
        batch_size = 32
        epochs = 100
        optims = [
            optimizers.Adam(0.001)
        ]

        ##
        print("Parameters    -- end --")
        ##

        return callbacks, train_datagen, batch_size, epochs, optims