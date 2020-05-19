import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils


class Preprocessing():
    def __init__(self, path):
        self.path = path

    def preprocessing(self):

        ##
        print("Preprocessing    -- start --")
        ##

        # read the data
        df = pd.read_csv(self.path)

        # convert pixels from list to ndarray
        img_array = df.pixels.apply(lambda x: np.array(x.split(' ')).reshape(48, 48, 1).astype('float32'))
        img_array = np.stack(img_array, axis=0)

        # One-Hot encode the emotions for each images
        le = LabelEncoder()
        img_labels = le.fit_transform(df.emotion)
        img_labels = np_utils.to_categorical(img_labels)

        # Split data into training and validation sets
        X_train, X_valid, y_train, y_valid = train_test_split(img_array, img_labels,
                                                              shuffle=True, stratify=img_labels,
                                                              test_size=0.1, random_state=42)
        # free-up some RAM
        del df
        del img_array
        del img_labels

        # get image properties
        img_width = X_train.shape[1]
        img_height = X_train.shape[2]
        img_depth = X_train.shape[3]
        num_classes = y_train.shape[1]

        # normalize training images
        X_train = X_train / 255.
        X_valid = X_valid / 255.

        ##
        print("Preprocessing    -- end --")
        ##
        return img_width, img_height, img_depth, num_classes, X_train, X_valid, y_train, y_valid