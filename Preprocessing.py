import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
tqdm().pandas()

import tensorflow as tf


from keras.utils import np_utils

class Preprocessing():

    def __init__(self, df):
        self.df = df

    def preprocessing(self):
        df = self.df

        # convert image pix from list to matrix
        img_array = df.pixels.progress_apply(lambda x: np.array(x.split(' ')).reshape(48, 48, 1).astype('float32'))
        img_array = np.stack(img_array, axis=0)

        le = LabelEncoder()
        img_labels = le.fit_transform(df.emotion)       # Get emotion nbr for each row
        # np.savetxt('text.txt', img_labels.astype(int))
        img_labels = np_utils.to_categorical(img_labels)        # Transform to categorical data [0,0,1,0,0] = 3rd em.
        # np.savetxt('text_1.txt', img_labels.astype(int))
        le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))     #map emotion in a dic

        X_train, X_valid, y_train, y_valid = train_test_split(img_array, img_labels,
                                                              shuffle=True, stratify=img_labels,
                                                              test_size=0.1, random_state=42)
        print("X training data: ", X_train.shape, "\n",
              "X validation data: ", X_valid.shape, "\n",
              "y training data: ", y_train.shape, "\n",
              "y validation data: ", y_valid.shape)

        return X_train, X_valid, y_train, y_valid


