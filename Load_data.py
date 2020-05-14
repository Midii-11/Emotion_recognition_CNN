import h5py
import pandas as pd
import cv2 as cv
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class Load_data():
    def __init__(self, fer_input_path):
        self.fer_input_path = fer_input_path

    def load_data(self):
        df = pd.read_csv(self.fer_input_path)
        emotion_txt = {0:'anger', 1:'disgust', 2:'fear',
                       3:'happiness', 4: 'sadness',
                       5: 'surprise', 6: 'neutral'}
        # print(sorted(df.emotion.value_counts()))

        sns.countplot(df.emotion)
        plt.show()

        fig = plt.figure(1, (14, 14))
        k = 0
        for label in sorted(df.emotion.unique()):
            for j in range(7):
                px = df[df.emotion == label].pixels.iloc[k]

                px = np.array(px.split(' ')).reshape(48, 48).astype('float32')

                k += 1
                ax = plt.subplot(7, 7, k)
                ax.imshow(px, cmap='gray')
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(emotion_txt[label])
                plt.tight_layout()

        plt.show()
        return df
