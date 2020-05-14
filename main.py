from Load_data import Load_data
from Preprocessing import Preprocessing
from Net import Net

import pandas as pd

if __name__ == "__main__":

    fer_input_path = "./fer2013/fer2013.csv"#

    ########
    # Load the data to have a look at it (from CSV to workable images)
    ########
    # load = Load_data(fer_input_path)
    # df = load.load_data()

    ########
    # Prepare data to be used as input to CNN
    ########
    df = pd.read_csv(fer_input_path)#
    pre = Preprocessing(df)
    X_train, X_valid, y_train, y_valid = pre.preprocessing()

    ########
    # Create, train and run the CNN
    ########
    nw = Net(X_train, X_valid, y_train, y_valid)
    nw.runnetwork()

