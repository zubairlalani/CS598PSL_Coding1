# import pandas as pd
from utils import preprocess_data, convert_to_binary
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
def main():
    # print("Hello World")
    # Setting up paths to train/test data
    DATAFOLDER = "pen+based+recognition+of+handwritten+digits/"
    TRAIN_FILENAME = "pendigits.tra"
    TEST_FILENAME = "pendigits.tes"
    train_path = DATAFOLDER + TRAIN_FILENAME
    test_path = DATAFOLDER + TEST_FILENAME


    # Reading in the data
    train_data = np.loadtxt(train_path, delimiter=",")
    test_data = np.loadtxt(test_path, delimiter=",")
    # print(train_data.shape)
    # print(train_data[:5])   # first 5 rows

    X_train = train_data[:, 0:16]
    y_train = train_data[:, 16]

    X_test = test_data[:, 0:16]
    y_test = test_data[:, 16]

    print("Training data shapes: ")
    print(X_train.shape)
    print(y_train.shape)

    print("Testing data shapes: ")
    print(X_test.shape)
    print(y_test.shape)

    
    X_train, y_train = preprocess_data(X_train, y_train, [0, 2])
    X_test, y_test = preprocess_data(X_test, y_test, [0, 2])

    print("Preprocessed Training data shapes: ")
    print(X_train.shape)
    print(y_train.shape)

    print("Preprocessed Testing data shapes: ")
    print(X_test.shape)
    print(y_test.shape)

    reg = LinearRegression().fit(X_train, y_train)
    res = reg.score(X_train, y_train)
    print("Linear Regression Score: ", res)

    predicted_labels = reg.predict(X_test)
    print(predicted_labels)
    y_res = convert_to_binary(predicted_labels)
    print(y_res)
    print("Unique y labels: ", np.unique(y_res)) # confirming that we only have either 0 or 2

    # neigh = KNeighborsClassifier(n_neighbors=3)
    # neigh.fit(X_train, y_train)
    


    
if __name__ == "__main__":
    main()