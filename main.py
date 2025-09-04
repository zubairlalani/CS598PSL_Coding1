# import pandas as pd
from utils import preprocess_data, convert_to_binary, create_plot, misclassification_err
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def main():
    FILTER_DIGITS = (0, 2) # Last 2 digits of Zubair's UIN
    K_RANGE = range(1, 21)
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
    n_samples = len(train_data)

    X_train = train_data[:, 0:16]
    y_train = train_data[:, 16]
    X_test = test_data[:, 0:16]
    y_test = test_data[:, 16]
    X_train, y_train = preprocess_data(X_train, y_train, FILTER_DIGITS)
    X_test, y_test = preprocess_data(X_test, y_test, FILTER_DIGITS)

    print(len(X_train))
    print(len(X_test))
    d0, d1 = FILTER_DIGITS
    threshold = 0.5 * (d0 + d1)  # midpoint between the two digits

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    yhat_train_cont = lr.predict(X_train)
    yhat_test_cont  = lr.predict(X_test)

    yhat_train_lr = np.where(yhat_train_cont >= threshold, d1, d0)
    yhat_test_lr  = np.where(yhat_test_cont  >= threshold, d1, d0)

    lr_train_err = misclassification_err(y_train, yhat_train_lr)
    lr_test_err  = misclassification_err(y_test,  yhat_test_lr)
    print(f"[Linear Regression] train error = {lr_train_err:.4f}, test error = {lr_test_err:.4f}")

    knn_train_errs = []
    knn_test_errs  = []

    for k in K_RANGE:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        yhat_train_knn = knn.predict(X_train)
        yhat_test_knn  = knn.predict(X_test)
        tr_err = misclassification_err(y_train, yhat_train_knn)
        te_err = misclassification_err(y_test,  yhat_test_knn)
        knn_train_errs.append(tr_err)
        knn_test_errs.append(te_err)
        print(f"k={k:2d} train err = {tr_err:.4f} | test err = {te_err:.4f}")

    # Find optimal k
    knn_test_errs = np.array(knn_test_errs)
    best_idx = int(np.argmin(knn_test_errs))
    best_k   = list(K_RANGE)[best_idx]
    best_err = float(knn_test_errs[best_idx])
    dof_best = len(X_train) / best_k

    print(f"\nOptimal k = {best_k} with test error = {best_err:.4f}")
    print(f"Degrees of freedom for k={best_k}: {len(X_train)}/{best_k} = {dof_best:.1f}")

    '''
    1) a) 
    '''
    ks = list(K_RANGE)

    n_train_total = len(X_train)

    create_plot(
        np.array(ks),
        knn_train_errs,
        knn_test_errs,
        n_train_total,
        lr_train_err,
        lr_test_err,
        df_lr=3,
        k_opt=best_k,
        title="kNN vs Linear Regression"
    )

    '''
    1) b) The plot does match our intuition of the bias-variance tradeoff. We can see that the training error 
    is very low as expected; however, once we generalize to unseen data we get the typical U shape curve as seen in lecture. The left
    side of the plot is the "most complex" model with most degrees of freedom so we have low bias but high variance.
    On the right side of the plot we have the "less complex" models which have higher bias and lower variance. Note that in our plot we plotted in order of increasing k value.
    The optimal k value is k = 2 with 780 degrees of freedom calculated by taking n=1560 and divding it by k=2. The training
    error for this value of k is about 0 whereas the testing error is 0.0069. 

    '''

    

if __name__ == "__main__":
    main()