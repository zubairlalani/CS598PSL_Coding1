import numpy as np
# Filters data so to only keep the features/labels 
# corresponding to a label within the filter_digits
def preprocess_data(X, y, filter_digits):
    mask = (y == filter_digits[0]) | (y == filter_digits[1])
    Xout = X[mask]
    yout = y[mask]
    return Xout, yout

def convert_to_binary(y, threshold=1):
    binarized_data = np.where(y >= threshold, 2, 0)
    return binarized_data