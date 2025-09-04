import numpy as np
import matplotlib.pyplot as plt

def preprocess_data(X, y, filter_digits):
    mask = (y == filter_digits[0]) | (y == filter_digits[1])
    Xout = X[mask]
    yout = y[mask]
    return Xout, yout

def convert_to_binary(y, threshold=1):
    binarized_data = np.where(y >= threshold, 2, 0)
    return binarized_data

def misclassification_err(y_true, y_pred):
    return np.mean(y_true != y_pred)

def create_plot(neighbor_size, train_err_knn, test_err_knn, 
n_train_total, lr_train_err, lr_test_err, df_lr=3, k_opt=None, title="kNN vs Linear Regression"):
    m = len(neighbor_size)
    x_positions = np.arange(1, m + 1)
    df_vals = np.round(n_train_total / neighbor_size).astype(int)
    idx_lr = int(np.argmin(np.abs(df_vals - df_lr)))
    x_lr = x_positions[idx_lr]

    y_vals = np.concatenate([train_err_knn, test_err_knn])
    ymin = float(np.min(y_vals)) - 0.01
    ymax = float(np.max(y_vals)) + 0.01

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.set_xlim(0.5, m + 0.5)
    ax.set_ylim(ymin, ymax)

    ax.set_xlabel("Degrees of Freedom")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(df_vals)
    ax.set_ylabel("Misclassification rate")

    ax2 = ax.secondary_xaxis('top')
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels(neighbor_size)
    ax2.set_xlabel("k (number of neighbors)")

    ax.plot(x_positions, test_err_knn,  marker="o", linestyle="-",  color="magenta", label="kNN Test Error")
    ax.plot(x_positions, train_err_knn, marker="o", linestyle="--", color="blue",    label="kNN Train Error")

    ax.grid(True, alpha=0.3)
    ax.set_title(title)
    ax.legend(loc="lower left", fontsize=9)
    plt.tight_layout()

    plt.tight_layout(rect=(0, 0.08, 1, 1))  # leave ~8% at bottom
    fig.text(
        0.5, 0.02,
        f"Linear Regression: train error = {lr_train_err:.4f}, test error = {lr_test_err:.4f}",
        ha="center", va="bottom", fontsize=10
    )
    plt.show()
