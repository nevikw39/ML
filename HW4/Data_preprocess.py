# Data Preprocess
#
# > *  Please do not modify this file


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import math


# load data
def load_data(path):
    data = np.load(path)
    X_train = data["x_train"]
    Y_train = data["y_train"]
    X_test = data["x_test"]

    # summarize loaded dataset
    print(f"Train: X={X_train.shape}, Y={Y_train.shape}")
    print(f"Test: X={X_test.shape}")
    # plot first few images
    for i in range(9):
        # define subplot
        plt.subplot(330 + 1 + i)
        # plot raw pixel data
        plt.imshow(X_train[i], cmap="gray", vmin=0, vmax=255)
    # show the figure
    plt.show()

    # GRADED CODE: multi-class classification (Data preprocessing)
    ### START CODE HERE ###
    def one_hot_encoding(label, num_of_classes=10):
        one_hot_label = np.zeros(num_of_classes)
        one_hot_label[int(label)] = 1
        return one_hot_label

    Y_train = np.array([one_hot_encoding(label) for label in Y_train])
    ### END CODE HERE ###

    print("shape of X_train: " + str(X_train.shape))
    print("shape of Y_train: " + str(Y_train.shape))
    print("shape of X_test: " + str(X_test.shape))

    # GRADED CODE: multi-class classification (Data preprocessing)
    ### START CODE HERE ###
    X_train = X_train.reshape(X_train.shape[0], -1) / 255.0
    X_test = X_test.reshape(X_test.shape[0], -1) / 255.0
    ### END CODE HERE ###

    print("\nshape of X_train: " + str(X_train.shape))
    print("shape of Y_train: " + str(Y_train.shape))
    print("shape of X_test: " + str(X_test.shape))

    return X_train, Y_train, X_test


def data_preprocess(X_train, Y_train):
    np.random.seed(1)
    permutation = list(np.random.permutation(X_train.shape[0]))
    X_train = X_train[permutation, :]
    Y_train = Y_train[permutation, :]

    n = X_train.shape[0]
    x_train = X_train[: int(n * 0.8), :]
    y_train = Y_train[: int(n * 0.8), :]
    x_val = X_train[int(n * 0.8) :, :]
    y_val = Y_train[int(n * 0.8) :, :]

    print("shape of x_train: " + str(x_train.shape))
    print("shape of y_train: " + str(y_train.shape))
    label_cnt = np.sum(y_train, axis=0)
    print(
        f"training data - percentage {[round(n / np.sum(np.array(label_cnt)), 2) for n in label_cnt]}"
    )
    if x_val is not None and y_val is not None:
        print("shape of x_val: " + str(x_val.shape))
        print("shape of y_val: " + str(y_val.shape))
        label_cnt = np.sum(y_val, axis=0)
        print(
            f"validation data - percentage {[round(n / np.sum(np.array(label_cnt)), 2) for n in label_cnt]}"
        )

    if x_val is not None and y_val is not None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))  # 1 row, 2 columns
        width = 0.4
        axes[0].bar(np.arange(10), np.sum(y_train, axis=0), width, color="b")
        axes[0].set_title("train")
        axes[0].set_xlabel("label")
        axes[0].set_ylabel("count")

        axes[1].bar(np.arange(10), np.sum(y_val, axis=0), width, color="b")
        axes[1].set_title("val")
        axes[0].set_xlabel("label")

        plt.tight_layout()
        plt.show()
    else:
        plt.bar(np.arange(10), np.sum(y_train, axis=0))
        plt.title("train")
        plt.xlabel("label")
        plt.ylabel("count")
        plt.show()

    return x_train, y_train, x_val, y_val
