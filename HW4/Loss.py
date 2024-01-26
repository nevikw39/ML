import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import math


# GRADED FUNCTION: compute_CCE_loss


def compute_CCE_loss(AL, Y):
    """
    Implement the categorical cross-entropy loss function using the above formula.

    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (n, C)
    Y -- true "label" vector (one hot vector, for example: [1,0,0] represents rock, [0,1,0] represents paper, [0,0,1] represents scissors
                                      in a Rock-Paper-Scissors, shape: (n, C)

    Returns:
    loss -- categorical cross-entropy loss
    """

    n = Y.shape[0]

    # Compute loss from aL and y.
    ### START CODE HERE ### (≈ 1 line of code)
    loss = (-1 / n) * (Y * np.log(AL + 1e-5)).sum().sum()
    ### END CODE HERE ###

    # To make sure your loss's shape is what we expect (e.g. this turns [[17]] into 17).
    loss = np.squeeze(loss)
    assert loss.shape == ()

    return loss


def compute_focal_loss(AL, Y, alpha, gamma):
    # Compute loss from aL and y.
    ### START CODE HERE ### (10 line of code)
    loss = (-1 / Y.shape[0]) * (
        np.take(alpha, Y.argmax(axis=1))
        * np.power(1 - (pt := np.max(AL * Y, axis=1)), gamma)
        * np.log(pt + 1e-5)
    ).sum()
    ### END CODE HERE ###

    loss = np.squeeze(loss)
    assert loss.shape == ()

    return loss
