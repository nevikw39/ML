# Model
#
# > * You can follow your work in HW3 to complete this file
# > * You can not modify the initialize_parameters in Model Class

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import math


class Dense:
    def __init__(self, n_x, n_y, seed=1):
        self.n_x = n_x
        self.n_y = n_y
        self.seed = seed
        self.initialize_parameters()

    def initialize_parameters(self):
        """
        Argument:
        self.n_x -- size of the input layer
        self.n_y -- size of the output layer
        self.parameters -- python dictionary containing your parameters:
                           W -- weight matrix of shape (n_x, n_y)
                           b -- bias vector of shape (1, n_y)
        """
        sd = np.sqrt(6.0 / (self.n_x + self.n_y))
        np.random.seed(self.seed)
        W = np.random.uniform(
            -sd, sd, (self.n_y, self.n_x)
        ).T  # the transpose here is just for the code to be compatible with the old codes
        b = np.zeros((1, self.n_y))

        assert W.shape == (self.n_x, self.n_y)
        assert b.shape == (1, self.n_y)

        self.parameters = {"W": W, "b": b}

    def forward(self, A):
        """
        Implement the linear part of a layer's forward propagation.

        Arguments:
        A -- activations from previous layer (or input data) with the shape (n, f^[l-1])
        self.cache -- a python tuple containing "A", "W" and "b" ; stored for computing the backward pass efficiently

        Returns:
        Z -- the input of the activation function, also called pre-activation parameter with the shape (n, f^[l])
        """

        ### START CODE HERE ### (≈ 2 line of code)
        Z = A @ self.parameters["W"] + self.parameters["b"]
        self.cache = (A, self.parameters["W"], self.parameters["b"])
        ### END CODE HERE ###

        assert Z.shape == (A.shape[0], self.parameters["W"].shape[1])

        return Z

    def backward(self, dZ):
        """
        Implement the linear portion of backward propagation for a single layer (layer l)

        Arguments:
        dZ -- Gradient of the loss with respect to the linear output (of current layer l), same shape as Z
        self.cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer
        self.dW -- Gradient of the loss with respect to W (current layer l), same shape as W
        self.db -- Gradient of the loss with respect to b (current layer l), same shape as b

        Returns:
        dA_prev -- Gradient of the loss with respect to the activation (of the previous layer l-1), same shape as A_prev

        """
        A_prev, W, b = self.cache
        m = A_prev.shape[0]

        ### START CODE HERE ### (≈ 3 lines of code)
        self.dW = (1 / m) * A_prev.T @ dZ
        self.db = (1 / m) * dZ.sum(axis=0, keepdims=True)
        dA_prev = dZ @ W.T
        ### END CODE HERE ###

        assert dA_prev.shape == A_prev.shape
        assert self.dW.shape == self.parameters["W"].shape
        assert self.db.shape == self.parameters["b"].shape

        return dA_prev

    def update(self, learning_rate):
        """
        Update parameters using gradient descent

        Arguments:
        learning rate -- step size
        """

        ### START CODE HERE ### (≈ 2 lines of code)
        self.parameters["W"] -= learning_rate * self.dW
        self.parameters["b"] -= learning_rate * self.db
        ### END CODE HERE ###


class Activation:
    def __init__(self, activation_function, loss_function, alpha=None, gamma=None):
        self.activation_function = activation_function
        self.loss_function = loss_function
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, Z):
        if self.activation_function == "sigmoid":
            """
            Implements the sigmoid activation in numpy

            Arguments:
            Z -- numpy array of any shape
            self.cache -- stores Z as well, useful during backpropagation

            Returns:
            A -- output of sigmoid(z), same shape as Z
            """

            ### START CODE HERE ### (≈ 8 lines of code)
            A = (
                np.greater(Z, 0).astype(float)
                + np.less_equal(Z, 0).astype(float) * np.exp(Z)
            ) / (1 + np.exp(-np.abs(Z)))
            self.cache = Z
            ### END CODE HERE ###

            return A
        elif self.activation_function == "relu":
            """
            Implement the RELU function in numpy
            Arguments:
            Z -- numpy array of any shape
            self.cache -- stores Z as well, useful during backpropagation
            Returns:
            A -- output of relu(z), same shape as Z

            """

            ### START CODE HERE ### (≈ 2 lines of code)
            A = np.maximum(Z, 0)
            self.cache = Z
            ### END CODE HERE ###

            assert A.shape == Z.shape

            return A
        elif self.activation_function == "softmax":
            """
            Implements the softmax activation in numpy

            Arguments:
            Z -- np.array with shape (n, C)
            self.cache -- stores Z as well, useful during backpropagation

            Returns:
            A -- output of softmax(z), same shape as Z
            """

            ### START CODE HERE ### (≈ 3 lines of code)
            b = np.max(Z, axis=1).reshape(-1, 1)
            A = np.exp(Z - b) / np.exp(Z - b).sum(axis=1).reshape(-1, 1)
            self.cache = Z
            ### END CODE HERE ###

            return A
        else:
            assert (
                0
            ), f"you're using undefined activation function {self.activation_function}"

    def backward(self, dA=None, Y=None):
        if self.activation_function == "sigmoid":
            """
            Implement the backward propagation for a single SIGMOID unit.
            Arguments:
            dA -- post-activation gradient, of any shape
            self.cache -- 'Z' where we store for computing backward propagation efficiently
            Returns:
            dZ -- Gradient of the loss with respect to Z
            """

            ### START CODE HERE ### (≈ 9 lines of code)
            sigmoid = lambda x: (
                np.greater(x, 0).astype(float)
                + np.less_equal(x, 0).astype(float) * np.exp(x)
            ) / (1 + np.exp(-np.abs(x)))
            Z = self.cache
            dZ = dA * sigmoid(Z) * (1 - sigmoid(Z))
            ### END CODE HERE ###

            assert dZ.shape == Z.shape

            return dZ

        elif self.activation_function == "relu":
            """
            Implement the backward propagation for a single RELU unit.
            Arguments:
            dA -- post-activation gradient, of any shape
            self.cache -- 'Z' where we store for computing backward propagation efficiently
            Returns:
            dZ -- Gradient of the loss with respect to Z
            """

            ### START CODE HERE ### (≈ 3 lines of code)
            Z = self.cache
            dZ = dA * np.greater(Z, 0).astype(float)
            ### END CODE HERE ###

            assert dZ.shape == Z.shape

            return dZ

        elif (
            self.activation_function == "softmax"
            and self.loss_function == "cross_entropy"
        ):
            """
            Implement the backward propagation for a [SOFTMAX->CCE LOSS] unit.
            Arguments:
            Y -- true "label" vector (one hot vector, for example: [1,0,0] represents rock, [0,1,0] represents paper, [0,0,1] represents scissors
                                      in a Rock-Paper-Scissors, shape: (n, C)
            self.cache -- 'Z' where we store for computing backward propagation efficiently
            Returns:
            dZ -- Gradient of the cost with respect to Z
            """

            # GRADED FUNCTION: softmax_CCE_backward
            ### START CODE HERE ### (≈ 3 lines of code)
            Z = self.cache
            s = np.exp(Z - (b := np.max(Z, axis=1).reshape(-1, 1))) / np.exp(Z - b).sum(
                axis=1
            ).reshape(-1, 1)
            dZ = s - Y
            ### END CODE HERE ###

            assert dZ.shape == self.cache.shape

            return dZ
        elif (
            self.activation_function == "softmax" and self.loss_function == "focal_loss"
        ):
            """
            Implement the backward propagation for a [SOFTMAX->FOCAL LOSS] unit.
            Arguments:
            Y -- true "label" vector (one hot vector, for example: [1,0,0] represents rock, [0,1,0] represents paper, [0,0,1] represents scissors
                                      in a Rock-Paper-Scissors, shape: (n, C)
            self.cache -- 'Z' where we store for computing backward propagation efficiently
            Returns:
            dZ -- Gradient of the cost with respect to Z
            alpha -- weighting factors correspond to each class, shape: (C,)
            gamma -- modulating factor, a float
            """

            # FUNCTION: softmax_focalLoss_backward
            ## START CODE HERE ### (≈ 10 lines of code)
            Z = self.cache
            p = np.exp(Z - (b := np.max(Z, axis=1).reshape(-1, 1))) / np.exp(Z - b).sum(
                axis=1
            ).reshape(-1, 1)
            pt = np.full(p.shape, np.max(p * Y, axis=1).reshape(-1, 1))
            dZ = np.take(self.alpha, Y.argmax(axis=1)).reshape(-1, 1) * (
                self.gamma
                * np.power(1 - pt, self.gamma - 1)
                * np.log(pt + 1e-5)
                * ((Y * pt) - p * pt)
                - np.power(1 - pt, self.gamma) * (Y - p)
            )
            ## END CODE HERE ###

            assert dZ.shape == self.cache.shape

            return dZ


class Model:
    def __init__(self, config):
        self.config = config
        self.units = config.layers_dims
        self.activation_functions = config.activation_fn
        self.loss_function = config.loss_function
        self.alpha = config.alpha
        self.gamma = config.gamma
        self.initialize_parameters()
        self.check = True

    def initialize_parameters(self):
        """
        Arguments:
        self.units -- number of nodes/units for each layer, starting from the input dimension and ending with the output dimension (i.e., [4, 4, 1])
        self.activation_functions -- activation functions used in each layer (i.e, ["relu", "sigmoid"])
        self.loss_function -- ["cross_entropy", "focal_loss"]
        self.alpha -- weighting factors used by focal loss correspond to each class, shape: (C,)
        self.gamma -- a float, used by focal loss
        """
        self.linear = []  # a list to store the dense layers when initializing the model
        self.activation = (
            []
        )  # a list to store the activation function layers when initializing the model

        # FUNCTION: model_initialize_parameters
        ### DO NOT MODIFY THIS PART ###
        for i in range(len(self.units) - 1):
            dense = Dense(self.units[i], self.units[i + 1], i)
            self.linear.append(dense)

        for i in range(len(self.activation_functions)):
            self.activation.append(
                Activation(
                    self.activation_functions[i],
                    self.loss_function,
                    self.alpha,
                    self.gamma,
                )
            )

    def forward(self, X):
        """
        Arguments:
        X -- input data: shape (n, f)

        Returns:
        A -- output of L-layer neural network, probability vector corresponding to your label predictions, shape (n, C)
        """
        A = X

        # GRADED FUNCTION: model_forward
        ### START CODE HERE ### (≈ 4 lines of code)
        for l, a in zip(self.linear, self.activation):
            A = a.forward(l.forward(A))
        ### END CODE HERE ###

        return A

    def backward(self, AL=None, Y=None):
        """
        Arguments:
        For multi-class classification,
        AL -- output of L-layer neural network, probability vector corresponding to your label predictions, shape (n, C)
        Y -- true "label" vector (one hot vector, for example: [1,0,0] represents rock, [0,1,0] represents paper, [0,0,1] represents scissors
                                      in a Rock-Paper-Scissors, shape: (n, C)

        Returns:
        dA_prev -- post-activation gradient
        """

        L = len(self.linear)
        C = Y.shape[1]

        # assertions
        warning = "Warning: only the following 4 combinations are allowed! \n \
                    1. binary classification: sigmoid + cross_entropy) \n \
                    2. binary classification: softmax + focal_loss) \n \
                    3. multi-class classification: softmax + cross_entropy) \n \
                    4. multi-class classification: softmax + focal_loss)"
        assert self.loss_function in [
            "cross_entropy",
            "focal_loss",
        ], "you're using undefined loss function!"
        if Y.shape[1] <= 2:  # in binary classification
            if self.loss_function == "cross_entropy":
                assert self.activation_functions[-1] == "sigmoid", warning
                assert (
                    self.units[-1] == 1
                ), "you should set last dim to 1 when using sigmoid + cross_entropy in binary classification!"
            elif self.loss_function == "focal_loss":
                assert self.activation_functions[-1] == "softmax", warning
                assert (
                    self.units[-1] == 2
                ), "you should set last dim to 2 when using softmax + focal_loss in binary classification!"
        else:  # in multi-class classification
            assert self.activation_functions[-1] == "softmax", warning
            assert (
                self.units[-1] == Y.shape[1]
            ), f"you should set last dim to {Y.shape[1]}(the number of classes) in multi-class classification!"

        # FUNCTION: model_backward
        ### START CODE HERE ### (≈ 20 lines of code)

        if self.activation_functions[-1] == "sigmoid":
            if self.loss_function == "cross_entropy":
                # Initializing the backpropagation
                dAL = -np.divide(Y, AL + 1e-5) + np.divide(1 - Y, 1 - AL + 1e-5)

                # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "dAL". Outputs: "dA_prev"
                dZ = self.activation[-1].backward(dA=dAL)
                dA_prev = self.linear[-1].backward(dZ)
        elif self.activation_functions[-1] == "softmax":
            # Initializing the backpropagation
            dZ = self.activation[-1].backward(Y=Y)

            # Lth layer (LINEAR) gradients. Inputs: "dZ". Outputs: "dA_prev"
            dA_prev = self.linear[-1].backward(dZ)

        # Loop from l=L-2 to l=0
        # lth layer: (RELU -> LINEAR) gradients.
        # Inputs: "dA_prev". Outputs: "dA_prev"
        for a, l in zip(self.activation[-2::-1], self.linear[-2::-1]):
            dZ = a.backward(dA=dA_prev)
            dA_prev = l.backward(dZ)
        ### END CODE HERE ###

        return dA_prev

    def update(self, learning_rate):
        """
        Arguments:
        learning_rate -- step size
        """

        L = len(self.linear)

        # FUNCTION: model_update_parameters
        ### START CODE HERE ### (≈ 2 lines of code)
        for l in self.linear:
            l.update(learning_rate)
        ### END CODE HERE ###
