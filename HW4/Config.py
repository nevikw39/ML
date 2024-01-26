# Config
#
# > * Please do not modify this file

import numpy as np


class Config:
    def __init__(self, layers_dims, loss_function):
        self.layers_dims = layers_dims  #  2-layer model
        self.activation_fn = ["relu", "softmax"]
        self.learning_rate = 1e-3
        self.num_iterations = 100
        self.batch_size = 128
        self.classes = 10  # keep track of loss
        self.print_loss = True
        self.print_freq = 100
        self.loss_function = loss_function
        self.gamma = 2.0
        self.alpha = np.array([(i + 1) ** 2 for i in range(self.classes)])
