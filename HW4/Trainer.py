import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import math


from Loss import compute_CCE_loss, compute_focal_loss
from Model import *
from Utils import random_mini_batches, predict


def trainer(model, config, x_train_pca, y_train, x_val_pca, y_val):
    losses = []
    for i in range(config.num_iterations):
        for x_batch, y_batch in random_mini_batches(x_train_pca, y_train):
            AL = model.forward(x_batch)
            dA_prev = model.backward(AL, y_batch)
            model.update(config.learning_rate)
        if config.loss_function == "cross_entropy":
            loss = compute_CCE_loss(model.forward(x_val_pca), y_val)
        else:
            loss = compute_focal_loss(
                model.forward(x_val_pca), y_val, config.alpha, config.gamma
            )
        losses.append(loss)
        if not i % config.print_freq:
            print("Loss after iteration", i, ":", loss)
            predict(x_val_pca, y_val, model)
            print()

    # plot the loss
    plt.figure(figsize=(4, 2))
    plt.plot(np.squeeze(losses))
    plt.ylabel("loss")
    plt.xlabel("iterations (per hundreds)")
    plt.title("Learning rate =" + str(config.learning_rate))
    plt.show()
    ### END CODE HERE ###
