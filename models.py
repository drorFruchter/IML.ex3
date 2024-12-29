import numpy as np
import torch
from torch import nn


class Ridge_Regression:

    def __init__(self, lambd):
        self.lambd = lambd
        self.W = None
        self.bias = None

    def fit(self, X: np.ndarray, Y: np.ndarray):

        """
        Fit the ridge regression model to the provided data.
        :param X: The training features.
        :param Y: The training labels.
        """

        Y = 2 * (Y - 0.5) # transform the labels to -1 and 1, instead of 0 and 1.

        ########## YOUR CODE HERE ##########

        # compute the ridge regression weights using the formula from class / exercise.
        # you may not use np.linalg.solve, but you may use np.linalg.inv

        ####################################
        X = X.T

        N_train = X.shape[1]
        X_extended = np.vstack([X, np.ones((1, X.shape[1]))])

        XX_T = X_extended @ X_extended.T / N_train
        I = np.eye(X_extended.shape[0])

        inverse = np.linalg.inv(XX_T + self.lambd * I)
        XY_T = (X_extended @ Y) / N_train

        W_extended = inverse @ XY_T

        self.W = W_extended[:-1]
        self.bias = W_extended[-1]

    def predict(self, X):
        """
        Predict the output for the provided data.
        :param X: The data to predict. 
        :return: The predicted output. 
        """
        preds = None
        ########## YOUR CODE HERE ##########

        # compute the predicted output of the model.
        # name your predicitons array preds.

        ####################################

        X = X.T

        preds = self.W @ X + self.bias
        preds = np.where(preds >= 0, 1, -1)
        # transform the labels to 0s and 1s, instead of -1s and 1s.
        # You may remove this line if your code already outputs 0s and 1s.
        preds = (preds + 1) / 2

        return preds.flatten()



class Logistic_Regression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Logistic_Regression, self).__init__()

        ########## YOUR CODE HERE ##########

        # define a linear operation.

        ####################################
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        """
        Computes the output of the linear operator.
        :param x: The input to the linear operator.
        :return: The transformed input.
        """
        # compute the output of the linear operator

        ########## YOUR CODE HERE ##########

        # return the transformed input.
        # first perform the linear operation
        # should be a single line of code.

        ####################################

        return self.linear(x)

    def predict(self, x):
        """
        THIS FUNCTION IS NOT NEEDED FOR PYTORCH. JUST FOR OUR VISUALIZATION
        """
        x = torch.from_numpy(x).float().to(self.linear.weight.data.device)
        x = self.forward(x)
        x = nn.functional.softmax(x, dim=1)
        x = x.detach().cpu().numpy()
        x = np.argmax(x, axis=1)
        return x
