import torch
from ridge_classification import Ridge_Classification
from grdaient_descent import Gradient_Descent
from log_regression import logistic_regression
from helpers import *
import numpy as np


if __name__ == '__main__':
    np.random.seed(42)
    torch.manual_seed(42)

    train_X, train_Y, _ = read_data_demo("train.csv")
    validation_X, validation_Y, _ = read_data_demo("validation.csv")
    test_X, test_Y, _ = read_data_demo("test.csv")

    #Task 3
    # Ridge_Classification(train_X, train_Y, validation_X, validation_Y, test_X, test_Y)

    #Task 4
    # Gradient_Descent(f, learning_rate=0.1, iterations=1000)

    #Task 5
    print("running logistic_regression on binary case")
    logistic_regression(train_X, train_Y, validation_X, validation_Y, test_X, test_Y, [0.1, 0.01, 0.001], 2, 10, binary_case=True, decay_lr=False)

    #multi class data
    train_X, train_Y, _ = read_data_demo("train_multiclass.csv")
    validation_X, validation_Y, _ = read_data_demo("validation_multiclass.csv")
    test_X, test_Y, _ = read_data_demo("test_multiclass.csv")

    print("running logistic_regression on multiclass case")
    logistic_regression(train_X, train_Y, validation_X, validation_Y, test_X, test_Y, [0.01, 0.001, 0.0001], num_classes=5, num_epochs=30, binary_case=False, decay_lr=True)

