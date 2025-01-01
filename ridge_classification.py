import numpy as np
from models import Ridge_Regression
import matplotlib.pyplot as plt
from helpers import *


lambda_values = [0., 2., 4., 6., 8., 10.]

def evaluate_model(model, X, Y):
    return np.mean(Y == model.predict(X))

def plot_accuracies(test_accuracies, train_accuracies, validation_accuracies):
    plt.plot(lambda_values, train_accuracies, label='Training Accuracy')
    plt.plot(lambda_values, validation_accuracies, label='Validation Accuracy')
    plt.plot(lambda_values, test_accuracies, label='Test Accuracy')
    plt.xlabel('Lambda (λ)')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Lambda')
    plt.legend()
    plt.show()

def Ridge_Classification(train_X, train_Y, validation_X, validation_Y, test_X, test_Y):
    train_accuracies, validation_accuracies, test_accuracies = [], [], []
    best_accuracy, best_lambd = 0, 0
    worst_accuracy, worst_lambd = float("inf"), 0.

    for lambd in lambda_values:
        print(f"\nRidge Regression with lambda = {lambd}...")
        model = Ridge_Regression(lambd=lambd)
        model.fit(train_X, train_Y)

        train_accuracies.append(evaluate_model(model, train_X, train_Y))
        validation_accuracies.append(evaluate_model(model, validation_X, validation_Y))
        test_accuracies.append(evaluate_model(model, test_X, test_Y))

        validation_accuracy = validation_accuracies[-1]
        if validation_accuracy > best_accuracy:
            best_accuracy, best_lambd = validation_accuracy, lambd
        if validation_accuracy < worst_accuracy:
            worst_accuracy, worst_lambd = validation_accuracy, lambd

        print(f"lambd={lambd}, accuracy={validation_accuracy}")

    plot_accuracies(test_accuracies, train_accuracies, validation_accuracies)

    for lambd in [best_lambd, worst_lambd]:
        model = Ridge_Regression(lambd=lambd)
        model.fit(train_X, train_Y)
        plot_decision_boundaries(model, test_X, test_Y, title=f"lambda={lambd} Decision Boundaries")
