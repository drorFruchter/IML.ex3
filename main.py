from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, TensorDataset

import models
from helpers import *
from models import *
import numpy as np
import matplotlib.pyplot as plt

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

def f(point):
    x,y = point
    return (x - 3) ** 2 + (y - 5) ** 2

def grad_f(point):
    x,y = point
    grad_x = 2 * (x - 3)
    grad_y = 2 * (y - 5)
    return np.array([grad_x, grad_y])

def Gradient_Descent(f, learning_rate, iterations):
    point = np.array([0, 0])
    history = np.array(point)
    for _ in range(iterations):
        point = point - grad_f(point) * learning_rate
        history = np.vstack((history, point))

    plt.figure(figsize=(10, 8))
    plt.scatter(history[:, 0], history[:, 1],
                c=range(iterations + 1), cmap='viridis',
                alpha=1, s=20)
    plt.colorbar(label='Iteration')
    plt.plot(history[:, 0], history[:, 1], 'r-', alpha=0.3)
    plt.plot(3, 5, 'r*', markersize=15, label='Global Minimum (3,5)')
    plt.plot(0, 0, 'go', label='Start (0,0)')
    plt.plot(history[-1, 0], history[-1, 1], 'bo', label='End Point')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Gradient Descent Path')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

def compute_accuracy(model, device, criterion, loader):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total, running_loss / len(loader)


def prepare_data_for_logistic_regression(train_X, train_Y, validation_X, validation_Y, test_X, test_Y, batch_size):
    train_X, train_Y = torch.FloatTensor(train_X), torch.FloatTensor(train_Y)
    validation_X, validation_Y = torch.FloatTensor(validation_X), torch.FloatTensor(validation_Y)
    test_X, test_Y = torch.FloatTensor(test_X), torch.FloatTensor(test_Y)

    train_loader = DataLoader(TensorDataset(train_X, train_Y), batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(TensorDataset(validation_X, validation_Y), batch_size=batch_size)
    test_loader = DataLoader(TensorDataset(test_X, test_Y), batch_size=batch_size)

    return train_loader, validation_loader, test_loader, (train_X, train_Y), (validation_X, validation_Y), (test_X, test_Y)

def train_model(model, train_loader, validation_loader, test_loader, criterion, optimizer, scheduler, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_losses, validation_losses, test_losses = [], [], []
    train_accuracies, validation_accuracies, test_accuracies = [], [], []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(outputs.data, 1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        if scheduler is not None:
            scheduler.step()

        train_acc = correct / total
        train_loss = running_loss / len(train_loader)
        validation_acc, validation_loss = compute_accuracy(model, device, criterion, validation_loader)
        test_acc, test_loss = compute_accuracy(model, device, criterion, test_loader)

        train_losses.append(train_loss)
        validation_losses.append(validation_loss)
        test_losses.append(test_loss)
        train_accuracies.append(train_acc)
        validation_accuracies.append(validation_acc)
        test_accuracies.append(test_acc)

        print(f'Epoch [{epoch + 1}/{num_epochs}]')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {validation_loss:.4f}, Val Acc: {validation_acc:.4f}')
        print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
        print('-' * 50)

    return {
        'train_losses': train_losses,
        'val_losses': validation_losses,
        'test_losses': test_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': validation_accuracies,
        'test_accuracies': test_accuracies
    }

def train_and_eval(X, Y, X_val, Y_val, X_test, Y_test, num_classes, num_epochs, learning_rate, decay_lr=False)
    train_loader, validation_loader, test_loader, (train_X, train_Y), (validation_X, validation_Y), (test_X, test_Y) = (
        prepare_data_for_logistic_regression(X, T, X_val, Y_val, X_test, Y_test, batch_size=32))

    model = models.Logistic_Regression(input_dim=2, output_dim=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    schedular = None
    if decay_lr:
        schedular = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.3)
    metrics = train_model(model, train_loader, validation_loader, test_loader, criterion, optimizer, schedular, num_epochs)
    return model, metrics


def logistic_regression(X, Y, X_val, Y_val, X_test, Y_test):
    def train_binary_model():
        learning_rates = [0.1, 0.01, 0.001]
        results = []

        for lr in learning_rates:
            model, metrics = train_and_eval(
                X=X, Y=Y,
                X_val=X_val,
                Y_val=Y_val,
                X_test=X_test, Y_test=Y_test,
                learning_rate=lr,
                num_epochs=10,
                num_classes=2
            )
            results.append((lr, model, metrics))

        # Find best model based on validation accuracy
        best_lr, best_model, best_metrics = max(results,
                                                key=lambda x: max(x[2]['val_accuracies']))

        return best_lr, best_model, best_metrics, results

    def train_multiclass_models():
        learning_rates = [0.01, 0.001, 0.0003]
        results = []

        for lr in learning_rates:
            model, metrics = train_and_evaluate(
                learning_rate=lr,
                num_epochs=30,
                train_file='train_multiclass.csv',
                val_file='validation_multiclass.csv',
                test_file='test_multiclass.csv',
                num_classes=3,
                decay_lr=True
            )
            results.append((lr, model, metrics))

        # Find best model based on validation accuracy
        best_lr, best_model, best_metrics = max(results,
                                                key=lambda x: max(x[2]['val_accuracies']))

        return best_lr, best_model, best_metrics, results

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

