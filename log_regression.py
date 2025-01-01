import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import models
from helpers import *

def plot_losses_over_epochs(train_losses, validation_losses, test_losses):
    plt.plot(train_losses, label='Training Loss')
    plt.plot(validation_losses, label='Validation Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss vs. Epochs')
    plt.legend()
    plt.show()

def plot_accuracies_over_epochs(train_accuracies, validation_accuracies, test_accuracies):
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(validation_accuracies, label='Validation Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Epochs')
    plt.legend()
    plt.show()

def prepare_data_for_logistic_regression(train_X, train_Y, validation_X, validation_Y, test_X, test_Y, batch_size):
    train_X, train_Y = torch.FloatTensor(train_X), torch.LongTensor(train_Y)
    validation_X, validation_Y = torch.FloatTensor(validation_X), torch.LongTensor(validation_Y)
    test_X, test_Y = torch.FloatTensor(test_X), torch.LongTensor(test_Y)

    train_loader = DataLoader(TensorDataset(train_X, train_Y), batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(TensorDataset(validation_X, validation_Y), batch_size=batch_size)
    test_loader = DataLoader(TensorDataset(test_X, test_Y), batch_size=batch_size)

    return train_loader, validation_loader, test_loader

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
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
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
        'train_loss': train_losses[-1],
        'val_loss': validation_losses[-1],
        'test_loss': test_losses[-1],
        'train_accuracy': train_accuracies[-1],
        'val_accuracy': validation_accuracies[-1],
        'test_accuracy': test_accuracies[-1]
    }

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

def train_and_eval(X, Y, X_val, Y_val, X_test, Y_test, num_classes, num_epochs, learning_rate, decay_lr=False):
    train_loader, validation_loader, test_loader = prepare_data_for_logistic_regression(X, Y, X_val, Y_val, X_test, Y_test, batch_size=32)

    model = models.Logistic_Regression(input_dim=2, output_dim=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    schedular = None
    if decay_lr:
        schedular = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.3)
    metrics = train_model(model, train_loader, validation_loader, test_loader, criterion, optimizer, schedular, num_epochs)
    return model, metrics


def run_logistic_regression(X, Y, X_val, Y_val, X_test, Y_test, learning_rates, num_classes, num_epochs, decay_lr=False):
    results = []

    for lr in learning_rates:
        model, metrics = train_and_eval(
            X=X, Y=Y,
            X_val=X_val,
            Y_val=Y_val,
            X_test=X_test, Y_test=Y_test,
            learning_rate=lr,
            num_epochs=num_epochs,
            num_classes=num_classes,
            decay_lr=decay_lr
        )
        results.append((lr, model, metrics))

    # Find best model based on validation accuracy
    best_lr, best_model, best_metrics = max(results, key=lambda x: (x[2]['val_accuracy']))
    return best_lr, best_model, best_metrics, results

def train_decision_tree(X_train, Y_train, X_test, Y_test, max_depth):
    clf = DecisionTreeClassifier(max_depth=max_depth)
    clf.fit(X_train, Y_train)
    accuracy = clf.score(X_test, Y_test)
    plot_decision_boundaries(clf, X_test, Y_test,
                             title=f"Decision Tree (max_depth={max_depth}) Decision Boundaries")
    return accuracy

def logistic_regression(train_X, train_Y, validation_X, validation_Y, test_X, test_Y, learning_rates, num_classes, num_epochs, binary_case, decay_lr=False):
    best_lr, best_model, best_metrics, results = run_logistic_regression(train_X, train_Y, validation_X, validation_Y, test_X, test_Y, learning_rates, num_classes, num_epochs, decay_lr)
    print(best_lr, best_model, best_metrics ,sep="\n")

    if binary_case:
        # Q1:
        plot_decision_boundaries(best_model, test_X, test_Y, title=f"Best Model (LR={best_lr}) Decision Boundaries")

        # Q2:
        plot_losses_over_epochs(best_model.train_losses, best_model.validation_losses, best_model.test_losses)

        # Q3: Compare with Ridge Regression results (from Sec. 3.2)
        # Assuming Ridge_Classification has been run and best_ridge_accuracy is available
        print(f"Best Logistic Regression Accuracy: {best_metrics['test_accuracy']}")
        # print(f"Best Ridge Regression Accuracy: {best_ridge_accuracy}")
        # Explanation: Compare the accuracies and discuss which method performed better and why.

    else:
        # Q1: Plot test and validation accuracies vs. learning rate
        lrs = [result[0] for result in results]
        val_accs = [result[2]['val_accuracy'] for result in results]
        test_accs = [result[2]['test_accuracy'] for result in results]

        plt.plot(lrs, val_accs, label='Validation Accuracy')
        plt.plot(lrs, test_accs, label='Test Accuracy')
        plt.xlabel('Learning Rate')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs. Learning Rate')
        plt.legend()
        plt.show()

        # Q2: Plot losses and accuracies over epochs for the best model
        plot_losses_over_epochs(best_model.train_losses, best_model.validation_losses, best_model.test_losses)
        plot_accuracies_over_epochs(best_model.train_accuracies, best_model.validation_accuracies,
                                    best_model.test_accuracies)

        # Q3: Train and evaluate Decision Tree with max_depth=2
        dt_accuracy_2 = train_decision_tree(train_X, train_Y, test_X, test_Y, max_depth=2)
        print(f"Decision Tree (max_depth=2) Accuracy: {dt_accuracy_2}")

        # Q4: Train and evaluate Decision Tree with max_depth=10
        dt_accuracy_10 = train_decision_tree(train_X, train_Y, test_X, test_Y, max_depth=10)
        print(f"Decision Tree (max_depth=10) Accuracy: {dt_accuracy_10}")

        # Compare the models and explain which one is more suitable for the task.
