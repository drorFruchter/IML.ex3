import numpy as np
import matplotlib.pyplot as plt

def f(point):
    x,y = point
    return (x - 3) ** 2 + (y - 5) ** 2

def grad_f(point):
    x,y = point
    grad_x = 2 * (x - 3)
    grad_y = 2 * (y - 5)
    return np.array([grad_x, grad_y])

def plot_gradient_descent(history, iterations):
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

def Gradient_Descent(f, learning_rate, iterations):
    point = np.array([0, 0])
    history = np.array(point)
    for _ in range(iterations):
        point = point - grad_f(point) * learning_rate
        history = np.vstack((history, point))

    plot_gradient_descent(history, iterations)

