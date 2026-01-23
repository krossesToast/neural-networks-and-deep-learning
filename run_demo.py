import random
import numpy as np
import matplotlib.pyplot as plt

import src.mnist_loader as mnist_loader
import src.network as network

# Load data
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

# Smaller network + fewer epochs for live demos
net = network.Network([784, 30, 10])
net.SGD(
    training_data,
    epochs=5,            # 👈 fast
    mini_batch_size=10,
    eta=3.0,
    test_data=test_data
)

# --- Evaluation helpers ---

def predict(net, x):
    """Return predicted digit"""
    return np.argmax(net.feedforward(x))

def show_image(x, predicted, actual):
    plt.imshow(x.reshape(28, 28), cmap="gray")
    plt.title(f"Predicted: {predicted} | Actual: {actual}")
    plt.axis("off")
    plt.show()

# --- Show some classifications ---

print("Showing correct classifications:")
correct = []
for x, y in test_data:
    if predict(net, x) == y:
        correct.append((x, y))
    if len(correct) == 3:
        break

for x, y in correct:
    show_image(x, predict(net, x), y)

print("Showing WRONG classifications:")
wrong = []
for x, y in test_data:
    if predict(net, x) != y:
        wrong.append((x, y))
    if len(wrong) == 3:
        break

for x, y in wrong:
    show_image(x, predict(net, x), y)
