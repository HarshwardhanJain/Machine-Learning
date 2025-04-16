# som_iris_with_accuracy.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom
from collections import defaultdict
from scipy.stats import mode

# Step 1: Load and scale the dataset
iris = load_iris()
X = iris.data
y = iris.target

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Step 2: Initialize and train SOM
som_grid_rows, som_grid_cols = 7, 7
som = MiniSom(x=som_grid_rows, y=som_grid_cols, input_len=X.shape[1], sigma=1.0, learning_rate=0.5)
som.random_weights_init(X_scaled)
som.train_random(data=X_scaled, num_iteration=1000)

# Step 3: Map each sample to its winning neuron
winner_neurons = [tuple(som.winner(x)) for x in X_scaled]

# Step 4: Group labels per neuron
neuron_to_labels = defaultdict(list)
for idx, neuron in enumerate(winner_neurons):
    neuron_to_labels[neuron].append(y[idx])

# Step 5: Assign majority label to each neuron (safe for all scipy versions)
neuron_majority_label = {}
for k, v in neuron_to_labels.items():
    result = mode(v, keepdims=True)  # ensures it's always indexable
    neuron_majority_label[k] = result.mode[0]

# Step 6: Predict labels based on neuron's majority vote
y_pred = np.array([neuron_majority_label[neuron] for neuron in winner_neurons])

# Step 7: Calculate accuracy
accuracy = np.mean(y_pred == y)
print(f"SOM Classification Accuracy (approximate): {accuracy * 100:.2f}%")

# Step 8: Visualize the SOM clustering
plt.figure(figsize=(10, 7))
colors = ['r', 'g', 'b']
markers = ['o', 's', 'D']

for i, x in enumerate(X_scaled):
    w = som.winner(x)
    plt.plot(w[0] + 0.5, w[1] + 0.5, markers[y[i]], markerfacecolor='None',
             markeredgecolor=colors[y[i]], markersize=12, markeredgewidth=2)

plt.title('SOM on Iris Dataset with Class Visualization')
plt.grid()
plt.show()
