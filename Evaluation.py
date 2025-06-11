# Plot the performance of the model for the different datasets

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

results = {
    "Unaltered": {
        'phenol': {'precision': 0.8875, 'recall': 0.663551, 'f1_score': 0.759358},
        'aldehyde': {'precision': 0.892857, 'recall': 0.746269, 'f1_score': 0.813008},
        'arene': {'precision': 0.928927, 'recall': 0.962445, 'f1_score': 0.945389},
        'macro': {'f1_score': 0.839252, 'EMR': 0.91076}
    },
    "SMOTE": {
        'phenol': {'precision': 0.787879, 'recall': 0.728972, 'f1_score': 0.757282},
        'aldehyde': {'precision': 0.923077, 'recall': 0.716418, 'f1_score': 0.806723},
        'arene': {'precision': 0.924275, 'recall': 0.961708, 'f1_score': 0.94262},
        'macro': {'f1_score': 0.835541, 'EMR': 0.90649}
    },
    "Augmented": {
        'phenol': {'precision': 0.868132, 'recall': 0.738318, 'f1_score': 0.79798},
        'aldehyde': {'precision': 0.980392, 'recall': 0.746269, 'f1_score': 0.847458},
        'arene': {'precision': 0.913194, 'recall': 0.968336, 'f1_score': 0.939957},
        'macro': {'f1_score': 0.861798, 'EMR': 0.906063}
    },
    "Augmented_Rare": {
        'phenol': {'precision': 0.886364, 'recall': 0.728972, 'f1_score': 0.8},
        'aldehyde': {'precision': 0.945455, 'recall': 0.776119, 'f1_score': 0.852459},
        'arene': {'precision': 0.897574, 'recall': 0.980854, 'f1_score': 0.937368},
        'macro': {'f1_score': 0.863276, 'EMR': 0.902647}
    }
}

classes = ['phenol', 'aldehyde', 'arene']
metrics = ['precision', 'recall', 'f1_score']
datasets = list(results.keys())

data = {metric: {cls: [] for cls in classes} for metric in metrics}
macro_f1 = []

for ds in datasets:
    for metric in metrics:
        for cls in classes:
            data[metric][cls].append(results[ds][cls][metric])
    macro_f1.append(results[ds]['macro']['f1_score'])

fig, axs = plt.subplots(2, 2, figsize=(14, 10))
axs = axs.flatten()

for i, metric in enumerate(metrics):
    ax = axs[i]
    for cls in classes:
        ax.plot(datasets, data[metric][cls], marker='o', label=cls)
    ax.set_title(f'{metric.capitalize()} per class')
    ax.set_ylim(0, 1)
    ax.set_ylabel(metric.capitalize())
    ax.grid(True)
    ax.legend()
    ax.set_xticklabels(datasets, rotation=20)

# Macro F1 score plot
axs[3].plot(datasets, macro_f1, marker='o', color='black')
axs[3].set_title('Macro F1-score')
axs[3].set_ylim(0, 1)
axs[3].set_ylabel('Macro F1-score')
axs[3].grid(True)
axs[3].set_xticklabels(datasets, rotation=20)

plt.tight_layout()
plt.show()
