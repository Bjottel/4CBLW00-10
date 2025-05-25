import json
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


ORIGINAL_CSV = 'spectra.csv'
AUGMENTED_CSV = 'augmented_spectra.csv'
IND_AUGMENTED_CSV = 'individual_augmented_spectra.csv'

label_map = ['phenol', 'aldehyde']
num_classes = len(label_map)

def labels_to_multi_hot_vector(labels):
    multi_hot_vector = []
    for label in label_map:
        multi_hot_vector.append(1 if label in labels else 0)
    return torch.tensor(multi_hot_vector, dtype=torch.float32)

class IRDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.df['spectrum'] = self.df['spectrum'].apply(json.loads)
        self.df['labels'] = self.df['labels'].apply(json.loads)
        self.df['labels'] = self.df['labels'].apply(labels_to_multi_hot_vector)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        spectrum = torch.tensor(self.df['spectrum'].iloc[idx], dtype=torch.float32)
        labels = self.df['labels'].iloc[idx]
        return spectrum, labels

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(3600, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.linear_relu_stack(x)

def evaluate_model(model, data_loader):
    model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for batch_inputs, batch_labels in data_loader:
            outputs = model(batch_inputs)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            all_labels.append(batch_labels)
            all_predictions.append(preds)

    y_true = torch.cat(all_labels).cpu().numpy()
    y_pred = torch.cat(all_predictions).cpu().numpy()

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_micro': f1_score(y_true, y_pred, average='micro'),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'precision_macro': precision_score(y_true, y_pred, average='macro'),
        'recall_macro': recall_score(y_true, y_pred, average='macro')
    }
    return metrics

def train_and_evaluate(csv_path, result_file):
    dataset = IRDataset(csv_path)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_set, test_set = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=64)

    device = 'cpu'
    model = NeuralNetwork().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters())

    num_epochs = 1
    for epoch in range(num_epochs):
        model.train()
        for batch_inputs, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

    # Evaluation
    metrics = evaluate_model(model, test_loader)

    print(f"Evaluation results for dataset: {csv_path}")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    with open(result_file, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    return metrics

def evaluate_individual_augmentations():
    df = pd.read_csv(IND_AUGMENTED_CSV)
    unique_methods = df['augmentation_type'].unique()
    
    method_metrics = {}
    
    for method in unique_methods:
        if method == 'none':
            continue
            
        print(f"\nEvaluating augmentation method: {method}")
        df_method = df[df['augmentation_type'] == method]
        
        temp_csv = f'temp_{method}.csv'
        df_method.to_csv(temp_csv, index=False)
        
        result_file = f'evaluation_{method}.json'
        metrics = train_and_evaluate(temp_csv, result_file)
        method_metrics[method] = metrics
        
    return method_metrics

def plot_metrics_comparison(metrics_dict, title="Augmentation Method Comparison"):
    labels = list(next(iter(metrics_dict.values())).keys())
    methods = list(metrics_dict.keys())

    x = np.arange(len(labels))
    width = 0.1

    plt.figure(figsize=(12, 7))
    for i, method in enumerate(methods):
        values = [metrics_dict[method][label] for label in labels]
        plt.bar(x + i*width, values, width=width, label=method)

    plt.xticks(x + width*(len(methods)-1)/2, labels, rotation=45)
    plt.ylabel("Score")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig("augmentation_methods_comparison.png")
    plt.show()


original_metrics = train_and_evaluate(ORIGINAL_CSV, 'evaluation_original.json')
augmented_metrics = train_and_evaluate(AUGMENTED_CSV, 'evaluation_augmented.json')
individual_metrics = evaluate_individual_augmentations()

all_metrics = {
    'original': original_metrics,
    'augmented_full': augmented_metrics,
    **individual_metrics
}

plot_metrics_comparison(all_metrics)
