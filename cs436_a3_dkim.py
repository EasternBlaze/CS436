import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
from torchvision import transforms
from sklearn.metrics import confusion_matrix
import seaborn as sns


# Line from pytorch tutorial, Section "Training on GPU" 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Model and training configurations
NUM_EPOCHS = 20
BATCH_SIZE = 128
LEARNING_RATE = 0.001
K_FOLDS = 5

# Class names taken from pytorch tutorial, Section 1 
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#  2. Data Loading and Preprocessing 
print("CIFAR-10 Dataset Load")
# transform taken from pytorch tutorial and MNIST notebook
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR-10 train and test datasets 
train_dataset_full = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
print(" Dataset Loaded ")


# MLP and CNN
# Took MLP structure from MNIST notebook
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(32 * 32 * 3, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )
    def forward(self, x):
        return self.layers(self.flatten(x))

# Took CNN structure from'LeNet5' class in MNIST
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, padding=2), 
            nn.ReLU(), 
            nn.MaxPool2d(2, 2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2), 
            nn.ReLU(), 
            nn.MaxPool2d(2, 2))
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 8 * 8, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 120)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(120, 10)

    def forward(self, x):
        # Pass input through convolutional layers
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.flatten(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out) # Final output layer
        return out

# c) Deeper CNN for Hypothesis 1
class DeeperCNN(nn.Module):
    def __init__(self):
        super(DeeperCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2))
        self.fc_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 1024), nn.ReLU(),
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, 10))
    def forward(self, x):
        return self.fc_block(self.layer2(self.layer1(x)))

# d) CNN with Dropout for Hypothesis 3
class CNNWithDropout(nn.Module):
    def __init__(self):
        super(CNNWithDropout, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.fc_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 256), nn.ReLU(),
            nn.Dropout(0.5), # Added Dropout
            nn.Linear(256, 120), nn.ReLU(),
            nn.Dropout(0.5), # Added Dropout
            nn.Linear(120, 10))
    def forward(self, x):
        return self.fc_block(self.layer2(self.layer1(x)))

# Helper functions
def get_model_size(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def plot_confusion_matrix(y_true, y_pred, class_names, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.show()

# This evaluation function is similar in purpose to the test functions in both
# pytorch  and 'Notebook_MNIST_Classification_LeNet_ResNet.ipynb'.
def evaluate_on_test_set(model, test_loader, device):
    model.eval()
    all_preds, all_labels = [], []
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return 100 * correct / total, all_labels, all_preds

# PART 1: MLP vs CNN with K-Fold Cross-Validation
# 'get_k_fold_data' function in kfold notebook.
def get_k_fold_data(k, i, dataset):
    fold_size = len(dataset) // k
    val_indices = list(range(i * fold_size, (i + 1) * fold_size))
    train_indices = list(range(0, i * fold_size)) + list(range((i + 1) * fold_size, len(dataset)))
    return Subset(dataset, train_indices), Subset(dataset, val_indices)

def k_fold_train(model_class, dataset, device):
    total_train_time = 0
    fold_train_accuracies = []
    
    print(f"\n Starting {K_FOLDS}-Fold Cross-Validation for {model_class.__name__} ")
    
    final_model = model_class().to(device)

    for i in range(K_FOLDS):
        train_subset, _ = get_k_fold_data(K_FOLDS, i, dataset)
        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
        
        net = model_class().to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()
        fold_start_time = time.time()
        
        for epoch in range(NUM_EPOCHS):
            net.train()
            correct, total = 0, 0
            for X, y in train_loader:
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()
                loss = criterion(net(X), y)
                loss.backward() 
                optimizer.step()
                
                if epoch == NUM_EPOCHS - 1:
                    _, predicted = torch.max(net(X).data, 1)
                    total += y.size(0)
                    correct += (predicted == y).sum().item()

        fold_train_time = time.time() - fold_start_time
        total_train_time += fold_train_time
        
        lastAcc = 100 * correct / total
        fold_train_accuracies.append(lastAcc)
        
        print(f'Fold {i + 1}, Train Time: {fold_train_time:.2f}s, Last Epoch training Accuracy: {lastAcc:.2f}%')
        
        if i == K_FOLDS - 1:
            final_model.load_state_dict(net.state_dict())

    return final_model, np.mean(fold_train_accuracies), total_train_time

# helper fuction to print results
def run_part1():
    print("\nPart 1")
    print(f"Using device: {device}")
    
    # MLP
    mlp_model, mlp_avg_train_acc, mlp_total_time = k_fold_train(MLP, train_dataset_full, device)
    mlp_model_size = get_model_size(mlp_model)
    mlp_test_acc, mlp_labels, mlp_preds = evaluate_on_test_set(mlp_model, test_loader, device)

    print("\n MLP Results")
    print(f"Total Training Time ({K_FOLDS} Folds): {mlp_total_time:.2f} seconds")
    print(f"Model Size (Parameters): {mlp_model_size:,}")
    print(f"Average Training Accuracy: {mlp_avg_train_acc:.2f}%")
    print(f"Test Accuracy: {mlp_test_acc:.2f}%")
    plot_confusion_matrix(mlp_labels, mlp_preds, classes, "MLP Confusion Matrix")

    # CNN
    cnn_model, cnn_avg_train_acc, cnn_total_time = k_fold_train(CNN, train_dataset_full, device)
    cnn_model_size = get_model_size(cnn_model)
    cnn_test_acc, cnn_labels, cnn_preds = evaluate_on_test_set(cnn_model, test_loader, device)

    print("\n CNN Results")
    print(f"Total Training Time ({K_FOLDS} Folds): {cnn_total_time:.2f} seconds")
    print(f"Model Size (Parameters): {cnn_model_size:,}")
    print(f"Average Training Accuracy: {cnn_avg_train_acc:.2f}%")
    print(f"Test Accuracy: {cnn_test_acc:.2f}%")
    plot_confusion_matrix(cnn_labels, cnn_preds, classes, "CNN Confusion Matrix")

    print("\n Part 1 results ")
    print(f" {'Model':<10}  {'Avg Fold Time (s)':<20}  {'Model Size':<15}  {'Avg Train Acc (%)':<20}  {'Test Acc (%)':<15} ")
    print(f" {'MLP':<10}  {mlp_total_time/K_FOLDS:<20.2f}  {mlp_model_size:<15,}  {mlp_avg_train_acc:<20.2f}  {mlp_test_acc:<15.2f} ")
    print(f" {'CNN':<10}  {cnn_total_time/K_FOLDS:<20.2f}  {cnn_model_size:<15,}  {cnn_avg_train_acc:<20.2f}  {cnn_test_acc:<15.2f} ")
    
    return cnn_test_acc

# Part 2 

def train_and_evaluate_hypothesis(model, train_loader, test_loader, device):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
            
    test_acc, _, _ = evaluate_on_test_set(model, test_loader, device)
    return test_acc

# helper fuction to print results
def run_part2(baseline_cnn_acc):
    print("\npart 2")
    
    train_indices = list(range(int(0.8 * len(train_dataset_full))))
    train_subset_hyp = Subset(train_dataset_full, train_indices)
    
    #  Hypothesis 1: Increase Model Complexity 
    print("\n Hypothesis 1: increase Model Complexity ")
    hyp1_loader = DataLoader(train_subset_hyp, batch_size=BATCH_SIZE, shuffle=True)
    deeper_cnn_model = DeeperCNN()
    h1_acc = train_and_evaluate_hypothesis(deeper_cnn_model, hyp1_loader, test_loader, device)
    print(f"Test accuracy changed from {baseline_cnn_acc:.2f}% to {h1_acc:.2f}%")

    #  Hypothesis 2: Decrease Batch Size 
    print("\n Hypothesis 2: decrease Batch Size ")
    hyp2_loader = DataLoader(train_subset_hyp, batch_size=32, shuffle=True) # Smaller batch size
    cnn_model_h2 = CNN()
    h2_acc = train_and_evaluate_hypothesis(cnn_model_h2, hyp2_loader, test_loader, device)
    print(f"Test accuracy changed from {baseline_cnn_acc:.2f}% to {h2_acc:.2f}%")
    
    #  Hypothesis 3: more regularization for better training (Dropout) 
    print("\n Hypothesis 3: add Regularization (Dropout) ")
    hyp3_loader = DataLoader(train_subset_hyp, batch_size=BATCH_SIZE, shuffle=True)
    dropout_cnn_model = CNNWithDropout()
    h3_acc = train_and_evaluate_hypothesis(dropout_cnn_model, hyp3_loader, test_loader, device)
    print(f"Test accuracy changed from {baseline_cnn_acc:.2f}% to {h3_acc:.2f}%")

    #  Summary Table 
    print("\n Hypothesis results: ")
    print(f" {'Hypothesis':<35}  {'Test Acc (%)':<20} ")
    print(f" {'Baseline CNN':<35}  {baseline_cnn_acc:<20.2f} ")
    print(f" {'1. Increase Model Complexity':<35}  {h1_acc:<20.2f} ")
    print(f" {'2. Decrease Batch Size (to 32)':<35}  {h2_acc:<20.2f} ")
    print(f" {'3. Add Dropout Regularization':<35}  {h3_acc:<20.2f} ")

#  Main Execution Block 
if __name__ == "__main__":
    baseline_acc = run_part1()
    run_part2(baseline_acc)