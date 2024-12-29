import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as fn
from torchvision.datasets import CIFAR10, MNIST
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

class ClassificationANN(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3, hidden_dim4, hidden_dim5, output_dim):
        super(ClassificationANN, self).__init__()
        # Random Weights
        self.hidden_layer1 = nn.Parameter(torch.randn(input_dim, hidden_dim1, requires_grad=True))
        self.hidden_layer2 = nn.Parameter(torch.randn(hidden_dim1, hidden_dim2, requires_grad=True) )
        self.hidden_layer3 = nn.Parameter(torch.randn(hidden_dim2, hidden_dim3, requires_grad=True)) 
        self.hidden_layer4 = nn.Parameter(torch.randn(hidden_dim3, hidden_dim4, requires_grad=True)) 
        self.hidden_layer5 = nn.Parameter(torch.randn(hidden_dim4, hidden_dim5, requires_grad=True)) 
        self.output_layer = nn.Parameter(torch.randn(hidden_dim5, output_dim, requires_grad=True)) 
        # Random Biases
        self.bias1 = nn.Parameter(torch.randn(hidden_dim1))
        self.bias2 = nn.Parameter(torch.randn(hidden_dim2))
        self.bias3 = nn.Parameter(torch.randn(hidden_dim3))
        self.bias4 = nn.Parameter(torch.randn(hidden_dim4))
        self.bias5 = nn.Parameter(torch.randn(hidden_dim5))
        self.bias6 = nn.Parameter(torch.randn(output_dim))
        
    def forward(self, input):
        input = input.view(input.size(0), -1)
        output1 = torch.mm(input, self.hidden_layer1) + self.bias1
        output1 = fn.relu(output1)  
        output2 = torch.mm(output1, self.hidden_layer2) + self.bias2
        output2 = fn.relu(output2)
        output3 = torch.mm(output2, self.hidden_layer3) + self.bias3
        output3 = fn.relu(output3)
        output4 = torch.mm(output3, self.hidden_layer4) + self.bias4
        output4 = fn.relu(output4)
        output5 = torch.mm(output4, self.hidden_layer5) + self.bias5
        output5 = fn.relu(output5)
        final_output = torch.mm(output5, self.output_layer) + self.bias6
        return final_output

def train_model(model, train_loader, n_epochs, loss_fn, optimizer):
    for epoch in range(n_epochs):
        model.train()  # Set model to training mode
        epoch_loss = 0
        
        for images, labels in train_loader:
            outputs = model(images)  # Forward pass
            loss = loss_fn(outputs, labels)  # Compute loss

            optimizer.zero_grad()
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {avg_loss:.4f}")

# Instantiate the model
input_dim = 28*28
hidden_dim1 = 200
hidden_dim2 = 170
hidden_dim3 = 140
hidden_dim4 = 90
hidden_dim5 = 50
output_dim = 10

model = ClassificationANN(input_dim, hidden_dim1, hidden_dim2, hidden_dim3, hidden_dim4, hidden_dim5, output_dim)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

train_loader
n_epochs = 10
Batch_size = 50
train_model(model, train_loader, n_epochs, loss_fn, optimizer)

from sklearn.metrics import confusion_matrix, classification_report

# Function to evaluate the model
def evaluate_model(model, test_loader):
    model.eval()  # Set model to evaluation mode
    all_predictions = []
    all_labels = []

    with torch.no_grad():  # Disable gradient calculation
        for images, labels in test_loader:
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            all_predictions.extend(predictions.tolist())
            all_labels.extend(labels.tolist())

    # Compute metrics
    acc = accuracy_score(all_labels, all_predictions)
    cm = confusion_matrix(all_labels, all_predictions)
    report = classification_report(all_labels, all_predictions, target_names=[str(i) for i in range(10)])

    print(f"Accuracy: {acc * 100:.2f}%")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)
    return acc, cm, report

evaluate_model(model, test_loader)
