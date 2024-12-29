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
from sklearn.metrics import mean_squared_error, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the data
data = fetch_california_housing()
X, y = data.data, data.target
y = y.reshape(-1, 1)  # Ensure y has shape (n_samples, 1)

scaler = StandardScaler()

X = scaler.fit_transform(X)
y = scaler.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train_t = torch.FloatTensor(X_train)
X_test_t = torch.FloatTensor(X_test)
y_train_t = torch.FloatTensor(y_train)
y_test_t = torch.FloatTensor(y_test)

# Define the Regression ANN class (modified with 2 additional hidden layers)
class RegressionANN(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3, hidden_dim4):
        super(RegressionANN, self).__init__()

        # Proper weight initialization using Xavier
        self.hidden_layer1 = nn.Parameter(torch.randn(input_dim, hidden_dim1) * np.sqrt(2. / input_dim))
        self.hidden_layer2 = nn.Parameter(torch.randn(hidden_dim1, hidden_dim2) * np.sqrt(2. / hidden_dim1))
        self.hidden_layer3 = nn.Parameter(torch.randn(hidden_dim2, hidden_dim3) * np.sqrt(2. / hidden_dim2))
        self.hidden_layer4 = nn.Parameter(torch.randn(hidden_dim3, hidden_dim4) * np.sqrt(2. / hidden_dim3))
        self.output_layer = nn.Parameter(torch.randn(hidden_dim4, 1) * np.sqrt(2. / hidden_dim4))

        # Proper bias initialization
        self.bias1 = nn.Parameter(torch.zeros(hidden_dim1))
        self.bias2 = nn.Parameter(torch.zeros(hidden_dim2))
        self.bias3 = nn.Parameter(torch.zeros(hidden_dim3))
        self.bias4 = nn.Parameter(torch.zeros(hidden_dim4))
        self.bias5 = nn.Parameter(torch.zeros(1))

    def forward(self, input):
        output1 = torch.mm(input, self.hidden_layer1) + self.bias1
        output1 = torch.relu(output1)
        output2 = torch.mm(output1, self.hidden_layer2) + self.bias2
        output2 = torch.relu(output2)
        output3 = torch.mm(output2, self.hidden_layer3) + self.bias3
        output3 = torch.relu(output3)
        output4 = torch.mm(output3, self.hidden_layer4) + self.bias4
        output4 = torch.relu(output4)
        final_output = torch.mm(output4, self.output_layer) + self.bias5
        return final_output


# Training function
def train_model(model, train_data, n_epoch, batch_size, loss_fn, optimizer):
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    for epoch in range(n_epoch):
        model.train()
        epoch_loss = 0.0

        for x_batch, y_batch in train_loader:
            # Forward pass
            yhat = model.forward(x_batch)
            loss = loss_fn(yhat, y_batch)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{n_epoch}, Loss: {avg_loss:.4f}")
    return model

# Initialize the model
input_dim = X_train.shape[1]
hidden_dim1 = 4
hidden_dim2 = 4
hidden_dim3 = 4
hidden_dim4 = 4
model = RegressionANN(input_dim, hidden_dim1, hidden_dim2, hidden_dim3, hidden_dim4)

# Define loss function and optimizer
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)  # Lowered learning rate

# Prepare training data
train_data = TensorDataset(X_train_t, y_train_t)

# Train the model
n_epochs = 100
batch_size = 32
trained_model = train_model(model, train_data, n_epochs, batch_size, loss_fn, optimizer)

# Testing the model
with torch.no_grad():
    y_pred = model.forward(X_test_t)
    test_loss = loss_fn(y_pred, y_test_t)
    print(f"Test Loss: {test_loss.item():.4f}")

# Sample predictions
print("Predictions:", y_pred[:5].flatten())
print("Actual:", y_test_t[:5].flatten())

# Convert regression predictions to discrete categories (e.g., bins)
num_bins = 4  # Define the number of bins
bins = np.linspace(y_test.min(), y_test.max(), num_bins + 1)  # Create bins
y_test_binned = np.digitize(y_test, bins[:-1]) - 1  # Bin the true values
y_pred_binned = np.digitize(y_pred.numpy(), bins[:-1]) - 1  # Bin the predicted values

# Calculate the confusion matrix
cm = confusion_matrix(y_test_binned, y_pred_binned)

# Ensure the labels match the actual bins
unique_bins = np.unique(np.concatenate((y_test_binned, y_pred_binned)))  # Find unique bins
labels = [f"Bin {i}" for i in unique_bins]  # Generate labels for unique bins

# Plot the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix for Binned Regression Predictions")
plt.show()

# Calculate accuracy
accuracy = accuracy_score(y_test_binned, y_pred_binned)
print(f"Accuracy: {accuracy:.2f}")
