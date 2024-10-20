import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from scripts.utils import load_data, save_model, log_metrics

# Load configuration from YAML
with open('./config/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Check if GPU is available
use_gpu = torch.cuda.is_available()
device = torch.device('cuda' if use_gpu else 'cpu')

# Basic Neural Network Model
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_model():
    # Load data
    train_loader, val_loader = load_data(config['data_path'], config['batch_size'])

    # Initialize the model, loss function, and optimizer
    model = SimpleNet().to(device)  # Move the model to the GPU if available
    criterion = nn.CrossEntropyLoss().to(device)  # Loss function on the same device
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    for epoch in range(config['num_epochs']):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)  # Move data to the correct device
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Print loss
        print(f'Epoch [{epoch+1}/{config["num_epochs"]}], Loss: {total_loss:.4f}')
        log_metrics(epoch, total_loss, config['log_path'])

    # Save the model
    save_model(model, config['model_save_path'])

if __name__ == "__main__":
    train_model()
