import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.models import resnet18
from data import get_loaders
from explore import train_dataset, test_dataset, train_loader, test_loader

SEED = torch.manual_seed(0)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")



# Model 1: Regular CNN

class Create_CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Linear(64, 2)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x
    


class Create_resnet18():
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)




# Instantiate model, loss, and optimizer

cnn = Create_CNN().to(device)
criterion = nn.CrossEntropyLoss()
cnn_optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)

num_epochs = 10

epoch_losses = []

print("Training CNN Model...")
for epoch in range(num_epochs):
    cnn.train()
    running_loss = 0.0
    total = 0

    for images, (labels, masks) in train_loader:
        images = images.to(device)
        labels = labels.to(device).long()

        cnn_optimizer.zero_grad()

        # Forward pass: predict class scores
        outputs = cnn(images)

        # Calculate loss using predictions and labels
        loss = criterion(outputs, labels)

        # Backpropagation and optimization
        loss.backward()
        cnn_optimizer.step()

        # Accumulate loss across samples
        running_loss += loss.item() * images.size(0)
        total += images.size(0)

    avg_train_loss = running_loss / total
    epoch_losses.append(avg_train_loss)

    print(
        f"Epoch [{epoch + 1}/{num_epochs}], "
        f"Average Training Loss: {avg_train_loss:.4f}"
    )


resnet = Create_resnet18().to(device)
criterion = nn.CrossEntropyLoss()
resnet_optimizer = torch.optim.Adam(resnet.parameters(), lr=0.01)