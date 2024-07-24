import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchvision.transforms import ToTensor
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import torchvision

# Define the neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Custom dataset class to handle image transformations
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def train_model():
    print("Training the model...")


    # Initialize the model, loss function, and optimizer
    model = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        for batch in train_loader:
            images, labels = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        # Evaluation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in test_loader:
                images, labels = batch[0].to(device), batch[1].to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Accuracy: {100 * correct / total:.2f}%")

    print("Training finished!")

    # Save the model parameters
    torch.save(model.state_dict(), 'bin/model_weights.pth')

def analyze_model():
    print("Analyzing the model...")
    # Load the model
    model = Net()
    model.load_state_dict(torch.load('bin/model_weights.pth'))
    model.eval()


    # Collect misclassified samples
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    misclassified_images = []
    misclassified_labels = []
    misclassified_predictions = []

    with torch.no_grad():
        for batch in test_loader:
            images, labels = batch[0].to(device), batch[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            # Find misclassified samples
            misclassified_mask = (predicted != labels)
            misclassified_images.extend(images[misclassified_mask].cpu())
            misclassified_labels.extend(labels[misclassified_mask].cpu())
            misclassified_predictions.extend(predicted[misclassified_mask].cpu())

            if len(misclassified_images) >= 10:
                break

    # Function to show images
    def imshow(img,ax):
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        ax.imshow(np.transpose(npimg, (1, 2, 0)),cmap='gray')

    # Plot 10 misclassified samples
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    for i, ax in enumerate(axes.flat):
        if i < len(misclassified_images):
            imshow(misclassified_images[i],ax)
            ax.set_title(f'True: {misclassified_labels[i]}\nPred: {misclassified_predictions[i]}')
            ax.axis('off')

    plt.tight_layout()
    os.makedirs(f'bin/misclassified_samples/', exist_ok=True)
    plt.savefig(f'bin/misclassified_samples/sample_{i+1}.png')
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description='MNIST Model Training and Analysis')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--analysis', action='store_true', help='Analyze the model')
    args = parser.parse_args()

    if args.train:
        train_model()
    if args.analysis:
        analyze_model()
    if not (args.train or args.analysis):
        print("Please specify either --train or --analysis")

if __name__ == "__main__":
    main()
