import os
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from efficientnet_pytorch import EfficientNet
from PIL import Image, ImageEnhance
from sklearn.model_selection import StratifiedKFold

# Define hyperparameters
EPOCH = 50
LR = 1e-3

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define a custom dataset class
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.labels = [int(label) for label in os.listdir(root_dir)]  # Get the directory names as labels
        self.images = []
        for label in self.labels:
                label_dir = os.path.join(root_dir, str(label))
                self.images.extend([(img, label) for img in os.listdir(label_dir)])
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name, label = self.images[idx]
        img_path = os.path.join(self.root_dir, str(label), img_name)
        image = Image.open(img_path).convert("L")  # Open the image as PIL Image and convert to grayscale
        sample = {"image": image, "label": label, "img_name": img_name}
        
        if self.transform:
            sample["image"] = self.transform(sample["image"])
        return sample

# Define data transformations
transform = transforms.Compose([
    transforms.Resize((384, 384)),
    # undo randomresizecrop below to get randomly cropped data then comment it, comment transforms.ToTensor() while doing the transform after getting the images comment this and preprocess step 
    #transforms.RandomResizedCrop(size=(384, 384), scale=(0.8, 1.0), ratio=(1, 1)),
    transforms.ToTensor(),
])

def save_transformed_image(image, save_dir):
    # Create directory if it doesn't exist
    #os.makedirs(save_dir, exist_ok=True)
    # Save transformed image
    image.save(save_dir)

import torchvision.transforms.functional as TF
# Preprocess data and create labeled dataset
def preprocess(image_dir, ground_truth_path, dest_dir):
    df = pd.read_csv(ground_truth_path)
    
    for idx, row in df.iterrows():
        img_name = row["image name"]
        img_quality = row["image quality level"]
        img_path = os.path.join(image_dir, img_name)
        dest_subdir = os.path.join(dest_dir, str(img_quality))
                
        os.makedirs(dest_subdir, exist_ok=True)

        img = Image.open(img_path)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.2)

        if img_quality == 2:
            img.save(os.path.join(dest_subdir, img_name))
        else:
            img.save(os.path.join(dest_subdir, img_name))
            img = transforms.RandomVerticalFlip(p=1)(img)
            img.save(os.path.join(dest_subdir, f"flip1_{img_name}"))
            img = transforms.RandomHorizontalFlip(p=1)(img)
            img.save(os.path.join(dest_subdir, f"flip2_{img_name}"))
            img = transforms.RandomVerticalFlip(p=1)(img)
            img.save(os.path.join(dest_subdir, f"flip3_{img_name}"))

            #img = transforms.RandomResizedCrop(size=(384, 384), scale=(0.8, 1.0), ratio=(1, 1))
            #img.save(os.path.join(dest_subdir, f"cropped_{img_name}"))
            transformed_image = transform(img)
    # Save transformed image
            save_transformed_image(transformed_image, os.path.join(dest_subdir, f"cropped_{img_name}"))

preprocess("./B. Image Quality Assessment/1. Original Images/a. Training Set",
           "./B. Image Quality Assessment/2. Groundtruths/a. DRAC2022_ Image Quality Assessment_Training Labels.csv",
           "./labeled_data")

# Load the labeled dataset
dataset = CustomDataset("./labeled_data", transform=transform)

# Define number of folds for cross-validation
num_folds = 3
skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

# Initialize a list to store validation accuracies for each fold
val_accuracies = []

# Iterate over folds
labels = np.array([sample['label'] for sample in dataset])

# Initialize StratifiedKFold with the correct labels
skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
for fold, (train_indices, val_indices) in enumerate(skf.split(dataset, labels)):
    print(f"Fold {fold + 1}")

    # Create data loaders for the current fold
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset, batch_size=16, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=16, sampler=val_sampler)

    # Initialize the model
    model = EfficientNet.from_pretrained('efficientnet-b2', num_classes=3)
    model._conv_stem = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)  # Adjust input channels
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    decayRate = 0.96
    my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)

    # Train the model
    max_val_accuracy = 0
    for epoch in range(EPOCH):
        model.train()
        running_loss = 0.0
        total_correct = 0
        total_samples = 0
        for i, data in enumerate(train_loader):
            inputs, labels = data["image"].to(device), data["label"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()
        
        train_accuracy = 100 * total_correct / total_samples

        # Validate the model
        model.eval()
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for data in val_loader:
                inputs, labels = data["image"].to(device), data["label"].to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                total_correct += (predicted == labels).sum().item()
        val_accuracy = 100 * total_correct / total_samples
        if val_accuracy > max_val_accuracy:
            max_val_accuracy = val_accuracy
            torch.save(model.state_dict(), f"model_fold_{fold}.pth")
        my_lr_scheduler.step()

        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}, Train Accuracy: {train_accuracy}%, Validation Accuracy: {val_accuracy}%")
    
    val_accuracies.append(max_val_accuracy)

# Calculate and print the average validation accuracy across all folds
avg_val_accuracy = np.mean(val_accuracies)
print(f"Average validation accuracy across {num_folds} folds: {avg_val_accuracy}%")

# Evaluate the model on the test set
test_dataset = CustomDataset("./B. Image Quality Assessment/1. Original Images/b. Testing Set", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=16)

# Load the model trained on the entire training set (not just one fold)
model = EfficientNet.from_pretrained('efficientnet-b2', num_classes=3)
model._conv_stem = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)  # Adjust input channels
model.load_state_dict(torch.load(f"model_fold_{fold}.pth"))
model = model.to(device)
model.eval()

# Initialize variables to calculate accuracy
total_correct = 0
total_samples = 0

# Evaluate the model on the test set
with torch.no_grad():
    for data in test_loader:
        inputs, labels = data["image"].to(device), data["label"].to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        total_correct += (predicted == labels).sum().item()

# Calculate test accuracy
test_accuracy = 100 * total_correct / total_samples
print(f"Accuracy on the test set: {test_accuracy}%")
