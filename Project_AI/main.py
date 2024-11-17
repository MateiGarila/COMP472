import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchvision.models import resnet18
import torchvision.models as models
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Step 1: Define the transformations
# Resize images to 224x224 (as required for ResNet-18) and normalize them
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.225, 0.225, 0.225])  # Normalization as per ImageNet
])

# Step 2: Load the CIFAR-10 dataset
# Download the dataset and apply transformations
train_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform
)
test_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform
)


# Function to extract a subset of the dataset
def get_subset_by_class(dataset, num_per_class):
    targets = torch.tensor(dataset.targets)  # Access labels
    class_indices = [torch.where(targets == cls)[0][:num_per_class] for cls in range(10)]
    indices = torch.cat(class_indices)  # Concatenate indices from all classes
    return Subset(dataset, indices)


# Extract subsets
train_subset = get_subset_by_class(train_dataset, 500)  # 500 images per class
test_subset = get_subset_by_class(test_dataset, 100)  # 100 images per class

# Create data loaders
train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_subset, batch_size=64, shuffle=False)

# Load pre-trained ResNet-18
resnet = resnet18(pretrained=True)

# Modify ResNet-18 to remove the last layer (fc layer)
feature_extractor = nn.Sequential(*list(resnet.children())[:-1])  # All layers except the last one
# feature_extractor = nn.Sequential(*list(resnet.children())[:-2])  # Retain one more convolutional layer

# Ensure model is in evaluation mode
feature_extractor.eval()

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
feature_extractor = feature_extractor.to(device)


def extract_features(data_loader, model):
    features = []
    labels = []
    with torch.no_grad():  # No gradient computation
        for images, target in data_loader:
            images = images.to(device)
            outputs = model(images).squeeze()  # Extract features
            features.append(outputs.cpu().numpy())  # Move to CPU and convert to NumPy
            labels.append(target.numpy())
    return np.concatenate(features), np.concatenate(labels)


# Extract features for train and test sets
train_features, train_labels = extract_features(train_loader, feature_extractor)
test_features, test_labels = extract_features(test_loader, feature_extractor)

# Save the features for future use (optional but recommended)
np.save('train_features.npy', train_features)
np.save('test_features.npy', test_features)

# Apply PCA
pca = PCA(n_components=100)  # Increase the number of principal components
train_features_pca = pca.fit_transform(train_features)
test_features_pca = pca.transform(test_features)


np.save('train_features_pca.npy', train_features_pca)
np.save('test_features_pca.npy', test_features_pca)

# # Verify shapes
# print("Train Features Shape:", train_features_pca.shape)
# print("Test Features Shape:", test_features_pca.shape)

# Print summary
print("Train Features Shape:", train_features_pca.shape)
print("Test Features Shape:", test_features_pca.shape)
print("Sample Train Feature Vector:", train_features_pca[0])
print("Sample Test Feature Vector:", test_features_pca[0])
print("Number of Training Features:", len(train_features_pca))
print("Number of Test Features:", len(test_features_pca))

print("Train labels:", train_labels)
print("Test labels:", test_labels)

# Reduce to 2D for visualization
tsne = TSNE(n_components=2, random_state=42)
train_features_2d = tsne.fit_transform(train_features_pca)

# Plot
plt.figure(figsize=(8, 6))
for cls in np.unique(train_labels):
    plt.scatter(
        train_features_2d[train_labels == cls, 0],
        train_features_2d[train_labels == cls, 1],
        label=f"Class {cls}",
        alpha=0.7
    )
plt.legend()
plt.title("2D Visualization of PCA-Reduced Features")
plt.show()

# Reduce to 2D for visualization
tsne = TSNE(n_components=2, random_state=42)
train_features_2d = tsne.fit_transform(train_features)

# Plot
plt.figure(figsize=(8, 6))
for cls in np.unique(train_labels):
    plt.scatter(
        train_features_2d[train_labels == cls, 0],
        train_features_2d[train_labels == cls, 1],
        label=f"Class {cls}",
        alpha=0.7
    )
plt.legend()
plt.title("2D Visualization of Reduced Features")
plt.show()

# # Step 3: Subset the dataset to use only 500 training and 100 test images per class
# def subset_cifar10(dataset, num_per_class):
#     class_counts = {i: 0 for i in range(10)}  # Initialize counters for each class
#     indices = []
#     labels = []
#
#     for idx, (_, label) in enumerate(dataset):
#         if class_counts[label] < num_per_class:
#             indices.append(idx)
#             labels.append(label)
#             class_counts[label] += 1
#         if all(count >= num_per_class for count in class_counts.values()):
#             break
#
#     # Create a subset of the original dataset
#     subset = torch.utils.data.Subset(dataset, indices)
#     return subset, np.array(labels) # Return the subset and corresponding labels
#
#
# # Use the function to subset the dataset
# train_subset, train_labels = subset_cifar10(training_set, 500)  # 500 images per class for training
# test_subset, test_labels = subset_cifar10(test_set, 100)  # 100 images per class for testing
#
# # Step 4: Create DataLoaders
# train_loader = torch.utils.data.DataLoader(train_subset, batch_size=64, shuffle=True)
# test_loader = torch.utils.data.DataLoader(test_subset, batch_size=64, shuffle=False)
#
# # Save the labels for future use
# np.save('train_labels.npy', train_labels)  # Save training labels
# np.save('test_labels.npy', test_labels)  # Save test labels
#
# # Load the pre-trained ResNet-18 model
# resnet18 = models.resnet18(pretrained=True)
#
# # Remove the final fully connected layer to use ResNet-18 as a feature extractor
# resnet18 = nn.Sequential(*list(resnet18.children())[:-1])
#
# # Set the model to evaluation mode
# resnet18.eval()
#
#
# def extract_features(model, dataloader):
#     features = []
#     with torch.no_grad():  # Disable gradient computation for efficiency
#         for images, _ in dataloader:
#             images = images.to('cuda' if torch.cuda.is_available() else 'cpu')  # Move images to GPU if available
#             model = model.to('cuda' if torch.cuda.is_available() else 'cpu')  # Move model to GPU if available
#
#             # Pass images through the model
#             outputs = model(images)
#             # Flatten the outputs from (batch_size, 512, 1, 1) to (batch_size, 512)
#             outputs = outputs.view(outputs.size(0), -1)
#             features.append(outputs.cpu().numpy())  # Move outputs to CPU and convert to NumPy
#
#     # Concatenate all features into a single NumPy array
#     return np.concatenate(features, axis=0)
#
#
# # Assuming train_loader and test_loader are defined as in the previous code
# train_features = extract_features(resnet18, train_loader)
# test_features = extract_features(resnet18, test_loader)
#
# # Save the features for future use (optional but recommended)
# np.save('train_features.npy', train_features)
# np.save('test_features.npy', test_features)
#
# # Initialize PCA to reduce the dimensionality to 50 components
# pca = PCA(n_components=100)
#
# # Fit PCA on the training feature vectors and transform both training and testing feature vectors
# train_features_pca = pca.fit_transform(train_features)
# test_features_pca = pca.transform(test_features)
#
# np.save('train_features_pca.npy', train_features_pca)
# np.save('test_features_pca.npy', test_features_pca)
#
#
# # Print summary
# print("Train Features Shape:", train_features_pca.shape)
# print("Test Features Shape:", test_features_pca.shape)
# print("Sample Train Feature Vector:", train_features_pca[0])
# print("Sample Test Feature Vector:", test_features_pca[0])
# print("Number of Training Features:", len(train_features_pca))
# print("Number of Test Features:", len(test_features_pca))
