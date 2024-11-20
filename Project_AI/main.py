import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.models as models
import numpy as np
from sklearn.decomposition import PCA  # Import PCA from sklearn

# Step 0: Define the device you are going to use
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Step 1: Define the transformations
# Resize images to 224x224 (as required for ResNet-18) and normalize them
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # Normalize images as per ImageNet
])
print("Done defining transformation")

# Step 2: Load the CIFAR-10 dataset with specified transformations
train_data = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform
)
test_data = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform
)
print("Done loading initial data")

# Step 3: Load pre-trained ResNet-18 model
resnet18 = models.resnet18(pretrained=True)
# Remove the last layer to use ResNet as a feature extractor
resnet18 = nn.Sequential(*list(resnet18.children())[:-1])
resnet18.eval()  # Set to evaluation mode

# Move the model to GPU
resnet18 = resnet18.to(device)
print("Done loading and setting pre-trained model")

# Step 4: Create DataLoaders for batch processing
train_loader = DataLoader(train_data, batch_size=16, shuffle=False)
test_loader = DataLoader(test_data, batch_size=16, shuffle=False)
print("Done creating loaders")


# Function to extract feature vectors
def extract_features(loader):
    features = []
    labels = []
    with torch.no_grad():  # Don’t calculate gradients, we just want features
        for images, label in loader:
            # Move images and labels to GPU
            images = images.to(device)
            output = resnet18(images)
            output = output.view(output.size(0), -1)  # Flatten the output

            # Move output and labels back to CPU before converting to numpy
            features.append(output.cpu().numpy())
            labels.extend(label.numpy())
    return np.concatenate(features), np.array(labels)

    # Extract features for training and testing data


# Step 5: Extract the Features from the loaders
training_features, training_labels = extract_features(train_loader)
print("Done extracting training features")

testing_features, testing_labels = extract_features(test_loader)
print("Done extracting testing features")

# Save the features for future use (optional but recommended)
np.save('train_features.npy', training_features)
np.save('training_labels.npy', training_labels)
np.save('test_features.npy', testing_features)
np.save('testing_labels.npy', testing_labels)

# Step 6: Apply PCA to reduce the feature dimensions from 512 to 50
pca = PCA(n_components=50)  # Set number of components to 50
training_features_reduced = pca.fit_transform(training_features)
print("Done applying PCA to training features")

testing_features_reduced = pca.transform(testing_features)  # Apply the same transformation to the test set
print("Done applying PCA to testing features")

# Save the features for future use (optional but recommended)
np.save('train_features_pca.npy', training_features_reduced)
np.save('test_features_pca.npy', testing_features_reduced)

# Store the processed data in data structures
processed_data = {
    'training_labels': training_labels,
    'training_features': training_features_reduced,
    'testing_labels': testing_labels,
    'testing_features': testing_features_reduced
}

# Example print statements to check the data format

# 1. First few training labels (first 5)
print("First 5 training labels:", training_labels[:5])

# 2. Shape of the training features (after PCA)
print("Shape of training features:", training_features.shape)

# 3. First little training feature vectors (first 5 vectors)
print("First 5 training feature vectors (PCA reduced):")
print(training_features[:5])

# 4. First few testing labels (first 5)
print("First 5 testing labels:", testing_labels[:5])

# 5. Shape of the testing features (after PCA)
print("Shape of testing features:", testing_features.shape)

# 6. First few testing feature vectors (first 5 vectors)
print("First 5 testing feature vectors (PCA reduced):")
print(testing_features[:5])

# import torch
# import torch.nn as nn
# import torchvision
# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader, Subset
# from torchvision.models import resnet18
# import torchvision.models as models
# import numpy as np
# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt

# # Step 1: Define the transformations
# # Resize images to 224x224 (as required for ResNet-18) and normalize them
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),  # Resize images to 224x224
#     transforms.ToTensor(),  # Convert images to PyTorch tensors
#     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # Normalization as per ImageNet
# ])
# print("Done defining transformation")
#
# # Step 2: Load the CIFAR-10 dataset
# # Download the dataset and apply transformations
# train_dataset = torchvision.datasets.CIFAR10(
#     root='./data', train=True, download=True, transform=transform
# )
# test_dataset = torchvision.datasets.CIFAR10(
#     root='./data', train=False, download=True, transform=transform
# )
# print("Done loading initial data")
#
#
# # Function to extract a subset of the dataset
# def get_subset_by_class(dataset, num_per_class):
#     targets = torch.tensor(dataset.targets)  # Access labels
#     class_indices = [torch.where(targets == cls)[0][:num_per_class] for cls in range(10)]
#     indices = torch.cat(class_indices)  # Concatenate indices from all classes
#     return Subset(dataset, indices)
#
#
# # Extract subsets
# train_subset = get_subset_by_class(train_dataset, 500)  # 500 images per class
# test_subset = get_subset_by_class(test_dataset, 100)  # 100 images per class
#
# # Create data loaders
# # train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
# # test_loader = DataLoader(test_subset, batch_size=64, shuffle=False)
#
# # Load pre-trained ResNet-18
# resnet18 = resnet18(pretrained=True)
#
# # Modify ResNet-18 to remove the last layer (fc layer)
# resnet18 = nn.Sequential(*list(resnet18.children())[:-1])  # All layers except the last one
#
# # Ensure model is in evaluation mode
# resnet18.eval()
#
# # Move to GPU if available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# resnet18 = resnet18.to(device)
# print("Done loading and setting pre-trained model")
#
# # Create DataLoaders for batch processing
# train_loader = DataLoader(train_subset, batch_size=16, shuffle=False)
# test_loader = DataLoader(test_subset, batch_size=16, shuffle=False)
# print("Done creating loaders")
#
#
# def extract_features(data_loader):
#     features = []
#     labels = []
#     with torch.no_grad():  # No gradient computation
#         for images, label in data_loader:
#             # Move images and labels to GPU
#             images = images.to(device)
#             output = resnet18(images)
#             output = output.view(output.size(0), -1) # Flatten the output
#
#             # Move output and labels to CPU before converting to numpy
#             features.append(output.cpu().numpy())
#             labels.extend(label.numpy())
#     return np.concatenate(features), np.array(labels)
#
#
# # Extract features for train and test sets
# train_features, train_labels = extract_features(train_loader)
# print("Done extracting training features")
#
# test_features, test_labels = extract_features(test_loader)
# print("Done extracting testing features")
#
#
# # Save the features for future use (optional but recommended)
# np.save('train_features.npy', train_features)
# np.save('test_features.npy', test_features)
#
# # Apply PCA
# pca = PCA(n_components=50)  # Increase the number of principal components
#
# train_features_pca = pca.fit_transform(train_features)
# print("Done applying PCA to training features")
#
# test_features_pca = pca.transform(test_features)
# print("Done applying PCA to testing features")
#
# np.save('train_features_pca.npy', train_features_pca)
# np.save('test_features_pca.npy', test_features_pca)
#
# # # Verify shapes
# # print("Train Features Shape:", train_features_pca.shape)
# # print("Test Features Shape:", test_features_pca.shape)
#
# # Print summary
# print("Train Features Shape:", train_features_pca.shape)
# print("Test Features Shape:", test_features_pca.shape)
# print("Sample Train Feature Vector:", train_features_pca[0])
# print("Sample Test Feature Vector:", test_features_pca[0])
# print("Number of Training Features:", len(train_features_pca))
# print("Number of Test Features:", len(test_features_pca))
#
# # print("Train labels:", train_labels)
# # print("Test labels:", test_labels)
#
# # Reduce to 2D for visualization
# tsne = TSNE(n_components=2, random_state=42)
# train_features_2d = tsne.fit_transform(train_features_pca)
#
# # Plot
# plt.figure(figsize=(8, 6))
# for cls in np.unique(train_labels):
#     plt.scatter(
#         train_features_2d[train_labels == cls, 0],
#         train_features_2d[train_labels == cls, 1],
#         label=f"Class {cls}",
#         alpha=0.7
#     )
# plt.legend()
# plt.title("2D Visualization of PCA-Reduced Features")
# plt.show()
