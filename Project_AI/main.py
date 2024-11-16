import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
from sklearn.decomposition import PCA

# Step 1: Define the transformations
# Resize images to 224x224 (as required for ResNet-18) and normalize them
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
])

# Step 2: Load the CIFAR-10 dataset
# Download the dataset and apply transformations
training_set = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform
)
test_set = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform
)


# Step 3: Subset the dataset to use only 500 training and 100 test images per class
def subset_cifar10(dataset, num_per_class):
    class_counts = {i: 0 for i in range(10)}  # Initialize counters for each class
    indices = []
    labels = []

    for idx, (_, label) in enumerate(dataset):
        if class_counts[label] < num_per_class:
            indices.append(idx)
            labels.append(label)
            class_counts[label] += 1
        if all(count >= num_per_class for count in class_counts.values()):
            break

    # Create a subset of the original dataset
    subset = torch.utils.data.Subset(dataset, indices)
    return subset, np.array(labels) # Return the subset and corresponding labels


# Use the function to subset the dataset
train_subset, train_labels = subset_cifar10(training_set, 500)  # 500 images per class for training
test_subset, test_labels = subset_cifar10(test_set, 100)  # 100 images per class for testing

# Step 4: Create DataLoaders
train_loader = torch.utils.data.DataLoader(train_subset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_subset, batch_size=64, shuffle=False)

# Save the labels for future use
np.save('train_labels.npy', train_labels)  # Save training labels
np.save('test_labels.npy', test_labels)  # Save test labels

# Load the pre-trained ResNet-18 model
resnet18 = models.resnet18(pretrained=True)

# Remove the final fully connected layer to use ResNet-18 as a feature extractor
resnet18 = nn.Sequential(*list(resnet18.children())[:-1])

# Set the model to evaluation mode
resnet18.eval()


def extract_features(model, dataloader):
    features = []
    with torch.no_grad():  # Disable gradient computation for efficiency
        for images, _ in dataloader:
            images = images.to('cuda' if torch.cuda.is_available() else 'cpu')  # Move images to GPU if available
            model = model.to('cuda' if torch.cuda.is_available() else 'cpu')  # Move model to GPU if available

            # Pass images through the model
            outputs = model(images)
            # Flatten the outputs from (batch_size, 512, 1, 1) to (batch_size, 512)
            outputs = outputs.view(outputs.size(0), -1)
            features.append(outputs.cpu().numpy())  # Move outputs to CPU and convert to NumPy

    # Concatenate all features into a single NumPy array
    return np.concatenate(features, axis=0)


# Assuming train_loader and test_loader are defined as in the previous code
train_features = extract_features(resnet18, train_loader)
test_features = extract_features(resnet18, test_loader)

# Save the features for future use (optional but recommended)
np.save('train_features.npy', train_features)
np.save('test_features.npy', test_features)

# Initialize PCA to reduce the dimensionality to 50 components
pca = PCA(n_components=100)

# Fit PCA on the training feature vectors and transform both training and testing feature vectors
train_features_pca = pca.fit_transform(train_features)
test_features_pca = pca.transform(test_features)

np.save('train_features_pca.npy', train_features_pca)
np.save('test_features_pca.npy', test_features_pca)


# Print summary
print("Train Features Shape:", train_features_pca.shape)
print("Test Features Shape:", test_features_pca.shape)
print("Sample Train Feature Vector:", train_features_pca[0])
print("Sample Test Feature Vector:", test_features_pca[0])
print("Number of Training Features:", len(train_features_pca))
print("Number of Test Features:", len(test_features_pca))
