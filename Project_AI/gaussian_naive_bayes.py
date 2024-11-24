import os
import pickle

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from joblib import dump, load


class GaussianNaiveBayes:
    def __init__(self):
        self.classes = None  # Unique class labels
        self.means = None  # Mean of each feature per class
        self.variances = None  # Variance of each feature per class
        self.priors = None  # Prior probabilities of each class

    """
    Train the model
    """
    def fit(self, image, label):

        self.classes = np.unique(label)
        num_classes = len(self.classes)
        num_features = image.shape[1]

        self.means = np.zeros((num_classes, num_features))
        self.variances = np.zeros((num_classes, num_features))
        self.priors = np.zeros(num_classes)

        for idx, cls in enumerate(self.classes):
            X_cls = image[label == cls]  # Select samples of the current class
            self.means[idx, :] = np.mean(X_cls, axis=0)
            self.variances[idx, :] = np.var(X_cls, axis=0)
            self.priors[idx] = X_cls.shape[0] / image.shape[0]  # Prior probability

    """
    Computes Gaussian probability density for given feature
    """
    def _gaussian_density(self, image, mean, variance):

        epsilon = 1e-9  # Small constant to avoid division by zero
        coef = 1.0 / np.sqrt(2.0 * np.pi * (variance + epsilon))
        exponent = np.exp(-((image - mean) ** 2) / (2.0 * (variance + epsilon)))
        return coef * exponent

    """
    Predict label given dataset
    """
    def predict(self, dataset):

        num_samples = dataset.shape[0]
        num_classes = len(self.classes)
        log_probs = np.zeros((num_samples, num_classes))

        for idx, cls in enumerate(self.classes):
            mean = self.means[idx]
            variance = self.variances[idx]
            prior = np.log(self.priors[idx])
            log_likelihood = np.sum(np.log(self._gaussian_density(dataset, mean, variance)), axis=1)
            log_probs[:, idx] = log_likelihood + prior

        return self.classes[np.argmax(log_probs, axis=1)]


# Main function to train and test the model
if __name__ == "__main__":
    # Load the training and test data
    train_features = np.load("train_features_pca.npy")
    train_labels = np.load("training_labels.npy")
    test_features = np.load("test_features_pca.npy")
    test_labels = np.load("testing_labels.npy")

    # Check if the model is already saved
    model_file = "GaussianNaiveBayes/gaussian_naive_bayes_model.pkl"

    if os.path.exists(model_file):
        # Load the pre-trained model
        with open(model_file, "rb") as file:
            model = pickle.load(file)
        print("Loaded the pre-trained Gaussian Naive Bayes model")
    else:
        # Initialize and train the Gaussian Naive Bayes model
        model = GaussianNaiveBayes()
        model.fit(train_features, train_labels)
        print("Finished training the Gaussian Naive Bayes model")

        # Save the trained model
        with open(model_file, "wb") as file:
            pickle.dump(model, file)
        print("Saved the trained Gaussian Naive Bayes model")

    # Predict the labels for the test set
    predictions = model.predict(test_features)
    print("Finished predicting test features")

    # Calculate accuracy
    accuracy = np.mean(predictions == test_labels) * 100
    print(f"Gaussian Naive Bayes Accuracy: {accuracy:.2f}%")

    # CIFAR-10 class labels
    class_names = [
        "0", "1", "2", "3", "4",
        "5", "6", "7", "8", "9"
    ]

    # Generate the confusion matrix
    cm = confusion_matrix(test_labels, predictions)

    # Normalize the confusion matrix for better visualization (optional)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    # Plot the confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Normalized Confusion Matrix for Custom Gaussian Naive Bayes")
    plt.show()

    # Compute metrics
    accuracy = accuracy_score(test_labels, predictions)
    precision = precision_score(test_labels, predictions, average="weighted")  # Weighted by class frequency
    recall = recall_score(test_labels, predictions, average="weighted")
    f1 = f1_score(test_labels, predictions, average="weighted")

    print(f"Custom Accuracy: {accuracy:.2f}")
    print(f"Custom Precision: {precision:.2f}")
    print(f"Custom Recall: {recall:.2f}")
    print(f"Custom F1-Score: {f1:.2f}\n")

    # Initialize the Gaussian Naive Bayes classifier
    # To use saved trained model comment out the training and saving lines
    gnb = GaussianNB()

    # Train the model
    gnb.fit(train_features, train_labels)

    # Save the model - to use saved model comment until this line (123)
    dump(gnb, "GaussianNaiveBayes/gaussian_naive_bayes_modelSK.joblib")

    # This is the saved trained model
    # gnb = load("GaussianNaiveBayes/gaussian_naive_bayes_modelSK.joblib")

    # Predict the labels for the test set
    y_pred = gnb.predict(test_features)

    # Calculate the accuracy
    accuracy = (accuracy_score(test_labels, y_pred)) * 100
    print(f'Sklearn model Accuracy: {accuracy:.2f}%')

    # Generate the confusion matrix
    cm = confusion_matrix(test_labels, y_pred)

    # Normalize the confusion matrix for better visualization (optional)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    # Plot the confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Normalized Confusion Matrix for Sklearn Gaussian Naive Bayes")
    plt.show()

    # Compute metrics
    accuracy = accuracy_score(test_labels, y_pred)
    precision = precision_score(test_labels, y_pred, average="weighted")  # Weighted by class frequency
    recall = recall_score(test_labels, y_pred, average="weighted")
    f1 = f1_score(test_labels, y_pred, average="weighted")

    print(f"Sklearn Accuracy: {accuracy:.2f}")
    print(f"Sklearn Precision: {precision:.2f}")
    print(f"Sklearn Recall: {recall:.2f}")
    print(f"Sklearn F1-Score: {f1:.2f}\n")
