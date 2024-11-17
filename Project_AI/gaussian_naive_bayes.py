import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


class GaussianNaiveBayes:
    def __init__(self):
        self.classes = None  # Unique class labels
        self.means = None  # Mean of each feature per class
        self.variances = None  # Variance of each feature per class
        self.priors = None  # Prior probabilities of each class

    def fit(self, X, y):
        """
        Train the Gaussian Naive Bayes model.
        Parameters:
            X: np.ndarray
                Training feature vectors, shape (num_samples, num_features)
            y: np.ndarray
                Class labels, shape (num_samples,)
        """
        self.classes = np.unique(y)
        num_classes = len(self.classes)
        num_features = X.shape[1]

        self.means = np.zeros((num_classes, num_features))
        self.variances = np.zeros((num_classes, num_features))
        self.priors = np.zeros(num_classes)

        for idx, cls in enumerate(self.classes):
            X_cls = X[y == cls]  # Select samples of the current class
            self.means[idx, :] = np.mean(X_cls, axis=0)
            self.variances[idx, :] = np.var(X_cls, axis=0)
            self.priors[idx] = X_cls.shape[0] / X.shape[0]  # Prior probability

    def _gaussian_density(self, x, mean, variance):
        """
        Compute the Gaussian probability density function for a given feature.
        """
        epsilon = 1e-9  # Small constant to avoid division by zero
        coef = 1.0 / np.sqrt(2.0 * np.pi * (variance + epsilon))
        exponent = np.exp(-((x - mean) ** 2) / (2.0 * (variance + epsilon)))
        return coef * exponent

    def predict(self, X):
        """
        Predict the class labels for a given dataset.
        Parameters:
            X: np.ndarray
                Feature vectors, shape (num_samples, num_features)
        Returns:
            np.ndarray
                Predicted class labels, shape (num_samples,)
        """
        num_samples = X.shape[0]
        num_classes = len(self.classes)
        log_probs = np.zeros((num_samples, num_classes))

        for idx, cls in enumerate(self.classes):
            mean = self.means[idx]
            variance = self.variances[idx]
            prior = np.log(self.priors[idx])
            log_likelihood = np.sum(np.log(self._gaussian_density(X, mean, variance)), axis=1)
            log_probs[:, idx] = log_likelihood + prior

        return self.classes[np.argmax(log_probs, axis=1)]


# Main function to train and test the model
if __name__ == "__main__":
    # Load the training and test data
    train_features = np.load("train_features_pca.npy")
    train_labels = np.load("train_labels.npy")
    test_features = np.load("test_features_pca.npy")
    test_labels = np.load("test_labels.npy")

    # Initialize and train the model
    model = GaussianNaiveBayes()
    model.fit(train_features, train_labels)

    # Predict the labels for the test set
    predictions = model.predict(test_features)

    # Calculate accuracy
    accuracy = np.mean(predictions == test_labels) * 100
    print(f"Gaussian Naive Bayes Accuracy: {accuracy:.2f}%")

    # Initialize the Gaussian Naive Bayes classifier
    gnb = GaussianNB()

    # Train the model
    gnb.fit(train_features, train_labels)

    # Predict the labels for the test set
    y_pred = gnb.predict(test_features)

    # Calculate the accuracy
    accuracy = (accuracy_score(test_labels, y_pred)) * 100
    print(f'Accuracy: {accuracy:.2f}%')
