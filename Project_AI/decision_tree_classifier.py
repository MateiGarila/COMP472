import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from joblib import dump, load
import seaborn as sns


class DecisionTreeClassifier:
    def __init__(self, max_depth):
        self.max_depth = max_depth
        self.tree = None

    def _gini(self, y):
        """
        Calculate Gini impurity for a given set of labels.
        """
        unique_classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / counts.sum()
        return 1 - np.sum(probabilities ** 2)

    def _split(self, X, y, feature_index, threshold):
        """
        Split the dataset based on a feature and a threshold.
        """
        left_mask = X[:, feature_index] <= threshold
        right_mask = X[:, feature_index] > threshold
        return (X[left_mask], y[left_mask]), (X[right_mask], y[right_mask])

    def _find_best_split(self, X, y):
        """
        Find the best feature and threshold to split the data.
        """
        num_samples, num_features = X.shape
        best_gini = float("inf")
        best_split = None

        for feature_index in range(num_features):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                (X_left, y_left), (X_right, y_right) = self._split(X, y, feature_index, threshold)
                if len(y_left) == 0 or len(y_right) == 0:
                    continue

                # Calculate weighted Gini impurity
                gini_left = self._gini(y_left)
                gini_right = self._gini(y_right)
                weighted_gini = (len(y_left) * gini_left + len(y_right) * gini_right) / num_samples

                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_split = {
                        "feature_index": feature_index,
                        "threshold": threshold,
                        "left": (X_left, y_left),
                        "right": (X_right, y_right)
                    }

        return best_split

    def _build_tree(self, X, y, depth=0):
        """
        Recursively build the decision tree.
        """
        num_samples, num_features = X.shape
        unique_classes = np.unique(y)

        # Stopping conditions
        if depth >= self.max_depth or len(unique_classes) == 1 or num_samples == 0:
            leaf_value = np.argmax(np.bincount(y)) if len(y) > 0 else None
            return {"leaf": True, "value": leaf_value}

        # Find the best split
        split = self._find_best_split(X, y)
        if not split:
            leaf_value = np.argmax(np.bincount(y))
            return {"leaf": True, "value": leaf_value}

        # Create subtree
        return {
            "leaf": False,
            "feature_index": split["feature_index"],
            "threshold": split["threshold"],
            "left": self._build_tree(*split["left"], depth + 1),
            "right": self._build_tree(*split["right"], depth + 1)
        }

    def fit(self, X, y):
        """
        Train the decision tree.
        """
        self.tree = self._build_tree(X, y)

    def _predict_one(self, x, tree):
        """
        Predict the class label for a single sample.
        """
        if tree["leaf"]:
            return tree["value"]
        feature_index = tree["feature_index"]
        threshold = tree["threshold"]

        if x[feature_index] <= threshold:
            return self._predict_one(x, tree["left"])
        else:
            return self._predict_one(x, tree["right"])

    def predict(self, X):
        """
        Predict the class labels for a dataset.
        """
        return np.array([self._predict_one(x, self.tree) for x in X])


# Main function to train and test the model
if __name__ == "__main__":
    # Load the training and test data
    train_features = np.load("train_features_pca.npy")
    train_labels = np.load("training_labels.npy")
    test_features = np.load("test_features_pca.npy")
    test_labels = np.load("testing_labels.npy")

    # Check if the model is already saved
    model_file = "DecisionTreeClassifier/decision_tree_model.pkl"

    if os.path.exists(model_file):
        # Load the pre-trained model
        with open(model_file, "rb") as file:
            dt = pickle.load(file)
        print("Loaded the pre-trained model")
    else:
        print("Training the model")
        # Train the decision tree with max_depth=10
        dt = DecisionTreeClassifier(max_depth=10)
        dt.fit(train_features, train_labels)
        print("Finished training the model")

        # Save the trained model
        with open(model_file, "wb") as file:
            pickle.dump(dt, file)
        print("Saved the trained model")

    # Predict on the test set
    predictions = dt.predict(test_features)
    print("Finished predicting test features")

    # Calculate accuracy
    accuracy = np.mean(predictions == test_labels) * 100
    print(f"Decision Tree Accuracy (max_depth=50): {accuracy:.2f}%")

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
    plt.title("Normalized Confusion Matrix for Custom Decision Tree Classifier")
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

    # Train and test the Decision Tree classifier with max_depth=50
    # To use saved trained model comment out the training and saving lines
    # clf = DecisionTreeClassifier(criterion='gini', max_depth=50, random_state=42)
    # clf.fit(train_features, train_labels)
    #
    # # Save the model - to use saved model comment until this line
    # dump(clf, "DecisionTreeClassifier/decision_tree_classifier_SK50.joblib")
    #
    # This is the saved trained model
    clf = load("DecisionTreeClassifier/decision_tree_classifier_SK10.joblib")

    # Predict on the test set
    predictions = clf.predict(test_features)

    # Calculate accuracy
    accuracy = accuracy_score(test_labels, predictions) * 100
    print(f"Scikit-learn Decision Tree Accuracy (max_depth=50): {accuracy:.2f}%")

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
    plt.title("Normalized Confusion Matrix for SKlearn Decision Tree Classifier")
    plt.show()

    # Compute metrics
    accuracy = accuracy_score(test_labels, predictions)
    precision = precision_score(test_labels, predictions, average="weighted")  # Weighted by class frequency
    recall = recall_score(test_labels, predictions, average="weighted")
    f1 = f1_score(test_labels, predictions, average="weighted")

    print(f"Sklearn Accuracy: {accuracy:.2f}")
    print(f"Sklearn Precision: {precision:.2f}")
    print(f"Sklearn Recall: {recall:.2f}")
    print(f"Sklearn F1-Score: {f1:.2f}\n")
