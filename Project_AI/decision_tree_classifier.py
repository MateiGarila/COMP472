import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


class DecisionTree:
    def __init__(self, max_depth=50):
        self.max_depth = max_depth
        self.tree = None

    def _gini_impurity(self, y):
        """
        Calculate the Gini impurity for a set of labels.
        """
        unique_classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        gini = 1 - np.sum(probabilities ** 2)
        return gini

    def _best_split(self, X, y):
        """
        Find the best split by iterating over all features and thresholds.
        """
        best_gini = float('inf')
        best_split = None

        for feature_idx in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_idx])
            for threshold in thresholds:
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                left_gini = self._gini_impurity(y[left_mask])
                right_gini = self._gini_impurity(y[right_mask])
                gini = (np.sum(left_mask) / len(y)) * left_gini + \
                       (np.sum(right_mask) / len(y)) * right_gini

                if gini < best_gini:
                    best_gini = gini
                    best_split = {
                        'feature_idx': feature_idx,
                        'threshold': threshold,
                        'left_mask': left_mask,
                        'right_mask': right_mask
                    }

        return best_split

    def _build_tree(self, X, y, depth):
        """
        Recursively build the decision tree.
        """
        if depth == 0 or len(np.unique(y)) == 1:
            return np.bincount(y).argmax()

        split = self._best_split(X, y)
        if not split:
            return np.bincount(y).argmax()

        left_tree = self._build_tree(X[split['left_mask']], y[split['left_mask']], depth - 1)
        right_tree = self._build_tree(X[split['right_mask']], y[split['right_mask']], depth - 1)

        return {
            'feature_idx': split['feature_idx'],
            'threshold': split['threshold'],
            'left': left_tree,
            'right': right_tree
        }

    def fit(self, X, y):
        """
        Train the decision tree.
        """
        self.tree = self._build_tree(X, y, self.max_depth)

    def _predict_single(self, x, tree):
        """
        Predict the label for a single data point.
        """
        if isinstance(tree, dict):
            feature_idx = tree['feature_idx']
            threshold = tree['threshold']
            if x[feature_idx] <= threshold:
                return self._predict_single(x, tree['left'])
            else:
                return self._predict_single(x, tree['right'])
        else:
            return tree

    def predict(self, X):
        """
        Predict labels for a dataset.
        """
        return np.array([self._predict_single(x, self.tree) for x in X])


# Main function to train and test the model
if __name__ == "__main__":
    # Load the training and test data
    train_features = np.load("train_features_pca.npy")
    train_labels = np.load("train_labels.npy")
    test_features = np.load("test_features_pca.npy")
    test_labels = np.load("test_labels.npy")

    # Train the decision tree with max_depth=50
    dt = DecisionTree(max_depth=20)
    dt.fit(train_features, train_labels)

    # Predict on the test set
    predictions = dt.predict(test_features)

    # Calculate accuracy
    accuracy = np.mean(predictions == test_labels) * 100
    print(f"Decision Tree Accuracy (max_depth=50): {accuracy:.2f}%")

    # depths = [1, 5, 10, 20, 50]
    # accuracies = []
    #
    # for depth in depths:
    #     dt = DecisionTree(max_depth=depth)
    #     dt.fit(train_features, train_labels)
    #     predictions = dt.predict(test_features)
    #     accuracy = np.mean(predictions == test_labels) * 100
    #     accuracies.append(accuracy)
    #     print(f"Depth: {depth}, Accuracy: {accuracy:.2f}%")
    #
    # plt.plot(depths, accuracies, marker='o')
    # plt.title("Decision Tree Depth vs. Accuracy")
    # plt.xlabel("Tree Depth")
    # plt.ylabel("Test Set Accuracy (%)")
    # plt.grid()
    # plt.show()

    # Train and test the Decision Tree classifier with max_depth=50
    clf = DecisionTreeClassifier(criterion='gini', max_depth=20, random_state=42)
    clf.fit(train_features, train_labels)

    # Predict on the test set
    predictions = clf.predict(test_features)

    # Calculate accuracy
    accuracy = accuracy_score(test_labels, predictions) * 100
    print(f"Scikit-learn Decision Tree Accuracy (max_depth=50): {accuracy:.2f}%")
