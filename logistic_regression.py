import numpy as np
from sklearn.preprocessing import StandardScaler


class LogisticRegressionOvR:
    def __init__(self, learning_rate=0.001, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.models = []
        self.scaler = StandardScaler(with_mean=False)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def compute_loss(self, y_true, y_pred, class_weights):
        epsilon = 1e-9
        y1 = y_true * np.log(y_pred + epsilon)
        y2 = (1 - y_true) * np.log(1 - y_pred + epsilon)
        return -np.mean(class_weights * (y1 + y2))

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Normalize the data
        if hasattr(X, 'toarray'):
            # Convert to dense array if necessary
            X = self.scaler.fit_transform(X.toarray())
        else:
            X = self.scaler.fit_transform(X)  # Already dense array

        # Train a separate model for each class (One-vs-Rest)
        unique_classes = np.unique(y)
        for cls in unique_classes:
            # Create binary labels for the current class
            y_binary = np.where(y == cls, 1, 0)

            # Initialize parameters for the current class
            weights = np.random.randn(n_features) * 0.01
            bias = 0

            # Calculate class weights
            class_counts = np.bincount(y_binary)
            total_samples = len(y_binary)
            class_weights = {0: total_samples /
                             class_counts[0], 1: total_samples / class_counts[1]}

            # Gradient descent for the current class
            for _ in range(self.n_iters):
                z = X.dot(weights) + bias
                A = self._sigmoid(z)
                weighted_loss = self.compute_loss(y_binary, A, np.array(
                    [class_weights[label] for label in y_binary]))
                dz = (A - y_binary) * \
                    np.array([class_weights[label] for label in y_binary])
                dw = (1 / n_samples) * X.T.dot(dz)
                db = (1 / n_samples) * np.sum(dz)
                weights -= self.lr * dw
                bias -= self.lr * db

            # Store the trained weights and bias for the current class
            self.models.append((weights, bias))

    def predict(self, X):
        # Normalize the data using the fitted scaler
        if hasattr(X, 'toarray'):
            # Convert to dense array if necessary
            X = self.scaler.transform(X.toarray())
        else:
            X = self.scaler.transform(X)  # Already dense array

        # Compute predictions for each class and choose the class with the highest probability
        predictions = []
        for weights, bias in self.models:
            z = X.dot(weights) + bias
            A = self._sigmoid(z)
            predictions.append(A)

        predictions = np.array(predictions)
        y_predicted_cls = np.argmax(predictions, axis=0)
        return y_predicted_cls
