import numpy as np
import argparse

def sigmoid(x: np.ndarray):
    """
    Implementation of the sigmoid function.

    Parameters:
        x (np.ndarray): Input np.ndarray.

    Returns:
        np.ndarray after applying the sigmoid function element-wise to the input.
    """
    return 1 / (1 + np.exp(-x))


def train_sgd(
    theta: np.ndarray,  # shape (D,) where D is feature dimension
    X: np.ndarray,      # shape (N, D) where N is number of examples
    y: np.ndarray,      # shape (N,)
    num_epoch: int,
    learning_rate: float
) -> None:
    """
    Train the logistic regression model using Stochastic Gradient Descent (SGD).

    Parameters:
        theta (np.ndarray): Initial model parameters.
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Labels.
        num_epoch (int): Number of training epochs.
        learning_rate (float): Learning rate for SGD.
    """
    N = X.shape[0]  # Number of examples
    for epoch in range(num_epoch):
        for i in range(N):
            # Compute the raw prediction value for the ith example
            prediction = sigmoid(np.dot(X[i], theta))  # shape ()
            
            # Compute the error for this single example
            error = y[i] - prediction  # shape ()
            
            # Compute the gradient for this single example
            gradient = X[i] * error  # shape (D,)
            
            # Update theta (parameters) using SGD
            theta += learning_rate * gradient  # Correct update rule


def predict(
    theta: np.ndarray,
    X: np.ndarray
) -> np.ndarray:
    """
    Make predictions using the logistic regression model.

    Parameters:
        theta (np.ndarray): Model parameters.
        X (np.ndarray): Feature matrix.

    Returns:
        np.ndarray: Predicted labels (0 or 1).
    """
    predictions = sigmoid(np.dot(X, theta))  # shape (N,)
    return np.round(predictions)  # Round predictions to get 0 or 1


def compute_error(
    y_pred: np.ndarray,
    y: np.ndarray
) -> float:
    """
    Compute the error rate.

    Parameters:
        y_pred (np.ndarray): Predicted labels.
        y (np.ndarray): True labels.

    Returns:
        float: Error rate.
    """
    return np.mean(y_pred != y)  # Compute error rate


if __name__ == '__main__':
    # Command-line argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("train_input", type=str, help='path to formatted training data')
    parser.add_argument("validation_input", type=str, help='path to formatted validation data')
    parser.add_argument("test_input", type=str, help='path to formatted test data')
    parser.add_argument("train_out", type=str, help='file to write train predictions to')
    parser.add_argument("test_out", type=str, help='file to write test predictions to')
    parser.add_argument("metrics_out", type=str, help='file to write metrics to')
    parser.add_argument("num_epoch", type=int, help='number of epochs of stochastic gradient descent to run')
    parser.add_argument("learning_rate", type=float, help='learning rate for stochastic gradient descent')
    args = parser.parse_args()

    # Load training data
    train_data = np.loadtxt(args.train_input, delimiter='\t')
    X_train = train_data[:, 1:]  # Features
    y_train = train_data[:, 0]   # Labels

    # Initialize theta (weights)
    D = X_train.shape[1]  # Number of features
    theta = np.zeros(D)    # Initialize parameters to zero

    # Train the model using SGD
    train_sgd(theta, X_train, y_train, args.num_epoch, args.learning_rate)

    # Make predictions on training data
    train_predictions = predict(theta, X_train)

    # Save training predictions
    np.savetxt(args.train_out, train_predictions, fmt='%d')

    # Load test data
    test_data = np.loadtxt(args.test_input, delimiter='\t')
    X_test = test_data[:, 1:]  # Features
    y_test = test_data[:, 0]   # Labels

    # Make predictions on test data
    test_predictions = predict(theta, X_test)

    # Save test predictions
    np.savetxt(args.test_out, test_predictions, fmt='%d')

    # Compute and save metrics
    train_error = compute_error(train_predictions, y_train)
    test_error = compute_error(test_predictions, y_test)

    # Write the error metrics to a file
    with open(args.metrics_out, 'w') as f:
        f.write(f"error(train): {train_error:.6f}\n")
        f.write(f"error(test): {test_error:.6f}\n")

