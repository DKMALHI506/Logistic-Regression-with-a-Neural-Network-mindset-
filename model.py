# model.py
# Logistic Regression with Neural Network Mindset
import numpy as np

# Sigmoid function
def sigmoid(z):
    """Compute the sigmoid of z"""
    return 1 / (1 + np.exp(-z))

# Initialize weights and bias
def initialize_with_zeros(dim):
    """Returns w vector of zeros and b = 0"""
    w = np.zeros((dim, 1))
    b = 0.0
    return w, b

# Forward and backward propagation
def propagate(w, b, X, Y):
    """Compute cost and gradients"""
    m = X.shape[1]

    # Forward propagation
    A = sigmoid(np.dot(w.T, X) + b)
    cost = (-1/m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))

    # Backward propagation
    dw = (1/m) * np.dot(X, (A - Y).T)
    db = (1/m) * np.sum(A - Y)

    grads = {"dw": dw, "db": db}
    return grads, cost

# Gradient descent optimization
def optimize(w, b, X, Y, num_iterations=2000, learning_rate=0.5, print_cost=False):
    """Optimize w and b using gradient descent"""
    costs = []

    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)
        w -= learning_rate * grads["dw"]
        b -= learning_rate * grads["db"]

        if i % 100 == 0:
            costs.append(cost)
            if print_cost:
                print(f"Cost after iteration {i}: {cost}")

    params = {"w": w, "b": b}
    grads = {"dw": grads["dw"], "db": grads["db"]}
    return params, grads, costs

# Prediction
def predict(w, b, X):
    """Predict 0/1 for X using learned weights"""
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    A = sigmoid(np.dot(w.T, X) + b)

    for i in range(A.shape[1]):
        Y_prediction[0, i] = 1 if A[0, i] > 0.5 else 0

    return Y_prediction

# Full model
def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    """Builds the logistic regression model"""
    w, b = initialize_with_zeros(X_train.shape[0])
    params, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    w, b = params["w"], params["b"]

    Y_prediction_train = predict(w, b, X_train)
    Y_prediction_test = predict(w, b, X_test)

    if print_cost:
        print("Train accuracy: {:.2f}%".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
        print("Test accuracy: {:.2f}%".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    return {"costs": costs, "Y_prediction_train": Y_prediction_train,
            "Y_prediction_test": Y_prediction_test, "w": w, "b": b,
            "learning_rate": learning_rate, "num_iterations": num_iterations}
