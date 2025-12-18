import os
import numpy as np
import cv2


def normalize_inputs(inputs, mean, sd):
    m = inputs.shape[0]
    inputs = inputs.reshape(m, -1)
    # inputs now have size (m, number of features)
    if mean is None and sd is None:
        mean_inputs = 1 / m * np.sum(inputs, axis = 0, keepdims = True)
        inputs = inputs - mean_inputs
        sd_inputs = 1e-7 + np.sqrt(1 / m * np.sum(inputs * inputs, axis = 0, keepdims = True))
        inputs = inputs / sd_inputs
        return inputs.T, mean_inputs, sd_inputs
    else:
        inputs = (inputs - mean) / sd
        return inputs.T


def retrieve_data():
    # Return training sets and validation sets
    path = os.path.join(os.getcwd(), "horse-or-human")
    training_imgs = []
    training_labels = []
    validation_imgs = []
    validation_labels = []
    for dir in os.listdir(path=path):
        if dir == "horse-or-human":
            continue
        new_path = os.path.join(path, dir)
        for sub_dir in os.listdir(new_path):
            more_new_path = os.path.join(new_path, sub_dir)
            print(more_new_path)
            cnt = 0
            for img in os.listdir(more_new_path):
                if cnt == 200:
                    break
                cnt += 1
                # Convert img into np arr
                img_path = os.path.join(more_new_path, img)
                img = cv2.imread(img_path)
                img = cv2.resize(img, (150, 150))

                # Add the correponsding labels for imgs
                if sub_dir == "horses":
                    label = 0
                else:
                    label = 1
                if dir == "train":
                    training_imgs.append(img)
                    training_labels.append(label)
                elif dir == "validation":
                    validation_imgs.append(img)
                    validation_labels.append(label)

    # Convert lists to NumPy arrays
    validation_imgs = np.array(validation_imgs)
    validation_labels = np.array(validation_labels)
    training_imgs = np.array(training_imgs)
    training_labels = np.array(training_labels)

    m = training_imgs.shape[0]
    n0 = training_imgs.reshape(m, -1).shape[1]

    # normalising inputs
    processed_training_imgs, mean, sd = normalize_inputs(training_imgs, None, None)
    processed_validation_imgs = normalize_inputs(validation_imgs, mean, sd)
    print("Finished uploading")

    return n0, processed_training_imgs, training_labels, processed_validation_imgs, validation_labels     


def multilayers_NN(L, L_structure, n, learning_rate, processed_train_imgs, train_labels, processed_valid_imgs, valid_labels):
    m = processed_train_imgs.shape[1]
    # Initialise weights and biases
    W, b = he_initialize(L, L_structure)
    # Repeat the traing n times
    for i in range(0, n):
        # Forward propagation
        A, Z = forward_prop(W, b, L, processed_train_imgs)
        # BCE loss calculation
        loss = bce_loss(A[L], train_labels.reshape(1, m))
        # Backward propagation
        d = back_prop(L, W, Z, A, train_labels.reshape(1, m))
        # Update weights and biases
        W, b = update(L, W, b, d, learning_rate)
        tmp = None
        if i == 0:
            tmp = "st"
        elif i == 1: 
            tmp = "nd"
        elif i == 2:
            tmp = "rd"
        else:
            tmp = "th"
        print(f"{i + 1}{tmp} training. Loss = {loss}")
    print("Finished training")

    # Check the accuracy of the model
    tmp, rand = forward_prop(W, b, L, processed_train_imgs)
    res = np.where(tmp[L] > 0.5, 1, 0) - train_labels.reshape(1, m)
    percent = np.count_nonzero(res == 0) / m * 100
    print(f"Accuracy for training set: {percent}")

    M = processed_valid_imgs.shape[1]
    tmp, rand = forward_prop(W, b, L, processed_valid_imgs)
    res = np.where(tmp[L] > 0.5, 1, 0) - valid_labels.reshape(1, M)
    percent = np.count_nonzero(res == 0) / M * 100
    print(f"Accuracy for dev set: {percent}")


def he_initialize(L, L_structure):
    W = [None, ]
    b = [None, ]
    # L_structure: an array of number of nodes in each layer from 0 to L - 1
    for i in range(1, L + 1):
        # Wi = np.random.randn(L_structure[i], L_structure[i - 1]) * 0.01
        # He initialisation
        Wi = np.random.randn(L_structure[i], L_structure[i - 1]) * np.sqrt(2 / L_structure[i - 1])
        W.append(Wi)
        bi = np.zeros((L_structure[i], 1))
        b.append(bi)
    return W, b


# def relu(X):
    # return np.maximum(0, X)


# def relu_derivative(X):
    # return np.where(X > 0, 1, 0)


def leaky_relu(x):
    return np.where(x > 0, x, 0.01 * x)


def leaky_relu_derivative(x):
    return np.where(x > 0, 1, 0.01)


def sigmoid(X):
    return 1 / (1 +  np.exp(-X))


def sigmoid_derivative(X):
    return sigmoid(X) * (1 - sigmoid(X))


def bce_loss(predicted_values, true_values):
    # both inputs have size (1, m)
    m = predicted_values.shape[1]
    return - 1 / m * np.sum((true_values * np.log(predicted_values) + (1 - true_values) * np.log(1 - predicted_values)), axis = 1)


def forward_prop(W, b, L, input):
    # input has this shape: (n0, m)
    A = [input, ]
    Z = [None, ]
    # Add A1 with this shape: (n1, m)
    Z.append(np.dot(W[1], input) + b[1])
    A.append(leaky_relu(Z[1]))

    for i in range (2, L):
        Z.append(np.dot(W[i], A[i - 1]) + b[i])
        A.append(leaky_relu(Z[i]))

    Z.append(np.dot(W[L], A[L - 1]) + b[L])
    A.append(sigmoid(Z[L]))
    return A, Z


def back_prop(L, W, Z, A, expected_output):
    # expected_output has size: (1, m)
    m = A[L].shape[1]
    d = {}
    # MSE loss: d[f"A{L}"] = A[L] - expected_output; d[f"Z{L}"] = d[f"A{L}"] * sigmoid_derivative(Z[L])
    # BCE loss:
    d[f"Z{L}"] = A[L] - expected_output
    d[f"W{L}"] = 1 / m * np.dot(d[f"Z{L}"], A[L - 1].T)
    d[f"b{L}"] = 1 / m * np.sum(d[f"Z{L}"], axis=1, keepdims=True)
    d[f"A{L - 1}"] = np.dot(W[L].T, d[f"Z{L}"])
    for i in range(1, L):
        d[f"Z{L - i}"] = d[f"A{L - i}"] * leaky_relu_derivative(Z[L - i])
        d[f"W{L - i}"] = 1 / m * np.dot(d[f"Z{L - i}"], A[L - i - 1].T)
        d[f"b{L - i}"] = 1 / m * np.sum(d[f"Z{L - i}"], axis=1, keepdims=True)
        d[f"A{L - i - 1}"] = np.dot(W[L - i].T, d[f"Z{L - i}"])
    return d


def update(L, W, b, d, learning_rate):
    for i in range(1, L + 1):
        W[i] = W[i] - learning_rate * d[f"W{i}"]
        b[i] = b[i] - learning_rate * d[f"b{i}"]
    return W, b


def main():
    n0, a, b, c, d = retrieve_data()
    NN = multilayers_NN(5, (n0, 64, 32, 16, 8, 1), 50, 0.001, a, b, c, d)


main()
