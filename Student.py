import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- STREAMLIT CONFIG ----------------
st.set_page_config(page_title="MLP Trainer", layout="centered")
st.title("🧠 MLP Trainer Interface")
st.write("Train a neural network from scratch with custom architecture & activations.")

# ---------------- DATA LOADING ----------------
@st.cache_data
def load_data(csv_file):
    df = pd.read_csv(csv_file)

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values.reshape(-1, 1)

    # Feature scaling
    X = (X - X.mean(axis=0)) / X.std(axis=0)

    return X, y, df

st.subheader("📂 Upload Dataset")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is None:
    st.warning("Please upload a CSV file to continue.")
    st.stop()

X, y, df = load_data(uploaded_file)
st.success("Dataset loaded successfully!")
st.dataframe(df.head())

# ---------------- ACTIVATION FUNCTIONS ----------------

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - x**2

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)

def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def elu_derivative(x, alpha=1.0):
    return np.where(x > 0, 1, alpha * np.exp(x))

def softplus(x):
    return np.log(1 + np.exp(x))

def softplus_derivative(x):
    return sigmoid(x)

def swish(x):
    return x * sigmoid(x)

def swish_derivative(x):
    sig = sigmoid(x)
    return sig + x * sig * (1 - sig)

def gelu(x):
    return 0.5 * x * (1 + np.tanh(
        np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)
    ))

def gelu_derivative(x):
    return 0.5 * (1 + np.tanh(
        np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)
    ))

def linear(x):
    return x

def linear_derivative(x):
    return np.ones_like(x)

# ---------------- ACTIVATION MAP ----------------
activation_map = {
    "Sigmoid": (sigmoid, sigmoid_derivative),
    "Tanh": (tanh, tanh_derivative),
    "ReLU": (relu, relu_derivative),
    "Leaky ReLU": (leaky_relu, leaky_relu_derivative),
    "ELU": (elu, elu_derivative),
    "Softplus": (softplus, softplus_derivative),
    "Swish": (swish, swish_derivative),
    "GELU": (gelu, gelu_derivative),
    "Linear": (linear, linear_derivative)
}

# ---------------- HYPERPARAMETERS ----------------
st.subheader("⚙ Training Parameters")

lr = st.number_input("Learning Rate", 0.0001, 1.0, 0.01, step=0.001)
epochs = st.number_input("Epochs", 10, 5000, 500, step=50)

activation_name = st.selectbox("Hidden Layer Activation", list(activation_map.keys()))
activation, activation_derivative = activation_map[activation_name]

num_hidden_layers = st.slider("Number of Hidden Layers", 1, 3, 1)
neurons_per_layer = st.slider("Neurons per Hidden Layer", 2, 12, 4)

# ---------------- MLP TRAINING ----------------
def train_mlp(X, y, lr, epochs, hidden_layers, neurons, activation, activation_derivative):
    np.random.seed(42)

    layers = [X.shape[1]] + [neurons] * hidden_layers + [1]

    weights = []
    biases = []

    for i in range(len(layers) - 1):
        weights.append(np.random.randn(layers[i], layers[i+1]) * 0.1)
        biases.append(np.zeros((1, layers[i+1])))

    for _ in range(epochs):
        # ---------- Forward Pass ----------
        activations = [X]

        for i in range(len(weights) - 1):
            z = activations[-1] @ weights[i] + biases[i]
            activations.append(activation(z))

        z_out = activations[-1] @ weights[-1] + biases[-1]
        y_pred = sigmoid(z_out)
        activations.append(y_pred)

        # ---------- Backpropagation ----------
        error = y - y_pred
        delta = error * sigmoid_derivative(y_pred)

        for i in reversed(range(len(weights))):
            dw = activations[i].T @ delta
            db = np.sum(delta, axis=0, keepdims=True)

            weights[i] += lr * dw
            biases[i] += lr * db

            if i != 0:
                delta = (delta @ weights[i].T) * activation_derivative(activations[i])

    return weights, biases

# ---------------- TRAIN BUTTON ----------------
if st.button("🚀 Train Model"):
    weights, biases = train_mlp(
        X, y, lr, epochs,
        num_hidden_layers,
        neurons_per_layer,
        activation,
        activation_derivative
    )

    st.success("Training completed!")

    # ---------------- DECISION BOUNDARY ----------------
    fig, ax = plt.subplots()

    ax.scatter(
        X[:, 0], X[:, 1],
        c=y.flatten(),
        cmap="RdYlGn",
        edgecolors="black"
    )

    xx, yy = np.meshgrid(
        np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 200),
        np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 200)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    a = grid

    for i in range(len(weights) - 1):
        a = activation(a @ weights[i] + biases[i])

    Z = sigmoid(a @ weights[-1] + biases[-1])
    Z = Z.reshape(xx.shape)

    ax.contour(xx, yy, Z, levels=[0.5], colors="blue")
    ax.set_title("Decision Boundary")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")

    st.pyplot(fig)
