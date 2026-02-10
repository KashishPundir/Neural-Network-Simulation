import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# ---------------- FIX RANDOMNESS ----------------
np.random.seed(42)

# ---------------- STREAMLIT CONFIG ----------------
st.set_page_config(page_title="Neural Network Simulation", layout="centered")
st.title("🧠 Neural Network Simulation")
st.write("Train a neural network from scratch, evaluate correctly, and make reliable predictions.")

# ---------------- DATA LOADING ----------------
@st.cache_data
def load_data(csv_file):
    df = pd.read_csv(csv_file)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values.reshape(-1, 1)

    mean = X.mean(axis=0)
    std = X.std(axis=0)
    X = (X - mean) / std

    return X, y, df, mean, std

uploaded_file = st.file_uploader("📂 Upload CSV file", type=["csv"])
if uploaded_file is None:
    st.stop()

X, y, df, X_mean, X_std = load_data(uploaded_file)
st.success("Dataset Loaded")
st.dataframe(df.head())

# ---------------- TRAIN / TEST SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ---------------- ACTIVATION FUNCTIONS ----------------
def relu(x): return np.maximum(0, x)
def relu_d(x): return (x > 0).astype(float)

def leaky_relu(x): return np.where(x > 0, x, 0.01 * x)
def leaky_relu_d(x): return np.where(x > 0, 1, 0.01)

def tanh(x): return np.tanh(x)
def tanh_d(x): return 1 - np.tanh(x) ** 2

def sigmoid(x): return 1 / (1 + np.exp(-x))
def sigmoid_d(x): return x * (1 - x)

def linear(x): return x
def linear_d(x): return np.ones_like(x)

activation_map = {
    "ReLU": (relu, relu_d),
    "Leaky ReLU": (leaky_relu, leaky_relu_d),
    "Tanh": (tanh, tanh_d),
    "Sigmoid": (sigmoid, sigmoid_d),
    "Linear": (linear, linear_d)
}

# ---------------- HYPERPARAMETERS ----------------
st.subheader("⚙ Training Parameters")

lr = st.slider("Learning Rate", 0.001, 0.3, 0.05)
epochs = st.slider("Epochs", 100, 3000, 500)
num_hidden_layers = st.slider("Hidden Layers", 1, 4, 2)
neurons = st.slider("Neurons per Hidden Layer", 2, 12, 6)

act_name = st.selectbox("Activation Function", activation_map.keys())
activation, activation_d = activation_map[act_name]

# ---------------- TRAINING FUNCTION ----------------
def train_nn(X, y):
    layers = [X.shape[1]] + [neurons] * num_hidden_layers + [1]

    weights, biases = [], []

    for i in range(len(layers) - 1):
        weights.append(np.random.randn(layers[i], layers[i + 1]) * 0.1)
        biases.append(np.zeros((1, layers[i + 1])))

    loss_history = []

    for _ in range(epochs):

        # ---- Forward Propagation ----
        activations = [X]
        pre_activations = []

        for i in range(len(weights) - 1):
            z = activations[-1] @ weights[i] + biases[i]
            pre_activations.append(z)
            activations.append(activation(z))

        z_out = activations[-1] @ weights[-1] + biases[-1]
        y_pred = sigmoid(z_out)

        # ---- Loss ----
        loss = np.mean((y - y_pred) ** 2)
        loss_history.append(loss)

        # ---- Backward Propagation ----
        delta = (y - y_pred) * sigmoid_d(y_pred)

        for i in reversed(range(len(weights))):
            weights[i] += lr * activations[i].T @ delta
            biases[i] += lr * np.sum(delta, axis=0, keepdims=True)

            if i != 0:
                delta = (delta @ weights[i].T) * activation_d(pre_activations[i - 1])

    return weights, biases, loss_history

# ---------------- FORWARD PASS ----------------
def forward_pass(X, weights, biases):
    a = X
    for i in range(len(weights) - 1):
        a = activation(a @ weights[i] + biases[i])
    return sigmoid(a @ weights[-1] + biases[-1])

# ---------------- SESSION STATE ----------------
if "trained" not in st.session_state:
    st.session_state.trained = False

# ---------------- TRAIN BUTTON ----------------
if st.button("🚀 Train Model"):
    weights, biases, loss_history = train_nn(X_train, y_train)

    st.session_state.weights = weights
    st.session_state.biases = biases
    st.session_state.loss_history = loss_history
    st.session_state.trained = True

    st.success("Training Completed")

# ---------------- RESULTS ----------------
if st.session_state.trained:
    weights = st.session_state.weights
    biases = st.session_state.biases
    loss_history = st.session_state.loss_history

    # ---- LOSS CURVE ----
    fig, ax = plt.subplots()
    ax.plot(loss_history)
    ax.set_title("Loss vs Epochs")
    st.pyplot(fig)

    # ---- TRAIN & TEST ACCURACY ----
    train_acc = np.mean((forward_pass(X_train, weights, biases) >= 0.5) == y_train)
    test_acc = np.mean((forward_pass(X_test, weights, biases) >= 0.5) == y_test)

    st.subheader("📊 Model Performance")
    st.write(f"Train Accuracy: {train_acc * 100:.2f}%")
    st.write(f"Test Accuracy: {test_acc * 100:.2f}%")

    # ---- TRUE DECISION BOUNDARY (2 FEATURES ONLY) ----
    if X.shape[1] == 2:
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 200),
            np.linspace(y_min, y_max, 200)
        )

        grid = np.c_[xx.ravel(), yy.ravel()]
        Z = forward_pass(grid, weights, biases).reshape(xx.shape)

        fig, ax = plt.subplots()
        ax.contourf(xx, yy, Z, levels=50, cmap="RdYlGn", alpha=0.6)
        ax.scatter(X[:, 0], X[:, 1], c=y.flatten(), edgecolors="black", cmap="RdYlGn")
        ax.set_title("True Neural Network Decision Boundary")
        st.pyplot(fig)

    # ---- USER PREDICTION ----
    st.subheader("🔮 Make a Prediction")

    feature_names = df.columns[:-1].tolist()
    user_input = []

    for i, feature in enumerate(feature_names):
        user_input.append(st.number_input(f"Enter {feature}", key=f"feat_{i}"))

    if st.button("Predict"):
        user_input = np.array(user_input).reshape(1, -1)
        user_input = (user_input - X_mean) / X_std

        prob = forward_pass(user_input, weights, biases)[0, 0]
        pred = "PASS" if prob > 0.06 else "FAIL"

        st.success(f"Predicted Probability: {prob:.4f}")
        st.success(f"Prediction: {pred}")
