# 🧠 Neural Network (MLP) Simulation with Interactive Visualization

An interactive Streamlit-based Machine Learning project that demonstrates how a Multi-Layer Perceptron (MLP) learns from data by allowing users to experiment with hyperparameters, visualize loss convergence, observe decision boundary changes, and make real-time predictions on a student performance dataset.

---

## 🚀 Project Motivation

Understanding how neural networks actually learn is often difficult because training happens behind the scenes.

This project is designed to:
- Visually explain how an MLP works
- Show how hyperparameters affect learning
- Help beginners build intuition around loss reduction and decision boundaries
- Bridge the gap between theory and practice in Machine Learning

Instead of treating neural networks as black boxes, this project turns them into interactive learning tools.

---

## 📌 What This Project Does

- Trains a custom-built Multi-Layer Perceptron from scratch using NumPy
- Uses a dummy student dataset (`student.csv`) for binary classification (PASS / FAIL)
- Allows users to dynamically:
  - Change learning rate
  - Change number of hidden layers
  - Change neurons per layer
  - Change activation functions
- Visualizes:
  - Loss vs Epochs
  - Decision boundary for 2D feature datasets
- Enables users to:
  - Enter custom feature values
  - Get prediction probability
  - See final classification as PASS or FAIL

---

## 🧪 Key Features

### ⚙️ Hyperparameter Control
Users can tune:
- Learning Rate
- Epochs
- Number of Hidden Layers
- Neurons per Hidden Layer
- Activation Function (Linear, ReLU, Leaky ReLU, Tanh, Sigmoid)

This helps users observe how model performance changes in real time.

---

### 📉 Loss Visualization
- Displays how loss decreases over epochs
- Helps understand convergence behavior
- Makes underfitting and overfitting patterns easier to spot

---

### 📊 Decision Boundary Visualization
- For datasets with two features, the app:
  - Plots original data points
  - Draws the learned decision boundary
- Users can visually see how different parameters reshape the boundary

---

### 🔮 Real-Time Predictions
- Users can input custom feature values
- The trained model:
  - Computes prediction probability
  - Classifies the result as PASS or FAIL

---

## 🛠️ Tech Stack Used

- Python
- NumPy
- Pandas
- Matplotlib
- Streamlit

Note: No deep learning frameworks like TensorFlow or PyTorch were used. The neural network is implemented from scratch to enhance conceptual understanding.

---

## 🧠 Model Architecture (High-Level)

- Input Layer → Number of features
- Hidden Layers → Fully connected layers with user-selected activation functions
- Output Layer → Single neuron with Sigmoid activation
- Loss Function → Mean Squared Error (MSE)
- Optimization → Gradient Descent with Backpropagation

---

## 📂 Dataset Details

- Dataset: student.csv
- Type: Dummy / Educational dataset[Attached in this repository]
- Target: Binary classification (PASS = 1, FAIL = 0)
- Preprocessing:
  - Feature standardization using mean and standard deviation

---

## ▶️ How to Run the Project

### Step 1: Clone the Repository
```bash
git clone https://github.com/KashishPundir/neural-network-simulation.git
cd neural-network-simulation
```

### Step 2: Install all mentioned dependencies

### Step 3: Run the Streamlit Application:
``` bash
streamlit run app.py
```

## 🧑‍💻 Author

Kashish Pundir

B.Tech CSE (Data Science)

Aspiring Data Scientist | Machine Learning Enthusiast


### This project focuses on building intuition rather than maximizing accuracy.
If you found this project helpful, consider starring ⭐ the repository.
