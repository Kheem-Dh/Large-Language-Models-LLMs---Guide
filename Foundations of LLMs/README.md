# 2. Foundations of LLMs: Introduction to Deep Learning

Deep Learning is a subset of machine learning that specializes in using multi-layer neural networks to automatically learn representations of data. Unlike traditional machine learning, deep learning models are capable of automatic feature extraction from raw data, which often results in higher performance for complex tasks.

---

## 2.1 Introduction to Deep Learning

### What is Deep Learning?

Deep Learning models consist of multiple layers of interconnected nodes, commonly known as neurons. Each connection has an associated weight, which is adjusted during training. By stacking multiple such layers (hence the term "deep"), these models can learn complex, hierarchical features from the input data.

### Key Concepts

#### 1. Activation Functions

An activation function determines the output of a neuron based on its input. It introduces non-linearity into the model, enabling it to learn from error and make predictions.

Common Activation Functions:
- **ReLU (Rectified Linear Unit)**: 
  \( f(x) = \max(0, x) \)
- **Sigmoid**: 
  \( f(x) = \frac{1}{1 + e^{-x}} \)
- **Tanh**: 
  \( f(x) = \frac{2}{1 + e^{-2x}} - 1 \)

#### 2. Layers

- **Input Layer**: Takes in the features and passes them to the next layer.
- **Hidden Layers**: Layers between the input and output. They process the features and pass the result to the next layer.
- **Output Layer**: Provides the final prediction or classification.

#### 3. Loss Functions

A loss function measures the difference between the predicted values and actual values. During training, the model aims to minimize this loss.

Common Loss Functions:
- **Mean Squared Error (MSE)**: Used for regression tasks.
  \( \text{MSE} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2 \)
- **Cross Entropy Loss**: Used for classification tasks.
  \( H(y, \hat{y}) = -\sum_{i} y_i \log(\hat{y}_i) \)

#### 4. Optimization Techniques

Optimizers adjust the model's weights based on the loss to improve the model's predictions. 

Common Optimizers:
- **Gradient Descent**: Updates weights in the direction of the steepest decrease of the loss.
- **Stochastic Gradient Descent (SGD)**: A variant of gradient descent that updates weights using only a single data point.
- **Adam**: Combines the best properties of other optimization algorithms to provide faster convergence.

---


