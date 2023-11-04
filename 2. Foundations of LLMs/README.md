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
### 2.2 Basics of Neural Networks

Neural networks form the backbone of deep learning, providing the framework for constructing complex models that can recognize patterns and make decisions.

#### What are Neural Networks?

Neural networks are computational models inspired by the human brain, consisting of interconnected units called neurons. These neurons receive input, process it, and transmit the output to other neurons. This structure allows neural networks to learn from data by adjusting the connections (weights) between neurons.

#### Key Components

1. **Neuron**: The basic unit of computation in a neural network.
2. **Weight**: A connection strength between two neurons, adjusted during training.
3. **Bias**: An additional parameter associated with each neuron that allows the model to fit the training data better.

#### Network Architectures

1. **Single Layer Perceptron (SLP)**: The simplest form of a neural network with no hidden layers.
2. **Multi-Layer Perceptron (MLP)**: A network with one or more hidden layers between the input and output layers.
3. **Convolutional Neural Networks (CNNs)**: Specialized for processing structured grid data such as images.
4. **Recurrent Neural Networks (RNNs)**: Designed to handle sequential data, with neurons that have feedback connections.

#### Training Neural Networks

Training a neural network involves adjusting the weights and biases of the network to minimize the difference between the predicted output and the actual output.

1. **Forward Propagation**: The process of passing input data through the network to get the output.
2. **Backpropagation**: The method of calculating the gradient of the loss function with respect to each weight by the chain rule, moving the error backward through the network.
3. **Gradient Descent**: An optimization algorithm that iteratively moves the weights in the direction that reduces the loss.

#### Challenges in Training

1. **Overfitting**: When a model learns the training data too well, including the noise.
2. **Underfitting**: When a model is too simple to learn the underlying pattern of the data.
3. **Vanishing/Exploding Gradients**: When gradients become too small or too large, hindering the network's training.

### Next Steps

Having understood the basics of neural networks, we will next explore how they can be structured to handle sequences in the context of language modeling. Stay tuned for the upcoming sections on Sequence-to-Sequence Models and Transformers and Attention Mechanisms.

---


