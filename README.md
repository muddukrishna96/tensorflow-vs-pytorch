
# Image Classification Neural Network (tensorflow-vs-pytorch)

This repository contains two implementations of a simple image classification neural network using TensorFlow and PyTorch.

## Requirements

- Python 3.x
- TensorFlow (for TensorFlow implementation)
- PyTorch (for PyTorch implementation)

## Usage

### TensorFlow

1. Install the required packages: `pip install tensorflow`
2. Run the script: `python tensorflow_impl.py`

### PyTorch

1. Install the required packages: `pip install torch`
2. Run the script: `python pytorch_impl.py`

## Model Details

- Input shape: 256x256x3
- Convolutional layers:
  - TensorFlow: 32 filters of size 3x3 and 64 filters of size 3x3, both with ReLU activation
  - PyTorch: 32 filters of size 3x3 and 64 filters of size 3x3, both with ReLU activation
- Pooling layers:
  - TensorFlow: Max pooling with 2x2 pool size
  - PyTorch: Max pooling with 2x2 pool size
- Hidden layers:
  - TensorFlow: Two fully connected layers with 128 and 64 units, both with ReLU activation
  - PyTorch: Three fully connected layers with 128, 64, and 10 units, all with ReLU activation
- Output layer:
  - TensorFlow: Fully connected layer with 10 units and softmax activation
  - PyTorch: Fully connected layer with 10 units

## Training and Evaluation

### TensorFlow

- Optimizer: Adam
- Loss function: Sparse categorical cross-entropy
- Metrics: Accuracy

### PyTorch

- Optimizer: Adam
- Loss function: Cross-entropy
- Metrics: Accuracy

