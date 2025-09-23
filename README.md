# Grad - A Minimal Neural Network Framework with Automatic Differentiation

A lightweight implementation of a neural network framework built from scratch in Python, featuring automatic differentiation (autograd) capabilities inspired by PyTorch's computational graph system.

## 🚀 Features

- **Automatic Differentiation**: Custom `Value` class that tracks gradients through computational graphs
- **Neural Network Components**: Complete implementation of neurons, layers, and multi-layer perceptrons (MLPs)
- **Backpropagation**: Efficient gradient computation using depth-first search
- **Visualization**: Graphical representation of computational graphs using Graphviz
- **Mathematical Operations**: Support for addition, multiplication, subtraction, division, power, and hyperbolic tangent

## 📁 Project Structure

```
Grad/
├── playground.ipynb    # Main implementation and demonstration
├── engine.py          # Core Value class implementation (if extracted)
├── visualize.py       # Visualization utilities (if extracted)
└── README.md          # This file
```


## 📦 Installation

### Clone the Repository
```bash
git clone https://github.com/trabelsiaziz/Grad.git
cd Grad
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Usage
1. **For interactive exploration**: Open `playground.ipynb` in Jupyter Notebook
```bash
jupyter notebook playground.ipynb
```

2. **For programmatic use**: Import the classes from the Python files
```python
from engine import Value, MLP
from visualize import draw_dot
```

## 🧠 Core Components

### Value Class

The foundation of the automatic differentiation system. Each `Value` object stores:

- `data`: The actual numerical value
- `grad`: The gradient of this value
- `_children`: Set of parent nodes in the computational graph
- `_op`: The operation that created this value
- `_backward`: Function to compute gradients during backpropagation

### Neural Network Architecture

#### Neuron

- Individual processing unit with weights and bias
- Applies weighted sum followed by tanh activation
- Randomly initialized weights between -1 and 1

#### Layer

- Collection of neurons with shared input
- Supports multiple neurons per layer
- Parameter extraction for training

#### MLP (Multi-Layer Perceptron)

- Complete neural network with multiple layers
- Built-in training loop with gradient descent
- Configurable architecture and learning rate

## 💻 Usage Examples

### Basic Value Operations

```python
from engine import Value
from engine import MLP

# Create values
a = Value(2.0)
b = Value(3.0)

# Perform operations
c = a + b
d = a * b
e = c.tanh()

# Compute gradients
e.grad = 1.0
MLP.Backprop(e)
print(f"Gradient of a: {a.grad}")
```

### Building and Training a Neural Network

```python
from engine import Value, MLP
import random

# Create training data
x = [Value(random.uniform(-1,1)) for _ in range(2)]
y = [Value(random.uniform(-1,1)) for _ in range(2)]

# Create MLP: 2 inputs, hidden layer with 2 neurons, output layer with 2 neurons
nn = MLP(input_size=2, layer_sizes=[2, 2], learning_rate=0.1)

# Train the network
epochs = 10
nn.Train(input=x, labels=y, epochs=epochs)
```

### Visualizing Computational Graphs
run this in jupyter notebook :

```python
from engine import Value
from visualize import draw_dot

# Create values
a = Value(2.0)
b = Value(3.0)

# Perform operations
c = a + b
d = a * b
e = c.tanh()

# Visualize the computational graph
draw_dot(e)
```

**📓 More Examples**: For additional examples and interactive demonstrations, check out `playground.ipynb` 

## 🎯 Example Output

```
Loss for the 0'th epoch is : 0.8234
Loss for the 1'th epoch is : 0.7891
Loss for the 2'th epoch is : 0.7523
...
Loss for the 9'th epoch is : 0.4567
```



**Note**: This is an educational implementation designed for learning purposes. 
