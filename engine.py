
"""
suppose you have this expression : 
x = b + d 
f = a * x + c 

the tree is gonna look something like this : 


 +-------------------+f +--------------------+  
 |                   ++-+                    |  
 |                    |                      |  
 |                    |                      |  
 v                    v                      v  
+--+      +--+      +---+       +----+     +---+
| a|      |* |  +---+ x +---+   |  + |     | c |
+--+      +--+  |   +-+-+   |   +----+     +---+
                |     |     |                   
                |     |     |                   
                |     |     |                   
                v     v     v                   
              +--+   +--+  +--+                 
              |b |   |+ |  | d|                 
              +--+   +--+  +--+                 



f._children = {a, x, c}

**grad: 
s --> c ---> f 

ds/df = ds/dc * dc/df 

"""

import math 
import random

class Value:
    def __init__(self, value):
        self.data = value
        self.grad = 0.0
        self._children = set()
        self._op = ""
        self.label = ""
        self._backward = lambda : None
            
    def __repr__(self):
        out = f"Value(data={self.data:.4f}, grad={self.grad:.4f}, children={self._children})"
        return out 

    def __add__(self, other):
        res = self.data + other.data
        out = Value(res)
        out._children.update([self, other])  
        out._op = "+"    
        out.label = "$"
        def compute_grad(): 
            for child in out._children:
                child.grad += out.grad
            # self.grad += out.grad
            # other.grad += out.grad
        
        out._backward = compute_grad 
        return out
    
    def __mul__(self, other): 
        res = self.data * other.data
        out = Value(res)
        out._op = "*"
        out._children.update([self, other])
        out.label = "$"    
        def compute_grad(): 
            total_power = 1.0
            for child in out._children:
                total_power *= child.data
            for child in out._children: 
                child.grad += out.grad * (total_power / child.data) 
            # self.grad += out.grad * other.data
            # other.grad += out.grad * self.data

        out._backward = compute_grad
        return out 
    
    def __rmul__(self, other):
        return self * other
        
    def tanh(self): 
        t = (math.exp(2*self.data) - 1)/(math.exp(2*self.data) + 1)
        out = Value(t)
        out._children = {self}
        out._op = "tanh"
        
        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        
        return out
    
    def __neg__(self):
        out = self * Value(-1) 
        return out
    
    def __sub__(self, other): 
        out = self + (-other)
        return out 
    
    
    def __pow__(self, other: float):
        out = Value(self.data**other)
        out._children = {self}
        out._op = "**"
        
        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward
        
        return out
    
    def __truediv__(self, other):
        out = self * (other**-1) 
        return out

class Neuron:
    """
    provide the number of weights for the neuron 
    """
    
    def __init__(self, input_size : int):
        self.bias = Value(random.uniform(-1, 1)) 
        self.weights = [Value(random.uniform(-1, 1)) for x in range(input_size)]
        self.bias.label = "bias"
        for i in range(input_size): 
            self.weights[i].label = f"w{i}"

    def __call__(self, input : list[Value]):
        for i in range(len(input)):
            input[i].label = f"x{i}" 
        tmp = sum((self.weights[i] * input[i] for i in range(len(input))), self.bias)
        out = tmp.tanh() 
        return out 
    
    def __repr__(self):
        return f"Neuron({self.weights})"


class Layer:
    """
    provide the number of neurons and number of inputs of each neuron 
    """
    def __init__(self, input_size: int, neuron_nbr: int):
         self.neurons = [Neuron(input_size=input_size) for _ in range(neuron_nbr)]
    
    def __repr__(self):
        return f"Layer({self.neurons})"

    def __call__(self, input:list[Value]):
        out = []
        for neuron in self.neurons : 
            out.append(neuron(input))
        return out

    def get_params(self): 
        out = []
        for neuron in self.neurons : 
            out.extend(neuron.weights)
        return out 

class MLP:
    """
    Multi-Layer Perceptron
    """
    def __init__(self, input_size: int, layer_sizes: list[int], learning_rate : float):
        self.learning_rate = learning_rate
        self.loss = Value(0.0) 
        self.sizes = [input_size] + layer_sizes
        self.layers = [Layer(self.sizes[i], self.sizes[i+1]) for i in range(len(layer_sizes))]
    
    def Get_params(self):
        params = []
        for layer in self.layers:
            params.extend(layer.get_params())
        return params
    
    def __repr__(self):
        return f"MLP({self.layers})"
    
    
    def Backprop(root : Value): 
        """
        backpropagation function 
        """ 
        def Dfs (node : Value): 
            node._backward()
            for child in node._children: 
                Dfs(child)
        
        root.grad = 1.0        
        Dfs(root)
    
    def Train(self, input: list[Value], labels : list[Value], epochs):
        if(self.sizes[-1] != len(labels)) :
            raise ValueError("Input and labels must have the same length")

        for _ in range(epochs):
            prediction = input
            for layer in self.layers:
                prediction = layer(prediction)
            
            sum = Value(0.0)
            for i in range(len(input)):
                sum += (prediction[i] - labels[i])**2
            self.loss = sum/Value(len(labels))
            
            params = self.Get_params()
            # zero grad
            for param in params: 
                param.grad = 0.0
            
            MLP.Backprop(self.loss)
            for param in params : 
                param.data -= self.learning_rate * param.grad
            
            print(f"Loss for the {_}'th epoch is : {self.loss.data}")
            
    
