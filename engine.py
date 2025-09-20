
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
        out = f"Value(data={self.data}, grad={self.grad}, children={self._children})"
        return out 

    def __add__(self, other):
        res = self.data + other.data
        out = Value(res)
        out._children.update([self, other])  
        out._op = "+"    
        
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
    def __init__(self, neuron_nbr: int, input_size: int):
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
    
