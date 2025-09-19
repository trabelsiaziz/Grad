
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
class Value:
    def __init__(self, value):
        self.data = value
        self.grad = 0.0
        self._children = set()
        self._op = ""
        self.label = ""
        self._backward = lambda : None
            
    def __repr__(self):
        out = f"Value(data={self.data}, grad={self.grad}, lab={self.label})"
        return out 

    def __add__(self, other):
        res = self.data + other.data
        out = Value(res)
        out._children.update([self, other])  
        out._op = "+"    
        
        def compute_grad(): 
            self.grad += res.grad 
            other.grad += res.grad          
        
        out._backward = compute_grad 
        return out
    
    def __mul__(self, other): 
        res = self.data * other.data
        out = Value(res)
        out._op = "*"
        out._children.update([self, other])    
        def compute_grad(): 
            self.grad += res.grad * other.data
            other.grad += res.grad * self.data
        
        out._backward = compute_grad
        return out 