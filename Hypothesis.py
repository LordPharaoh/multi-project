import numpy as np
from Vector import Vector

class Hypothesis:
    params = Vector()

    LINEAR = 0
    LOGISTIC = 1

    def __init__(self, x, basis=LINEAR):
        self.basis = basis
        if isinstance(x, Vector):
            self.params = x
        else:
            self.params = Vector.zero(x)
    
    def __call__(self, x):
            if self.basis == self.LINEAR:
                return self.params * x
            if self.basis == self.LOGISTIC:
                return (1 + np.exp(self.params * x * -1)) ** -1
    
    def gradient(self, training_example):
            return training_example.x * (self(training_example.x) - training_example.y) 
    
    def update(self, mean_gradient, step_size):
        self.params = self.params - (mean_gradient * step_size)
    
    def error(self, training_example):
        if self.basis == self.LINEAR:
            return (self(training_example.x) - training_example.y) ** 2
        if self.basis == self.LOGISTIC:
            return np.log(self(training_example.x)) if training_example.y < .5  else np.log(1 - self(training_example.x))


