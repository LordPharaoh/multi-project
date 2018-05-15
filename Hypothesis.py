from Vector import Vector

class Hypothesis:
	params = Vector()

	def __init__(self, x):
		if isinstance(x, Vector):
			self.params = x
		else:
			self.params = Vector.zero(x)
	
	def __call__(self, x):
			return self.params * x
	
	def residual(self, training_example):
		return training_example.y - self(training_example.x)

	def r_squared(self, training_example):
		return self.residual(training_example) ** 2

	def gradient(self, training_example):
                return training_example.x * self.residual(training_example)
	
	def update(self, mean_gradient, step_size):
		self.params = self.params + (mean_gradient * step_size)


