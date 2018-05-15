from Vector import Vector

class Hypothesis:
	params = Vector()

	def __init__(self, x):
		if isinstance(x, Vector):
			self.params = x
		else:
			self.params = Vector([0] * x)
	
	def __call__(self, x):
		if len(x) + 1 == len(self.params):
			# If x is too short, it probably means the constant is not included, so add constant to end
			return self.params * Vector((x + [1]))
		else:
			return self.params * x
	
	def residual(self, training_example):
		return training_example.y - self(training_example.x)

	def r_squared(self, training_example):
		return self.residual(training_example) ** 2

	def gradient(self, training_example):
                return self.residual(training_example) * training_example.x
	
	def update(self, mean_gradient, step_size):
		self.params -= (mean_gradient * step_size)


