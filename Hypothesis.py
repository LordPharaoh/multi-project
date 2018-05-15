class Hypothesis:
	params = Vector()

	def __init__(x):
		if isinstance(x, Vector):
			self.params = x
		else:
			self.params = Vector([0] * x)
	
	def __call__(x):
		if len(x) + 1 == len(self.params):
			# If x is too short, it probably means the constant is not included, so add constant to end
			return (self.params * x) + self.params[-1]
		else:
			return self.params * x
	
	def residual(training_example):
		return training_example.y - self(training_example.x)

	def r_squared(training_example):
		return self.residual(training_example) ** 2
