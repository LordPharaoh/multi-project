from TrainingSet import TrainingSet
from Vector import Vector
from Hypothesis import Hypothesis
from TrainingExample import TrainingExample

ts = TrainingSet(TrainingExample(Vector([1]), 1), TrainingExample(Vector([2]), 2), TrainingExample(Vector([3]), 3))
h = Hypothesis(2)
while ts.error(h) != 0:
	h.update(ts.mean_gradient(h), .1)
	print(ts.error(h))
