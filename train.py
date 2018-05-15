from TrainingSet import TrainingSet
from Vector import Vector
from Hypothesis import Hypothesis
from TrainingExample import TrainingExample

ts = TrainingSet(Hypothesis(Vector(3, 1)), Vector(-1,-1), Vector (1, 1), .1, 10)
#ts = TrainingSet(TrainingExample(Vector(1,1), 1), TrainingExample(Vector(2,1), 2), TrainingExample(Vector(3,1), 3))
h = Hypothesis(2)
for i in range(1000):
	mg = ts.mean_gradient(h)
	print("Mean Gradient {}".format(mg))
	h.update(mg, .1)
	print("Hypothesis {}".format(h.params))
	print("Error {}".format(ts.error(h)))
