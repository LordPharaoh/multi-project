import matplotlib.pyplot as plt
from Hypothesis import Hypothesis
from Vector import Vector
from random import uniform
from TrainingExample import TrainingExample

class TrainingSet(list):

    def __init__(self, hypothesis_or_list, bottom_range=0, top_range=0, degree_rand=0, num_examples=0):
        # If a list is passed in it will be nested (args = [[otherlist]]
        if isinstance(hypothesis_or_list, list):
            super(TrainingSet, self).__init__(hypothesis_or_list)
        else:
            #randomly generate tset
            super(TrainingSet, self).__init__()
            for num in range(num_examples):
                te_vector = TrainingExample(Vector.zero(len(hypothesis_or_list.params)), 0)
                for br, tr, i in zip(bottom_range, top_range, range(len(bottom_range) - 1)):
                    te_vector.x[i] = uniform(br, tr)
                te_vector.x[-1] = 1
                te_vector.y = hypothesis_or_list(te_vector.x) + uniform(-degree_rand,degree_rand)
                self.append(te_vector)

    def error(self, hypothesis):
        tesum = 0
        for te in self:
            tesum += hypothesis.error(te)
        return tesum/len(self)

    def mean_gradient(self, hypothesis):
        gradsum = Vector([0] * len(hypothesis.params))
        for te in self:
            gradsum = gradsum + hypothesis.gradient(te)
        return gradsum/len(self)
 
