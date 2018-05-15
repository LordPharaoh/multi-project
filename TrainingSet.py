from Vector import Vector

class TrainingSet(list):

    def __init__(self, *args):
        if len(args) == 1:
            # If a list is passed in it will be nested (args = [[otherlist]]
            super(TrainingSet, self).__init__(args[0])
        else:
            super(TrainingSet, self).__init__(args)

    def error(self, hypothesis):
        tesum = 0
        for te in self:
            tesum += hypothesis.r_squared(te)
        return tesum/len(self)

    def mean_gradient(self, hypothesis):
        gradsum = Vector([0] * len(hypothesis.params))
        for te in self:
            gradsum = gradsum + hypothesis.gradient(te)
        return gradsum/len(self)
    
        
        
 
