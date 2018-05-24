from Vector import Vector

class TrainingExample:
    x = Vector()
    y = 0
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __str__(self):
        return "({} | {})".format(str(self.x), str(self.y))
    def __repr__(self):
        return str(self)
