class Vector(list):
    """ 
    I didn't mean to create a whole new vector class, I was using the one from vectors as well as a bunch of random
    helper functions from the beginning, and then I noticed the vectors.vector class was really bad an made an 
    improvement or two, and the next thing you know it's its own class
    """
    def __init__(self, *args):
        if len(args) == 1:
            # If a list is passed in it will be nested (args = [[otherlist]]
            super(Vector, self).__init__(args[0])
            self.order = len(args[0]) - 1
        else:
            self.order = len(args) - 1
            super(Vector, self).__init__(args)
        self.x = self[0]
        self.y = self[1]
        self.z = 0 if len(self) < 3 else self[2]
        self.a, self.b, self.c = self.x, self.y, self.z

    def slope(self, point):
        return (self.y - point.y) / (self.x - point.x)

    def midpoint(self, point):
		return (self + point) * .5

    def distance(self, p2):
        total = 0
        for v1, v2 in zip(self, p2):
            total += (v1 - v2) ** 2
        return total ** .5

    def collinear(*args):
        """ True if any number of given 2-dimensional points are collinear """
        slope = args[0].slope(args[1])
        for i in args[2:]:
            if args[0].slope(i) != slope:
                return False
        return True

    def complex(self):
        return self.x + self.y * 1j

    def __add__(self, other):
        if isinstance(other, Vector):
            return Vector(*[s + o for s, o in zip(self, other)])
        else:
            return Vector(*[i + other for i in self])

    def __mul__(self, other):
        """ Returns dot product if multiplied by a vector or a scalar product vector if multipled by a scalar"""
        if isinstance(other, Vector):
            total = 0
            for s, o in zip(self, other):
                total += s * o
            return total
        else:
            return Vector(*[i * other for i in self])

    def __sub__(self, other):
        return self + (other * -1)

    def __truediv__(self, other):
        return self * (other ** -1)

    def __matmul__(self, other):
        return Vector(np.cross(self, other))

    def cross(self, other):
        # This should be the new python3 "across" operator (@) but just in case we need python2
        # Too lazy to do cross products myself
        return Vector(np.cross(self, other))

    def __abs__(self):
        return self.distance(Vector(*[0 for i in self]))

    def __str__(self):
        ret = "<"
        for i in self:
            ret += str(i) + ", "
        return ret[:-2] + ">"

    def __repr__(self):
        return self.__str__()

# easier to think about
Point = Vector
