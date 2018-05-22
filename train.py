import matplotlib.pyplot as plt
from random import randint
from matplotlib import cm
import numpy as np
from numpy import arange
from mpl_toolkits.mplot3d import Axes3D
from TrainingSet import TrainingSet
from Vector import Vector
from Hypothesis import Hypothesis
from TrainingExample import TrainingExample

ts = TrainingSet(Hypothesis(Vector(randint(-5, 5), randint(-5, 5))), Vector.fill(-1, 2), Vector.fill(1, 2), .1, 20)

h = Hypothesis(2)

# y = theta_0 * x + theta_1
theta_0 = []
theta_1 = []

errors = []

for i in range(1000):

	mg = ts.mean_gradient(h)
	h.update(mg, 1)

	theta_0.append(h.params[0])
	theta_1.append(h.params[1])
	errors.append(ts.error(h))

surface_0 = []
surface_1 = []
surface_err = []

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Make data.
surface_0 = np.arange(-5, 5, 0.25)
surface_1 = np.arange(-5, 5, 0.25)
err = []
surface_0, surface_1 = np.meshgrid(surface_0, surface_1)

for s0, s1 in zip(surface_0, surface_1):
	err.append(ts.error(Hypothesis(Vector(s0, s1))))

surf = ax.plot_surface(surface_0, surface_1, err, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

ax.plot(theta_0, theta_1, errors)
plt.show()
