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

# initial learning stuff
coeff = Vector(randint(-5, 5), randint(-5, 5))
ts = TrainingSet(Hypothesis(coeff), Vector.fill(-1, 2), Vector.fill(1, 2), .1, 20)

# data to plot surface
surface_0 = np.arange(-5, 5, 0.25)
surface_1 = np.arange(-5, 5, 0.25)
err = []
surface_0, surface_1 = np.meshgrid(surface_0, surface_1)

for s0, s1 in zip(surface_0, surface_1):
    err.append(ts.error(Hypothesis(Vector(s0, s1))))


#make h start at the highest point so it's dramatic
m_idx = np.argmax(err)
h = Hypothesis(Vector(np.ndarray.flatten(surface_0)[m_idx], np.ndarray.flatten(surface_1)[m_idx]))
print(h.params)

theta_0 = []
theta_1 = []
errors = []

for i in range(1000):

	mg = ts.mean_gradient(h)
	h.update(mg, 1)

	theta_0.append(h.params[0])
	theta_1.append(h.params[1])
	errors.append(ts.error(h))

print("Actual values: {}".format(coeff))
print("Learned values: {}".format(h.params))
print("Percent error: {}".format(100 * sum([abs(c/p) for c, p in zip(coeff - h.params, h.params)])/len(coeff)))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(surface_0, surface_1, err, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

ax.plot(theta_0, theta_1, errors)
plt.show()

