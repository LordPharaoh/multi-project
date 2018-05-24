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
ts = TrainingSet(Hypothesis(coeff, Hypothesis.LOGISTIC), Vector.fill(-1, 2), Vector.fill(1, 2), 0, 100)

# data to plot surface
surface_0 = np.arange(-5, 5, 0.25)
surface_1 = np.arange(-5, 5, 0.25)
err = []
surface_0, surface_1 = np.meshgrid(surface_0, surface_1)

gradient_u = []
gradient_v = []
gradient_w = []

for s0, s1 in zip(surface_0, surface_1):
    err.append(ts.error(Hypothesis(Vector(s0, s1), Hypothesis.LOGISTIC)))
    grad = ts.mean_gradient(Hypothesis(Vector(s0, s1), Hypothesis.LOGISTIC)).unit() * .5
    gradient_u.append(grad.x)
    gradient_v.append(grad.y)
    gradient_w.append(0)


#make h start at the highest point so it's dramatic
m_idx = np.argmax(err)
m_idx = m_idx // len(surface_0), m_idx % len(surface_0)
h = Hypothesis(Vector(surface_0[m_idx], surface_1[m_idx]), Hypothesis.LOGISTIC)

theta_0 = []
theta_1 = []
errors = []

for i in range(1000):

    theta_0.append(h.params[0])
    theta_1.append(h.params[1])
    errors.append(ts.error(h))

    mg = ts.mean_gradient(h)
    h.update(mg, 1)

print("Actual values: {}".format(coeff))
print("Learned values: {}".format(h.params))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
q = ax.quiver(surface_0, surface_1, err, gradient_u, gradient_v, gradient_w)
surf = ax.plot_surface(surface_0, surface_1, np.array(err), cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

ax.plot(theta_0, theta_1, errors)
plt.show()

