import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


point = np.array([0, 0, 0])
normal = np.array([0, 0, 1])

source = np.array([4, 4, -0.1])

d = -point.dot(normal)

xx, yy = np.meshgrid(range(10), range(10))

z = (-normal[0] * xx - normal[1] * yy - d) * 1. / normal[2]

vector = np.array([0, 0, 1])

# source = np.array([0, 0, 0])

length = 10

u = np.random.random(100)
v = np.random.random(100)
w = np.random.random(100)
direction = np.array([u, v, w])

plt3d = plt.figure().gca(projection='3d')
plt3d.plot_surface(xx, yy, z)
plt3d.quiver(source[0], source[1], source[2], u, v, w, length=0.01, color='black', alpha=0.2)
plt.show()