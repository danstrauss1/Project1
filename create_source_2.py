from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np


fig = plt.figure()
ax = fig.gca(projection='3d')

# Create source location
source = np.array([0, 0, 0])

length = 10000

u = np.random.uniform(-1, 1, 100) * length
v = np.random.uniform(-1, 1, 100) * length
w = np.random.uniform(-1, 1, 100) * length
direction = np.array([u, v, w])

ax.quiver(source[0], source[1], source[2], u, v, w, length=0.01, color='black', alpha=0.2)

# directions = np.random.rand(100, 3) * np.random.uniform([-1, 1])
plt.show()
print()