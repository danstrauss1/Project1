from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np


fig = plt.figure()
ax = fig.gca(projection='3d')

for i in range(100):
    # x, y, z = np.meshgrid(np.arange(0, 1, 0.2),
    #                       np.arange(0, 1, 0.2),
    #                       np.arange(0, 1, 0.2))

    x, y, z = np.array([0, 0, 0])

    length = 10

    u = np.random.randint(-10, 10) * np.sqrt(length)
    v = np.random.randint(-10, 10) * np.sqrt(length)
    w = np.random.randint(-10, 10) * np.sqrt(length)

    ax.quiver(x, y, z, u, v, w, length=0.01, color='black', alpha=0.2)

plt.show()