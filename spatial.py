from skspatial.objects import Line, Plane
from skspatial.plotting import plot_3d
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# plane = Plane(point=[0, 0, 0], normal=[0, 0, 1])
# line = Line(point=[-1, -1, 5], direction=[0, 0, -1])
for _ in range(1):
    plane = Plane(point=[0, 0, 0], normal=[0, 0, 1])

    source_origin = np.array([0, 0, 10])
    # source_direction = 2 * np.random.random((3, 1)) - 1
    source_direction = np.random.randn(3, 1)
    # source_direction = np.array([0, 0, -1])
    line = Line(point=source_origin.flatten(), direction=source_direction.flatten())

    print(line)

    point_intersection = plane.intersect_line(line)

    try:
        print(plane.intersect_line(line))
    except:
        print('did not intersect')

    plot_3d(
        plane.plotter(lims_x=[-10, 10], lims_y=[-10, 10], alpha=0.2),
        line.plotter(),
        point_intersection.plotter(c='k', s=25),
    )
    plt.show()
print()