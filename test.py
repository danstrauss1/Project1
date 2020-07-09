from skspatial.objects import Line, Plane
from skspatial.plotting import plot_3d
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import glob
import pydicom
import os
import numpy as np
import time

detector = Plane(point=[0, 0, 0], normal=[0, 0, 1])

source_origin = np.array([0, 0, 3])
source_direction = 2 * np.random.random((3, 1)) - 1
photon = Line(point=source_origin.flatten(), direction=source_direction.flatten())


try:
    print(detector.intersect_line(photon))
except:
    print('did not intersect')