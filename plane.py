from sympy import Plane, Point3D, Ray3D
from sympy.plotting import plot3d
import random

source_loc = Point3D(10, 10, 10)
randx, randy, randz = random.randint(-10, 10), random.randint(-10, 10), random.randint(-10, 10)
rand_dir = Point3D(randx, randz, randz)

detector = Plane(Point3D(0, 0, 0), normal_vector=(0, 0, 1))
photon = Ray3D(source_loc, rand_dir)

print(f'photon interacts with detector at {detector.intersection(photon)}')
print()