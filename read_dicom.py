import pydicom as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import glob
import plotly.graph_objects as go
from scipy.ndimage import zoom
from sympy import Plane, Point3D, Line3D
from sympy.plotting import plot3d
import time
from mpl_toolkits.mplot3d import Axes3D


class Particle:

    num_of_particles = 0

    def __init__(self, x, y, z, Energy, Type='Photon'):
        self.x = x
        self.y = y
        self.z = z
        self.Energy = Energy
        # self.MFP = MFP
        self.Type = Type

        self.vx = np.random.random()
        self.vy = np.random.random()
        self.vz = np.random.random()

        self.num_of_particles += 1

    def increment_movement(self):
        self.x += self.vx
        self.y += self.vy
        self.z += self.vz

class Detector(object):
    prop_defaults = {
        "SID"
    }

    def __init__(self,
                 SID,
                 orientation,
                 xspacing,
                 yspacing,
                 xpixelcount,
                 ypixelcount,
                 xpixelorigin,
                 ypixelorigin):

        self.SID = SID
        self.orientation = orientation
        self.xspacing = xspacing
        self.yspacing = yspacing
        self.xpixelcount = xpixelcount
        self.ypixelcount = ypixelcount
        self.xpixelorigin = xpixelorigin
        self.ypixelorigin = ypixelorigin

        # Create meshgrid
        self.plane = create_plane(SID)
        # xarray = np.arange(self.xpixelorigin, (self.xpixelcount * self.xspacing), self.xpixelcount)
        # yarray = np.arange(self.ypixelorigin, (self.ypixelcount * self.yspacing), self.ypixelcount)
        #
        # xsize = self.xspacing * self.xpixelcount
        # ysize = self.yspacing * self.ypixelcount


def create_plane(zoff):
    points = [[1, 0, -zoff],
              [0, 1, -zoff],
              [1, 1, -zoff]]

    p0, p1, p2 = points
    x0, y0, z0 = p0
    x1, y1, z1 = p1
    x2, y2, z2 = p2

    ux, uy, uz = u = [x1-x0, y1-y0, z1-z0]
    vx, vy, vz = v = [x2-x0, y2-y0, z2-z0]

    u_cross_v = [uy*vz-uz*vy, uz*vx-ux*vz, ux*vy-uy*vx]

    point  = np.array(p0)
    normal = np.array(u_cross_v)

    d = -point.dot(normal)

    xx, yy = np.meshgrid(range(100), range(100))

    z = (-normal[0] * xx - normal[1] * yy - d) * 1. / normal[2]


    return xx, yy, z
    # plot the surface
    # plt3d = plt.figure().gca(projection='3d')
    # plt3d.plot_surface(xx, yy, z)
    # plt.show()

def generate_particles(x, y, z, count, energy, type, distribution):
    particle_list = []
    for i in range(count):
        if distribution == 'Uniform':
            particle = Particle(x, y, z, energy, type)
            particle_list.append(particle)
        elif distribution == 'Normal':
            e = np.random.normal(energy, np.sqrt(energy))
            if e < 0:
                e = 0
            particle = Particle(x, y, z, e, type)
            particle_list.append(particle)
        elif distribution == 'Gamma':
            particle = Particle(x, y, z, np.random.gamma(1, 1), type)

    return particle_list

def plot_particle_distribution(particle_list):

    energies = []
    for particle in particle_list:
        energies.append(particle.Energy)
    plt.hist(energies, bins=25)
    plt.show()

def load_ct(path):
    """Method to load CT
    input: path to dicom directory
    output: list of dicom image slices
    """

    # Uncomment below for files starting in "CT"
    # files = glob.glob(os.path.join(path, 'CT*'))

    # Uncomment below for files ending in ".dcm"
    files = glob.glob(os.path.join(path, '*.dcm'))

    slices = []
    for file in files:
        dcm = pd.dcmread(file)
        try:
            slicePos = dcm.InstanceNumber
            if slicePos is not None:
                # print(f'read file: {dcm.InstanceNumber}')
                slices.append(dcm)
        except:
            print(f"Slice position for file {dcm} not recognized")
    slices.sort(key=lambda x: int(x.InstanceNumber))
    return slices

def plot_3d_ct(vol):
    nb_frames = len(vol)
    vol = zoom(vol, (1, 0.5, 0.5))
    r, c = vol[0].shape
    fig = go.Figure(frames=[go.Frame(data=go.Surface(
        # z=(6.7 - k * 0.1) * np.ones((r, c)),
        z=(k * np.ones((r, c))),
        surfacecolor=np.flipud(vol[(nb_frames - 1) - k]),
        cmin=vol.min(), cmax=vol.max()
    ),
        name=str(k)  # you need to name the frame for the animation to behave properly
    )
        for k in range(nb_frames - 1)])

    # Add data to be displayed before animation starts
    fig.add_trace(go.Surface(
        z=nb_frames * np.ones((r, c)),
        surfacecolor=np.flipud(vol[(nb_frames - 1)]),
        colorscale='Gray',
        cmin=vol.min(), cmax=vol.max(),
        colorbar=dict(thickness=20, ticklen=4)
    ))

    def frame_args(duration):
        return {
            "frame": {"duration": duration},
            "mode": "immediate",
            "fromcurrent": True,
            "transition": {"duration": duration, "easing": "linear"},
        }

    sliders = [
        {
            "pad": {"b": 10, "t": 60},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": [
                {
                    "args": [[f.name], frame_args(0)],
                    "label": str(k),
                    "method": "animate",
                }
                for k, f in enumerate(fig.frames)
            ],
        }
    ]

    # Layout
    fig.update_layout(
        title=f'Slices in volumetric data -- Resolution: {r} x {c} x {nb_frames}',
        width=1000,
        height=1000,
        scene=dict(
            zaxis=dict(range=[-0.1, nb_frames], autorange=False),
            aspectratio=dict(x=1, y=1, z=1),
        ),
        updatemenus=[
            {
                "buttons": [
                    {
                        "args": [None, frame_args(50)],
                        "label": "&#9654;",  # play symbol
                        "method": "animate",
                    },
                    {
                        "args": [[None], frame_args(0)],
                        "label": "&#9724;",  # pause symbol
                        "method": "animate",
                    },
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 70},
                "type": "buttons",
                "x": 0.1,
                "y": 0,
            }
        ],
        sliders=sliders
    )

    fig.show()

def create_plane(vol):
    planes = []
    for x, y in enumerate(vol):
        # planes.append(Plane((0, 0, x), (0, 0, x), (0, 0, x)))
        planes.append(Plane(Point3D(0, 0, x),
                            normal_vector=(0, 0, 1)))
    return planes

def compile_3d_array(slices):
    ct_vol = []
    for slice in slices:
        try:
            ct_vol.append(slice.pixel_array)
        except:
            print(f"Slice # {slice} contains no pixel data")
    return np.asarray(ct_vol)

def does_intersect(plane, line):
    return plane.intersection(line)


t1 = time.time()
ct = load_ct('./Medium')
# ct = load_ct('../../PROJ#1 MC/Medium')
vol = compile_3d_array(ct)
r, c = vol[0].shape
nb_frames = len(vol)
particles = generate_particles(0, 0, 0, 1000, 6, 'Photon', 'Normal')
print(particles[0].x)
for particle in particles:
    particle.increment_movement()
print(particles[0].x)
plot_particle_distribution(particles)
# planes = create_plane(vol)
# print(planes)
plot_3d_ct(vol)
t2 = time.time()

print(f'time to complete: {t2-t1} seconds')
# point1 = Point3D(1, 1, 1)
# point2 = Point3D(2, 3, 4)
# point3 = Point3D(2, 2, 2)
#
# plane = Plane(point1, point2, point3)
# print(plane)
# plot3d(plane)

# print(r, c, nb_frames)
# plt.imshow(ct[-1].pixel_array,
#            cmap='gray')
# plt.show()