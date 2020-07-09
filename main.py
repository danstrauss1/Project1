from skspatial.objects import Line, Plane
from skspatial.plotting import plot_3d
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import glob
import pydicom
import os
import numpy as np
import time

def compile_3d_array(slices):
    ct_vol = []
    for slice in slices:
        try:
            ct_vol.append(slice.pixel_array)
        except:
            print(f"Slice # {slice} contains no pixel data")
    return np.asarray(ct_vol)

def create_plane(vol):
    planes = []
    load_every = 1
    for x, y in enumerate(vol):
        if x % load_every == 0:
            planes.append(Plane(point=[0, 0, x],
                                normal=(0, 0, 1)))
    return planes

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
        dcm = pydicom.dcmread(file)
        try:
            slicePos = dcm.InstanceNumber
            if slicePos is not None:
                # print(f'read file: {dcm.InstanceNumber}')
                slices.append(dcm)
        except:
            print(f"Slice position for file {dcm} not recognized")
    slices.sort(key=lambda x: int(x.InstanceNumber))
    return slices

def strikes_detector(detector, line, xmin, ymin, xspacing, yspacing, xcount, ycount):
    xmax = xspacing * xcount
    ymax = yspacing * ycount

    try:
        point_intersection = detector.intersect_line(line)
        if xmin <= point_intersection[0] <= xmax and ymin <= point_intersection[1] <= ymax:
            return line#, point_intersection
    except:
        return False

def create_photon_array(num_photons):
    photon_array = []
    num_photons = 100
    for i in range(num_photons):
        source_origin = np.array([0, 0, 800])
        # source_direction = np.random.randn(3, 1)
        source_direction = np.array([10, 10, 1])

        photon = Line(point=source_origin.flatten(), direction=source_direction.flatten())
        photon_array.append(photon)
    return photon_array

def find_CT_strike(image_planes, line):
    loc = []
    for slice in image_planes:
        loc.append(slice.intersect_line(line))
    return loc



t1 = time.time()
ct = load_ct('./Medium')
# ct = load_ct('../../PROJ#1 MC/Medium')
vol = compile_3d_array(ct)
# r, c = vol[0].shape
# nb_frames = len(vol)
image_planes = create_plane(vol)

detector = Plane(point=[0, 0, -1], normal=[0, 0, 1])

photon = Line([0, 0, 800], [0, 0, 1])
print(photon)

# hit_list = []
# for photon in photons:
#     hit = strikes_detector(detector, photon, -1000, 1000, -1000, 1000)
#     hit_list.append(hit)
# print(hit_list)
#
# for hit in hit_list:
#     if hit is not None:
#         strikes = find_CT_strike(image_planes, hit)
#
# print('strikes', strikes)
# detector.intersect_line(photon)
# planes = create_plane(vol)
# print(planes)

# pixels_vals = []
# for i, ct_slice in enumerate(ct):
#     xidx = int(strikes[i][0])
#     yidx = int(strikes[i][1])
#
#     pixels_vals.append(ct[i].pixel_array[xidx, yidx])
# print('pixel vals', pixels_vals)
t2 = time.time()

print(f'time to complete: {t2-t1} seconds')
print()