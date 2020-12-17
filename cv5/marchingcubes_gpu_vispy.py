import glob
import os
import sys
import time
from itertools import chain

import numpy as np
from multiprocessing import Process, Queue, Pool, freeze_support

from functools import partial
from numba import cuda
import math

import vispy
from vispy import app, scene, io
from scipy import interpolate

from cv5.tables import triangulation, cornerIndexAFromEdge, cornerIndexBFromEdge, triangulationB

load_dir = "./sph_17_40x40x40_mu_3_5/"
load_limit = 100
save_dir = "./sph_17_40x40x40_mu_3_5_iso_81_v_3/"
debug = False
save = False
samplesPerAxis = 81 #41

isoLevelOffset = 0
isoLevelFixed = 0

smoothingDistance = 0.1
smoothingFactor = 2 #2
smoothingCount = 0  #0 disabled

offset = (-0.5, -0.5, 0)
step = 1.0 / samplesPerAxis
volume = step ** 3

triangulation_table = triangulationB
pden = 6
pvol = 8

'''
struct Particle {
    0 half position_x; // particle position (m)
    1 half position_y;
    2 half position_z;

    3 half velocity_x; // particle velocity (m/s)
    4 half velocity_y;
    5 half velocity_z;

    6 half rho; // density (kg/m3)
    7 half pressure;
    8 half radius; // particle radius (m)
};
'''


def timing(f):
    def pvolap(*args, **kwargs):
        if debug:
            time1 = time.time()
            ret = f(*args, **kwargs)
            time2 = time.time()
            print('\t\t{:20s} {:#6.3f} ms'.format(f.__name__, (time2 - time1) * 1000.0))
        else:
            ret = f(*args, **kwargs)
        return ret

    return pvolap


def struct_unpack(x):
    return np.frombuffer(x, dtype=np.float16)


@timing
def load_points(filename):
    with open(filename, "rb") as f:
        points = np.array([struct_unpack(chunk) for chunk in iter(partial(f.read, 18), b'')])

    return np.array(points)


@cuda.jit
def computeGridPoint_gpu(grid, points):
    x, y, z = cuda.grid(3)
    if x < grid.shape[0] and y < grid.shape[1] and z < grid.shape[2]:
        rp = (offset[0] + x * step - step,
              offset[1] + y * step - step,
              offset[2] + z * step - step,
              offset[0] + x * step + step,
              offset[1] + y * step + step,
              offset[2] + z * step + step)

        sum = 0
        for p in points:
            if rp[0] <= p[0] < rp[3] and rp[1] <= p[1] < rp[4] and rp[2] <= p[2] < rp[5]:
                mult = 1.0
            else:
                mult = 0.0
            sum += mult * p[pden] * (p[pvol] ** 3)  # density * volume

        grid[z, y, x] = sum / volume


o = 1/(math.pi*smoothingDistance**3)
c = 4/3*math.pi # V = 4/3 πr³
@cuda.jit
def computeGridPointSPH_gpu(grid, points):
    x, y, z = cuda.grid(3)
    if x < grid.shape[0] and y < grid.shape[1] and z < grid.shape[2]:
        rp = (offset[0] + x * step,
              offset[1] + y * step,
              offset[2] + z * step)
        W = 0
        for p in points:
            r = math.sqrt((rp[0] - p[0])**2 + (rp[1] - p[1])**2 + (rp[2] - p[2])**2)
            q = r / smoothingDistance
            if 0 <= q <= 1:
                w = o * (1-(3/2)*q**2*(1-q/2))
            elif 1 < q <= 2:
                w = (o/4)*(2-q)**3
            else:
                w = 0

            W += w * p[pden] * (c*p[pvol] ** 3)  # density * volume

        grid[z, y, x] = W

@cuda.jit
def computeGridPoint2_gpu(grid, points):
    i = cuda.grid(1)
    if i < points.shape[0]:
        point = points[i]
        x = int(math.floor((point[0] - offset[0]) / step))
        y = int(math.floor((point[1] - offset[1]) / step))
        z = int(math.floor((point[2] - offset[2]) / step))
        if z < grid.shape[0] and y < grid.shape[1] and x < grid.shape[2]:
            grid[z, y, x] += point[pden] * (point[pvol] ** 3)  # density * volume

@cuda.jit
def zeroGrid_gpu(grid):
    x, y, z = cuda.grid(3)
    if x < grid.shape[0] and y < grid.shape[1] and z < grid.shape[2]:
        grid[z, y, x] = 0

@cuda.jit
def divideVolumeGrid_gpu(grid):
    x, y, z = cuda.grid(3)
    if x < grid.shape[0] and y < grid.shape[1] and z < grid.shape[2]:
        grid[z, y, x] /= volume

@cuda.jit
def smoothGridPoint_gpu(grid):
    x, y, z = cuda.grid(3)
    if smoothingFactor < x < grid.shape[0] - smoothingFactor \
            and smoothingFactor < y < grid.shape[1] - smoothingFactor \
            and smoothingFactor < z < grid.shape[2] - smoothingFactor:

        result = 0
        r = 2 * smoothingFactor + 1
        for sz in range(r):
            for sy in range(r):
                for sx in range(r):
                    result += grid[
                        z + sz - smoothingFactor,
                        y + sy - smoothingFactor,
                        x + sx - smoothingFactor
                    ]

        grid[z, x, y] = result / (r ** 3)


@timing
def smoothGrid_gpu(d_grid):
    threadsPerBlock = (16, 16, 4)
    blocksPerGrid = (
        math.ceil(samplesPerAxis / threadsPerBlock[0]),
        math.ceil(samplesPerAxis / threadsPerBlock[1]),
        math.ceil(samplesPerAxis / threadsPerBlock[2])
    )

    for x in range(smoothingCount):
        smoothGridPoint_gpu[blocksPerGrid, threadsPerBlock](d_grid)

    isoLevel, isoMin, isoMax = np.average(d_grid), np.min(d_grid), np.max(d_grid)
    #isoLevel /= samplesPerAxis ** 3

    return (isoLevel, isoMin, isoMax), d_grid

@timing
def computeGrid_gpu(points, d_grid):
    d_points = cuda.to_device(points.astype(dtype=np.float))

    threadsPerBlock3d = (16, 16, 4)
    blocksPerGrid3d = (
        math.ceil(samplesPerAxis / threadsPerBlock3d[0]),
        math.ceil(samplesPerAxis / threadsPerBlock3d[1]),
        math.ceil(samplesPerAxis / threadsPerBlock3d[2])
    )

    threadsPerBlock1d = 1024
    blocksPerGrid1d = math.ceil(len(points) / threadsPerBlock1d)

    # Cubic Spline smoothing kernel
    computeGridPointSPH_gpu[blocksPerGrid3d, threadsPerBlock3d](d_grid, d_points)

    # Volume raw grid
    #computeGridPoint_gpu[blocksPerGrid3d, threadsPerBlock3d](d_grid, d_points)

    # Faster volume raw grid
    #zeroGrid_gpu[blocksPerGrid3d, threadsPerBlock3d](d_grid)
    #computeGridPoint2_gpu[blocksPerGrid1d, threadsPerBlock1d](d_grid, d_points)
    #divideVolumeGrid_gpu[blocksPerGrid3d, threadsPerBlock3d](d_grid)

    isoLevel, isoMin, isoMax = np.average(d_grid), np.min(d_grid), np.max(d_grid)
    #isoLevel /= samplesPerAxis ** 3

    return (isoLevel, isoMin, isoMax), d_grid


@cuda.jit
def getTriangles_gpu(grid, isoLevel, io_triangles, triangulation, cornerIndexAFromEdge, cornerIndexBFromEdge, offset):
    x, y, z = cuda.grid(3)
    if x < grid.shape[0] - 1 and y < grid.shape[1] - 1 and z < grid.shape[2] - 1:
        # Select 8 corners
        cubeCorners = (
            ((x + 0), (y + 0), (z + 0)),
            ((x + 1), (y + 0), (z + 0)),
            ((x + 1), (y + 0), (z + 1)),
            ((x + 0), (y + 0), (z + 1)),
            ((x + 0), (y + 1), (z + 0)),
            ((x + 1), (y + 1), (z + 0)),
            ((x + 1), (y + 1), (z + 1)),
            ((x + 0), (y + 1), (z + 1))
        )

        # Compute type, 256 types
        cubeIndex = 0
        for i in range(0, 8):
            iso = grid[cubeCorners[i][2], cubeCorners[i][1], cubeCorners[i][0]]
            if iso < isoLevel:
                cubeIndex |= 1 << i
            else:
                cubeIndex = cubeIndex

        # Create triangles
        i = 0
        r = z * samplesPerAxis * samplesPerAxis + y * samplesPerAxis + x
        rs = 0
        for n in range(0, 16):
            io_triangles[r, n, 0, 0] = 0
        while triangulation[cubeIndex][i] != -1:
            # Indices
            a = cornerIndexAFromEdge[triangulation[cubeIndex][i]]
            b = cornerIndexBFromEdge[triangulation[cubeIndex][i]]

            # Interpolate vector
            v1, v2 = (
                         offset[0] + cubeCorners[a][0] * step,
                         offset[1] + cubeCorners[a][1] * step,
                         offset[2] + cubeCorners[a][2] * step
                     ), (
                         offset[0] + cubeCorners[b][0] * step,
                         offset[1] + cubeCorners[b][1] * step,
                         offset[2] + cubeCorners[b][2] * step
                     )
            # (iso - v1) * (v2 - v1)
            t = (isoLevel - grid[cubeCorners[a][2], cubeCorners[a][1], cubeCorners[a][0]]) / (
                    grid[cubeCorners[b][2], cubeCorners[b][1], cubeCorners[b][0]] - grid[
                cubeCorners[a][2], cubeCorners[a][1], cubeCorners[a][0]])
            vt = (
                v1[0] + t * (v2[0] - v1[0]),
                v1[1] + t * (v2[1] - v1[1]),
                v1[2] + t * (v2[2] - v1[2])
            )
            io_triangles[r, rs, 1, :] = vt

            # Indices
            a = cornerIndexAFromEdge[triangulation[cubeIndex][i + 1]]
            b = cornerIndexBFromEdge[triangulation[cubeIndex][i + 1]]

            # Interpolate vector
            v1, v2 = (
                         offset[0] + cubeCorners[a][0] * step,
                         offset[1] + cubeCorners[a][1] * step,
                         offset[2] + cubeCorners[a][2] * step
                     ), (
                         offset[0] + cubeCorners[b][0] * step,
                         offset[1] + cubeCorners[b][1] * step,
                         offset[2] + cubeCorners[b][2] * step
                     )
            # (iso - v1) * (v2 - v1)
            t = (isoLevel - grid[cubeCorners[a][2], cubeCorners[a][1], cubeCorners[a][0]]) / (
                    grid[cubeCorners[b][2], cubeCorners[b][1], cubeCorners[b][0]] - grid[
                cubeCorners[a][2], cubeCorners[a][1], cubeCorners[a][0]])
            vt = (
                v1[0] + t * (v2[0] - v1[0]),
                v1[1] + t * (v2[1] - v1[1]),
                v1[2] + t * (v2[2] - v1[2])
            )
            io_triangles[r, rs, 2, :] = vt

            # Indices
            a = cornerIndexAFromEdge[triangulation[cubeIndex][i + 2]]
            b = cornerIndexBFromEdge[triangulation[cubeIndex][i + 2]]

            # Interpolate vector
            v1, v2 = (
                         offset[0] + cubeCorners[a][0] * step,
                         offset[1] + cubeCorners[a][1] * step,
                         offset[2] + cubeCorners[a][2] * step
                     ), (
                         offset[0] + cubeCorners[b][0] * step,
                         offset[1] + cubeCorners[b][1] * step,
                         offset[2] + cubeCorners[b][2] * step
                     )
            # (iso - v1) * (v2 - v1)
            t = (isoLevel - grid[cubeCorners[a][2], cubeCorners[a][1], cubeCorners[a][0]]) / (
                    grid[cubeCorners[b][2], cubeCorners[b][1], cubeCorners[b][0]] - grid[
                cubeCorners[a][2], cubeCorners[a][1], cubeCorners[a][0]])
            vt = (
                v1[0] + t * (v2[0] - v1[0]),
                v1[1] + t * (v2[1] - v1[1]),
                v1[2] + t * (v2[2] - v1[2])
            )
            io_triangles[r, rs, 3, :] = vt

            # Set as active
            io_triangles[r, rs, 0, 0] = 1
            i = i + 3
            rs += 1


@timing
def triangulate_gpu(d_grid, isoLevel, d_triangles, d_triangulation, d_cornerA, d_cornerB, d_offset):
    threadsPerBlock = (16, 16, 4)
    blocksPerGrid = (
        math.ceil((samplesPerAxis - 1) / threadsPerBlock[0]),
        math.ceil((samplesPerAxis - 1) / threadsPerBlock[1]),
        math.ceil((samplesPerAxis - 1) / threadsPerBlock[2])
    )

    getTriangles_gpu[blocksPerGrid, threadsPerBlock](d_grid, isoLevel, d_triangles, d_triangulation, d_cornerA,
                                                     d_cornerB, d_offset)

    triangles = d_triangles.copy_to_host()
    triangles = triangles.reshape(triangles.shape[0] * triangulation_table.shape[1], 4, 3)
    triangles = triangles[triangles[:, 0, 0] == 1]
    triangles = triangles[:, 1:4, :]
    return triangles


@timing
def march_gpu(points, prepared):
    d_grid, d_triangles, d_triangulation, d_cornerA, d_cornerB, d_offset = prepared

    iso, d_grid = computeGrid_gpu(points, d_grid)

    if smoothingFactor > 0:
        iso, d_grid = smoothGrid_gpu(d_grid)

    if isoLevelFixed > 0:
        isoLevel = isoLevelFixed
    else:
        isoLevel = iso[0] + isoLevelOffset
    print("\tiso ({:3.1f}, {:3.1f}, {:3.1f})".format(isoLevel, iso[0], iso[2]), end="\n")

    triangles = triangulate_gpu(d_grid, isoLevel, d_triangles, d_triangulation, d_cornerA, d_cornerB, d_offset)

    return np.array(triangles), iso


def computeSurface(file, prepared):
    return march_gpu(load_points(file), prepared)[0]

prepared = None
@timing
def prepare_compute():
    global prepared

    if prepared is not None:
        return prepared

    prepared = (
        cuda.to_device(np.zeros(shape=(samplesPerAxis, samplesPerAxis, samplesPerAxis))),
        cuda.to_device(np.zeros(shape=(samplesPerAxis ** 3, triangulation_table.shape[1], 4, 3))),
        cuda.to_device(triangulation_table),
        cuda.to_device(cornerIndexAFromEdge),
        cuda.to_device(cornerIndexBFromEdge),
        cuda.to_device(offset)
    )

    return prepared

@timing
def clean_compute():
    global prepared

    if prepared is not None:
        return prepared

    del prepared


def createVisual(file):
    print(os.path.basename(file)[:-4], end="\n")
    npf = "{}.npy".format(os.path.join(save_dir, os.path.basename(file)[:-4]))

    if os.path.exists(npf):
        # Load surface
        data = np.load(npf)
    else:
        # Compute surface
        prepared = prepare_compute()
        data = computeSurface(file, prepared)
        np.save(npf, data)

    print("\tdata", data.shape, end="\n")

    if len(data) > 0:
        # Create mesh visual
        surface = scene.visuals.Mesh(data, color=(0.5, 0.6, 1, 1), shading='smooth')
        surface.transform = scene.transforms.STTransform(translate=(0.5, 0.5, 0))
    else:
        surface = None
    return file, surface

i = 0
view = None
saved = False
surfaces = []
images = []
size = 0
def main():
    global surfaces
    global view
    global size
    global i

    # Find data
    if load_limit > 0:
        loaded_files = list(filter(os.path.isfile, glob.glob(load_dir + "*.bin")))[0:load_limit]
    else:
        loaded_files = list(filter(os.path.isfile, glob.glob(load_dir + "*.bin")))
    size = len(loaded_files)

    # Prepare save dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Compute surfaces
    #with Pool(6) as p:
        #surfaces = sorted(p.map(createVisual, loaded_files), key=lambda x: x[0])[:, 1]
    surfaces = [createVisual(f)[1] for f in loaded_files]
    clean_compute()

    vispy.gloo.gl.glEnable(vispy.gloo.gl.GL_DEPTH_TEST)

    # Create a canvas with a 3D viewport
    canvas = scene.SceneCanvas(keys='interactive')
    view = canvas.central_widget.add_view()

    i = 0
    def redraw(event):
        global i
        global surfaces
        global view
        global images
        global save_dir
        global saved

        size = surfaces.__len__()
        if size > 1:
            if i >= size:
                i = 0
                if not saved:
                    saved = True

            if i == 0 and surfaces[size - 1] is not None:
                surfaces[size - 1].parent = None
            elif surfaces[i - 1] is not None:
                surfaces[i - 1].parent = None

        if surfaces[i] is not None:
            surfaces[i].parent = view.scene

        if size > 1:
            i += 1

        canvas.update()
        if save and not saved:
            img = canvas.render()
            name = "sph{}".format(i)
            io.write_png("{}{}.png".format(save_dir, name), img)


    timer = app.timer.Timer()
    timer.connect(redraw)
    timer.start(0.1)

    # Add a 3D axis
    axis = scene.visuals.XYZAxis(parent=view.scene)

    # Use a 3D camera
    cam = scene.TurntableCamera(elevation=30, azimuth=30)
    cam.set_range((-0.2, 1.2), (-0.2, 1.2), (-0.2, 1.2))
    view.camera = cam

    canvas.show()
    if sys.flags.interactive == 0:
        app.run()

if __name__ == '__main__':
    freeze_support()
    main()
