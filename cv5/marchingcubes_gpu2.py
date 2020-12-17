import numpy as np
import matplotlib as mpl
#mpl.use("Qt5Cairo")
#mpl.use('TkAgg')
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, ArtistAnimation
from matplotlib import animation
from functools import partial
from itertools import chain
from multiprocessing import Pool
import os
import glob
from numba import cuda
import math
import sys
import asyncio

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from cv5.tables import triangulation, cornerIndexAFromEdge, cornerIndexBFromEdge

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

samplesPerAxis = 81
isoLevelOffset = 0
w = 6
wr = 8
offset = (-0.5, -0.5, 0)
step = 1.0 / samplesPerAxis
volume = step ** 3


def struct_unpack(x):
    return np.frombuffer(x, dtype=np.float16)


def load_points(filename):
    with open(filename, "rb") as f:
        points = np.array([struct_unpack(chunk) for chunk in iter(partial(f.read, 18), b'')])

    return np.array(points)


@cuda.jit
def computeGridPoint_gpu(points, grid):
    x, y, z = cuda.grid(3)
    if x < grid.shape[0] and y < grid.shape[1] and z < grid.shape[2]:
        rp = (offset[0] + x * step,
              offset[1] + y * step,
              offset[2] + z * step,
              offset[0] + x * step + step,
              offset[1] + y * step + step,
              offset[2] + z * step + step)

        sum = 0
        for p in points:
            if rp[0] <= p[0] < rp[3] and rp[1] <= p[1] < rp[4] and rp[2] <= p[2] < rp[5]:
                mult = 1.0
            else:
                mult = 0.0
            sum += mult * p[w] * (p[wr] ** 3)  # density * volume

        grid[z, y, x] = sum / volume


def computeGrid_gpu(points, d_grid):
    d_points = cuda.to_device(points.astype(dtype=np.float))
    #d_grid = cuda.to_device(grid)

    threadsPerBlock = (16, 16, 4)
    blocksPerGrid = (
        math.ceil(samplesPerAxis / threadsPerBlock[0]),
        math.ceil(samplesPerAxis / threadsPerBlock[1]),
        math.ceil(samplesPerAxis / threadsPerBlock[2])
    )

    computeGridPoint_gpu[blocksPerGrid, threadsPerBlock](d_points, d_grid)

    isoLevel, isoMin, isoMax = np.sum(d_grid), np.min(d_grid), np.max(d_grid)
    isoLevel /= samplesPerAxis ** 3

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


def appendArr(arr, data):
    arr.extend(data)


def getValid(x, triangles):
    v, a, b, c = x
    if v[0] == 1:
        appendArr(triangles, [(a, b, c)])


def march_gpu(points):
    d_grid, d_triangles, d_triangulation, d_cornerA, d_cornerB, d_offset = prepare_compute()

    iso, d_grid = computeGrid_gpu(points, d_grid)
    #isoLevel = iso[0] + isoLevelOffset
    isoLevel = 1600

    threadsPerBlock = (16, 16, 4)
    blocksPerGrid = (
        math.ceil((samplesPerAxis - 1) / threadsPerBlock[0]),
        math.ceil((samplesPerAxis - 1) / threadsPerBlock[1]),
        math.ceil((samplesPerAxis - 1) / threadsPerBlock[2])
    )

    #d_grid = cuda.to_device(grid)
    getTriangles_gpu[blocksPerGrid, threadsPerBlock](d_grid, isoLevel, d_triangles, d_triangulation, d_cornerA,
                                                     d_cornerB, d_offset)

    r_triangles = d_triangles.copy_to_host()

    triangles = []
    [getValid(r[x], triangles) for x in range(0, 15) for r in r_triangles]

    return np.array(triangles), iso


def construct_surface(points):
    return march_gpu(points)


def prepare_matplotlib(fname = ""):
    #plt.ion()
    name = "Marching Cubes {}".format(fname)
    fig = plt.figure(name, figsize=(10.80, 10.80), dpi=100)
    fig.suptitle(name)
    ax = fig.gca(projection='3d')
    ax.set_zlim(0, 1)
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    #fig.canvas.draw_idle()
    return fig, ax

prepared = None
def prepare_compute():
    global prepared

    if prepared is not None:
        return prepared

    prepared = (
        cuda.to_device(np.zeros(shape=(samplesPerAxis, samplesPerAxis, samplesPerAxis))),
        cuda.to_device(np.zeros(shape=(samplesPerAxis ** 3, 15, 4, 3))),
        cuda.to_device(triangulation),
        cuda.to_device(cornerIndexAFromEdge),
        cuda.to_device(cornerIndexBFromEdge),
        cuda.to_device(offset)
    )

    return prepared

def clean_compute():
    global prepared

    if prepared is not None:
        del prepared

def compute(file, saved_files):
    name = os.path.basename(file)[:-4]
    iso = (0, 0, 0)
    if name not in saved_files:
        # compute
        points = load_points(file)
        points, iso = construct_surface(points)

        np.save(save_dir + name, points)

    print("{} ({:3.1f}, {:3.1f}, {:3.1f})".format(name, iso[1], iso[0], iso[2]), end=", ")


def convert(file, saved_files, image_files):
    name = os.path.basename(file)[:-4]
    print(name, end=", ")
    if name in saved_files and name not in image_files:
        # load
        points = np.load(save_dir + name + ".npy")

        if len(points) > 0:
            # Precompute drawing data
            tri_points = list(chain.from_iterable(points))
            X, Y, Z = zip(*tri_points)
            tri_idx = [(3 * i, 3 * i + 1, 3 * i + 2) for i in range(len(points))]
            fig, ax = prepare_matplotlib(name)
            surf = ax.plot_trisurf(X, Y, Z, color='cadetblue', triangles=tri_idx, edgecolor='none', linewidth=0, antialiased=False)
            fig.savefig(save_dir+"{}.png".format(name), bbox_inches='tight', dpi=100)
            plt.close(fig)


def create_video(image_files):
    imageNames = "images.txt"
    fileName = "sph.mp4"
    fps = 10
    duration = 1/fps

    with open(save_dir+imageNames, "w") as f:
        f.write("\n".join(["file '{}.png'\nduration {}".format(f,duration) for f in image_files]))

    command = "cd && ffmpeg -r {} -c:v libx264 -crf 0 -c:a aac -strict -2 -y {} -f concat -i {}".format(fps, fileName, imageNames)
    print(command)
    os.chdir(save_dir)
    os.system(command)
    os.system('"{}"'.format(fileName))
    os.chdir("..")


data_dir = "./sph_17_40x40x40_mu_3_5/"
save_dir = "./sph_17_40x40x40_mu_3_5_iso_81_5/"
threads = 12

def main():
    print("Preparing data ...")
    files = list(filter(os.path.isfile, glob.glob(data_dir + "*.bin")))
    files.sort(key=lambda x: os.path.split(x)[1])
    saved_files = [os.path.basename(f)[:-4] for f in list(filter(os.path.isfile, glob.glob(save_dir + "*.npy")))]

    print("Computing data ...")
    data = [(f, saved_files) for f in files]
    with Pool(int(threads/2)) as p:
        p.starmap(compute, data)
        # Clean up
        clean_compute()

    print("\nConverting data ...")
    saved_files = [os.path.basename(f)[:-4] for f in list(filter(os.path.isfile, glob.glob(save_dir + "*.npy")))]
    image_files = [os.path.basename(f)[:-4] for f in list(filter(os.path.isfile, glob.glob(save_dir + "*.png")))]
    data = [(f, saved_files, image_files) for f in files]
    with Pool(threads) as p:
        p.starmap(convert, data)

    print("\nSaving video ...")
    image_files = [os.path.basename(f)[:-4] for f in list(filter(os.path.isfile, glob.glob(save_dir + "*.png")))]
    create_video(image_files)


if __name__ == '__main__':
    main()
