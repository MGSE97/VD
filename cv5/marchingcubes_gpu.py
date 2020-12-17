import numpy as np
import matplotlib as mpl
mpl.use("Qt5Cairo")
#mpl.use('TkAgg')
#mpl.use('Agg')
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
#isoLevel = 800
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
    isoLevel = iso[0] + 100

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
    [getValid(r[x], triangles) for x in range(0, 16) for r in r_triangles]

    return np.array(triangles), iso


def construct_surface(points):
    return march_gpu(points)


def compute(file, saved_files):
    name = os.path.basename(file)[:-4]
    iso = (0, 0, 0)
    if name not in saved_files:
        # compute
        points = load_points(file)
        points, iso = construct_surface(points)

        np.save(save_dir + name, points)

    print("{} ({:3.1f}, {:3.1f}, {:3.1f})".format(name, iso[1], iso[0], iso[2]), end=", ")


def load(file, saved_files, ax, fig):
    name = os.path.basename(file)[:-4]
    print(name, end=", ")
    if name in saved_files:
        # load
        points = np.load(save_dir + name + ".npy")

    if len(points) > 0:
        # Precompute drawing data
        tri_points = list(chain.from_iterable(points))
        X, Y, Z = zip(*tri_points)
        tri_idx = [(3 * i, 3 * i + 1, 3 * i + 2) for i in range(len(points))]
        if fig is None:
            fig, ax = prepare_matplotlib()
        surf = ax.plot_trisurf(X, Y, Z, color='cadetblue', triangles=tri_idx, edgecolor='none', linewidth=0, antialiased=False)
        if fig is None:
            surf = (surf._vec, surf._facecolors, surf._facecolors3d, surf._original_facecolor, surf._segslices)
    else:
        X = Y = Z = tri_idx = surf = None

    return name, (X, Y, Z, tri_idx), surf


def animate(frame, computed, ax, fig, plt, surf):
    #X, Y, Z, tri_idx = computed[frame][1]
    print(frame, end=", ")
    fig.suptitle("Marching Cubes {}".format(computed[frame][0]))
    #surf[0].remove()
    #surf[0] = ax.plot_trisurf(X, Y, Z, color='cadetblue', triangles=tri_idx, edgecolor='none', linewidth=0, antialiased=False)
    surf[0]._vec, surf[0]._facecolors, surf[0]._facecolors3d, surf[0]._original_facecolor, surf[0]._segslices = computed[frame][2]
    return []


def prepare_matplotlib():
    #plt.ion()
    name = "Marching Cubes"
    fig = plt.figure(name)
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
        cuda.to_device(np.zeros(shape=(samplesPerAxis ** 3, 16, 4, 3))),
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

def matplotlib(computed, fig, ax):
    X, Y, Z, tri_idx = computed[0][1]
    surf = [ax.plot_trisurf(X, Y, Z, color='cadetblue', triangles=tri_idx, edgecolor='none', linewidth=0, antialiased=False)]
    anim = FuncAnimation(fig, animate, fargs=(computed, ax, fig, plt, surf), interval=100, frames=len(computed) - 1, blit=True, repeat=True, cache_frame_data=len(computed))

    '''
    frames = [[ax.plot_trisurf(X, Y, Z, color='cadetblue', triangles=tri_idx, edgecolor='none', linewidth=0, antialiased=False)] for name, (X, Y, Z, tri_idx) in computed]
    #frames = computed[:][2]
    anim = ArtistAnimation(fig, frames, interval=100, blit=True, repeat=True, repeat_delay=1000)
    '''

    print("\nSaving animation ...")
    anim.save(save_dir+"animation.mp4", fps=60, dpi=360, bitrate=1800)
    print("\nSaved animation ...")

    plt.show()

def animate_seq(frame, files, saved_files, ax, fig, surf):
    if surf[0] is not None:
        surf[0].remove()
    name, (X, Y, Z, tri_idx), surf = load(files[frame], saved_files, ax, fig)
    surf = [surf]
    return []

def matplotlib_seq(files, saved_files, fig, ax):
    surf = [None]
    anim = FuncAnimation(fig, animate_seq, fargs=(files, saved_files, ax, fig, surf), interval=100, frames=len(files) - 1, blit=True, repeat=True, cache_frame_data=1)

    print("\nSaving animation ...")
    anim.save(save_dir+"animation.mp4", fps=60, dpi=360, bitrate=1800)
    print("\nSaved animation ...")

    plt.show()


data_dir = "./sph_17_40x40x40_mu_3_5/"
save_dir = "./sph_17_40x40x40_mu_3_5_iso_81/"
large_data = True

def main():
    print("Preparing data ...")
    files = list(filter(os.path.isfile, glob.glob(data_dir + "*.bin")))
    files.sort(key=lambda x: os.path.split(x)[1])
    saved_files = [os.path.basename(f)[:-4] for f in list(filter(os.path.isfile, glob.glob(save_dir + "*.npy")))]

    fig, ax = prepare_matplotlib()

    print("Computing data ...")
    if large_data:
        for f in files:
            compute(f, saved_files)
        clean_compute()
    else:
        data = [(f, saved_files) for f in files]
        with Pool(11) as p:
            p.starmap(compute, data)
            # Clean up
            clean_compute()

    print("\nLoading data ...")
    saved_files = [os.path.basename(f)[:-4] for f in list(filter(os.path.isfile, glob.glob(save_dir + "*.npy")))]
    if not large_data:
        data = [(f, saved_files, None, None) for f in files]
        with Pool(11) as p:
            computed = sorted([r for r in p.starmap(load, data) if r[2] is not None], key=lambda x: x[0])

    print("\nShowing data ...")
    if __name__ == '__main__':
        if not large_data:
            matplotlib(computed, fig, ax)
        else:
            matplotlib_seq(files, saved_files, fig, ax)


if __name__ == '__main__':
    main()
