import numpy as np
import matplotlib as mpl
from matplotlib.animation import FuncAnimation
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from functools import partial
from itertools import chain
from multiprocessing import Pool
import os
import glob
from numba import cuda
import math

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
def struct_unpack(x):
    return np.frombuffer(x, dtype=np.float16)


def load_points(filename):
    with open(filename, "rb") as f:
        points = np.array([struct_unpack(chunk) for chunk in iter(partial(f.read, 18), b'')])

    return np.array(points)


def draw_points(fig, ax, triangles):
    ax.cla()
    ax.set_zlim(0, 1)
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    tri_points = list(chain.from_iterable(triangles))
    X, Y, Z = zip(*tri_points)
    tri_idx = [(3 * i, 3 * i + 1, 3 * i + 2) for i in range(len(triangles))]
    ax.plot_trisurf(X, Y, Z, triangles=tri_idx, antialiased=False)
    #X, Y, Z = points[:, 0], points[:, 1], points[:, 2]
    #ax.scatter(X, Y, Z, edgecolors='black', alpha=0.1)

def draw_points2(fig, ax, X, Y, Z, tri_idx):
    ax.cla()
    ax.set_zlim(0, 1)
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.plot_trisurf(X, Y, Z, triangles=tri_idx, antialiased=False)


samplesPerAxis = 40
#isoLevel = 800
w = 6
wr = 8
w2 = 3
offset = (-0.5, -0.5, 0)
step = 1.0/samplesPerAxis


def interpolateVerts(v1, v2, isoLevel, offset):
    v1, v2 = np.array((offset[0] + v1[0], offset[1] + v1[1], offset[2] + v1[2], v1[3])), np.array((offset[0] + v2[0], offset[1] + v2[1], offset[2] + v2[2], v2[3]))
    t = (isoLevel - v1[3]) / (v2[3] - v1[3])
    return v1[0:3] + t * (v2[0:3]-v1[0:3])


def indexFromCoord(x, y, z, points):
    '''rx, ry, rz = offset[0]+x, offset[1]+y, offset[2]+z
    rp = np.array((rx, ry, rz))
    deltas = points - rp
    dist_2 = np.einsum('ij,ij->i', deltas, deltas)
    i = np.argmin(dist_2)
    return i, points'''
    return z * samplesPerAxis * samplesPerAxis + y * samplesPerAxis + x, points

def setPoint(i, x, y, z, grid, cube):
    cube[i] = [
        x*step,
        y*step,
        z*step,
        grid[z, y, x]
    ]


grid_distance = 1.0/float(samplesPerAxis)

def computePoint(x, y, z, points, computed, cubeCorners):
    rp = np.array((offset[0] + x, offset[1] + y, offset[2] + z))
    if computed[x, y, z, w2] != 0:
        cubeCorners.append(computed[x, y, z, :])
        return

    deltas = points[:, 0:3] - rp
    dist_2 = np.einsum('ij,ij->i', deltas, deltas)

    densities = []
    for i, d in enumerate(dist_2):
        if d <= grid_distance:
            densities.append(points[i][w])

    if len(densities) == 0:
        computed[x, y, z, :] = x, y, z, -1
    else:
        computed[x, y, z, :] = x, y, z, np.average(densities)

    cubeCorners.append(computed[x, y, z, :])


volume = step**3
def computeGridPoint(*args):
    points, x, y, z = args
    rp = np.array(
        (offset[0] + x*step,
         offset[1] + y*step,
         offset[2] + z*step,
         offset[0] + x*step + step,
         offset[1] + y*step + step,
         offset[2] + z*step + step))
    sum = np.sum([p[w] * (p[wr]**3) for p in points if rp[0] <= p[0] < rp[3] and rp[1] <= p[1] < rp[4] and rp[2] <= p[2] < rp[5]])
    return (x,y,z, sum / volume)

def setGrid(grid, x, y, z, s):
    grid[z, y, x] = s
    return s

def computeGrid_cpu(points, grid):
    data = [(points, x, y, z) for x in range(0, samplesPerAxis) for y in range(0, samplesPerAxis) for z in range(0, samplesPerAxis)]
    with Pool(11) as p:
        result = p.starmap(computeGridPoint, data)

    isoLevel = np.sum([setGrid(grid, *r) for r in result])
    isoLevel /= result.__len__()

    return isoLevel

@cuda.jit
def computeGridPoint_gpu(points, grid):
    x, y, z = cuda.grid(3)
    if x < grid.shape[0] and y < grid.shape[1] and z < grid.shape[2]:
        rp =(offset[0] + x*step,
             offset[1] + y*step,
             offset[2] + z*step,
             offset[0] + x*step + step,
             offset[1] + y*step + step,
             offset[2] + z*step + step)

        sum = 0
        for p in points:
            if rp[0] <= p[0] < rp[3] and rp[1] <= p[1] < rp[4] and rp[2] <= p[2] < rp[5]:
                mult = 1.0
            else:
                mult = 0.0
            sum += mult * p[w] * (p[wr]**3)

        grid[z, y, x] = sum / volume

def computeGrid_gpu(points, grid):
    d_points = cuda.to_device(points.astype(dtype=np.float))
    d_grid = cuda.to_device(grid)

    threadsPerBlock = (16, 16, 4)
    blocksPerGrid = (
        math.ceil(grid.shape[0] / threadsPerBlock[0]),
        math.ceil(grid.shape[1] / threadsPerBlock[1]),
        math.ceil(grid.shape[2] / threadsPerBlock[2])
    )

    computeGridPoint_gpu[blocksPerGrid, threadsPerBlock](d_points, d_grid)

    points = d_points.copy_to_host()
    grid = d_grid.copy_to_host()

    isoLevel = np.sum(grid)
    isoLevel /= samplesPerAxis**3

    return isoLevel, grid, points

@cuda.jit
def getTriangles_gpu(grid, isoLevel, io_triangles, triangulation, cornerIndexAFromEdge, cornerIndexBFromEdge, offset):
    x, y, z = cuda.grid(3)
    if x < grid.shape[0]-1 and y < grid.shape[1]-1 and z < grid.shape[2]-1:
        # 8 corners of the current cube
        cubeCorners = (
            ((x+0), (y+0), (z+0)),
            ((x+1), (y+0), (z+0)),
            ((x+1), (y+0), (z+1)),
            ((x+0), (y+0), (z+1)),
            ((x+0), (y+1), (z+0)),
            ((x+1), (y+1), (z+0)),
            ((x+1), (y+1), (z+1)),
            ((x+0), (y+1), (z+1))
        )

        # Calculate unique index for each cube configuration.
        # There are 256 possible values
        # A value of 0 means cube is entirely inside surface; 255 entirely outside.
        # The value is used to look up the edge table, which indicates which edges of the cube are cut by the isosurface.
        cubeIndex = 0
        for i in range(0, 8):
            iso = grid[cubeCorners[i][2], cubeCorners[i][1], cubeCorners[i][0]]
            if iso < isoLevel:
                cubeIndex |= 1 << i
            else:
                cubeIndex = cubeIndex

        # Create triangles for current cube configuration
        i = 0
        r = z * samplesPerAxis * samplesPerAxis + y * samplesPerAxis + x
        rs = 0
        #for n in range(0, 16):
            #io_triangles[r, n, 0, :] = (0, 0, 0, 0)
        while triangulation[cubeIndex][i] != -1:
            # get indices
            a = cornerIndexAFromEdge[triangulation[cubeIndex][i]]
            b = cornerIndexBFromEdge[triangulation[cubeIndex][i]]

            # interpolate vectors
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
            t = (isoLevel - grid[cubeCorners[a][2], cubeCorners[a][1], cubeCorners[a][0]]) / (grid[cubeCorners[b][2], cubeCorners[b][1], cubeCorners[b][0]] - grid[cubeCorners[a][2], cubeCorners[a][1], cubeCorners[a][0]])
            vt = (
                v1[0] + t * (v2[0] - v1[0]),
                v1[1] + t * (v2[1] - v1[1]),
                v1[2] + t * (v2[2] - v1[2])
            )
            io_triangles[r, rs, 1, :] = vt

            # get indices
            a = cornerIndexAFromEdge[triangulation[cubeIndex][i + 1]]
            b = cornerIndexBFromEdge[triangulation[cubeIndex][i + 1]]

            # interpolate vectors
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
            t = (isoLevel - grid[cubeCorners[a][2], cubeCorners[a][1], cubeCorners[a][0]]) / (grid[cubeCorners[b][2], cubeCorners[b][1], cubeCorners[b][0]] - grid[cubeCorners[a][2], cubeCorners[a][1], cubeCorners[a][0]])
            vt = (
                v1[0] + t * (v2[0] - v1[0]),
                v1[1] + t * (v2[1] - v1[1]),
                v1[2] + t * (v2[2] - v1[2])
            )
            io_triangles[r, rs, 2, :] = vt

            # get indices
            a = cornerIndexAFromEdge[triangulation[cubeIndex][i + 2]]
            b = cornerIndexBFromEdge[triangulation[cubeIndex][i + 2]]

            # interpolate vectors
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
            t = (isoLevel - grid[cubeCorners[a][2], cubeCorners[a][1], cubeCorners[a][0]]) / (grid[cubeCorners[b][2], cubeCorners[b][1], cubeCorners[b][0]] - grid[cubeCorners[a][2], cubeCorners[a][1], cubeCorners[a][0]])
            vt = (
                v1[0] + t * (v2[0] - v1[0]),
                v1[1] + t * (v2[1] - v1[1]),
                v1[2] + t * (v2[2] - v1[2])
            )
            io_triangles[r, rs, 3, :] = vt

            # set as active
            io_triangles[r, rs, 0, :] = (1, 0, 0)
            i = i + 3
            rs += 1

def getTriangles(grid, isoLevel, x, y, z):
    # Select 8 corners
    cubeCorners = np.zeros(shape=(8, 4))
    setPoint(0, x, y, z, grid, cubeCorners)
    setPoint(1, x + 1, y, z, grid, cubeCorners)
    setPoint(2, x + 1, y, z + 1, grid, cubeCorners)
    setPoint(3, x, y, z + 1, grid, cubeCorners)
    setPoint(4, x, y + 1, z, grid, cubeCorners)
    setPoint(5, x + 1, y + 1, z, grid, cubeCorners)
    setPoint(6, x + 1, y + 1, z + 1, grid, cubeCorners)
    setPoint(7, x, y + 1, z + 1, grid, cubeCorners)

    # Compute type, 256 types
    cubeIndex = 0
    if cubeCorners[0][w2] < isoLevel:
        cubeIndex |= 1
    if cubeCorners[1][w2] < isoLevel:
        cubeIndex |= 2
    if cubeCorners[2][w2] < isoLevel:
        cubeIndex |= 4
    if cubeCorners[3][w2] < isoLevel:
        cubeIndex |= 8
    if cubeCorners[4][w2] < isoLevel:
        cubeIndex |= 16
    if cubeCorners[5][w2] < isoLevel:
        cubeIndex |= 32
    if cubeCorners[6][w2] < isoLevel:
        cubeIndex |= 64
    if cubeCorners[7][w2] < isoLevel:
        cubeIndex |= 128

    triangles = []
    # Create triangles
    i = 0
    while triangulation[cubeIndex][i] != -1:
        # Indices
        a0 = cornerIndexAFromEdge[triangulation[cubeIndex][i]]
        b0 = cornerIndexBFromEdge[triangulation[cubeIndex][i]]

        a1 = cornerIndexAFromEdge[triangulation[cubeIndex][i + 1]]
        b1 = cornerIndexBFromEdge[triangulation[cubeIndex][i + 1]]

        a2 = cornerIndexAFromEdge[triangulation[cubeIndex][i + 2]]
        b2 = cornerIndexBFromEdge[triangulation[cubeIndex][i + 2]]

        # Interpolate vectors
        tri = [
            interpolateVerts(cubeCorners[a0], cubeCorners[b0], isoLevel, offset),
            interpolateVerts(cubeCorners[a1], cubeCorners[b1], isoLevel, offset),
            interpolateVerts(cubeCorners[a2], cubeCorners[b2], isoLevel, offset)
        ]

        triangles.append(tri)
        i = i + 3

    return triangles

def appendArr(arr, data):
    arr.extend(data)

def march_cpu(points):
    grid = np.zeros(shape=(samplesPerAxis, samplesPerAxis, samplesPerAxis))
    #isoLevel = computeGrid_cpu(points, grid)
    isoLevel, grid, points = computeGrid_gpu(points, grid)
    print(volume, step, isoLevel)

    xyz = [(grid, isoLevel, x, y, z) for x in range(0, samplesPerAxis-1) for y in range(0, samplesPerAxis - 1) for z in range(0, samplesPerAxis-1)]
    with Pool(11) as p:
        results = p.starmap(getTriangles, xyz)

    triangles = []
    [appendArr(triangles, r) for r in results]

    return np.array(triangles)

def getValid(x, triangles):
    v, a, b, c = x
    if v[0] == 1:
        appendArr(triangles, [(a, b, c)])

def march_gpu(points):
    grid = np.zeros(shape=(samplesPerAxis, samplesPerAxis, samplesPerAxis))
    isoLevel, grid, points = computeGrid_gpu(points, grid)
    print(isoLevel)

    threadsPerBlock = (16, 16, 4)
    blocksPerGrid = (
        math.ceil((samplesPerAxis - 1) / threadsPerBlock[0]),
        math.ceil((samplesPerAxis - 1) / threadsPerBlock[1]),
        math.ceil((samplesPerAxis - 1) / threadsPerBlock[2])
    )

    d_grid = cuda.to_device(grid)
    d_triangles = cuda.to_device(np.zeros(shape=(samplesPerAxis**3, 16, 4, 3)))
    d_triangulation = cuda.to_device(triangulation)
    d_cornerA = cuda.to_device(cornerIndexAFromEdge)
    d_cornerB = cuda.to_device(cornerIndexBFromEdge)
    d_offset = cuda.to_device(offset)
    getTriangles_gpu[blocksPerGrid, threadsPerBlock](d_grid, isoLevel, d_triangles, d_triangulation, d_cornerA, d_cornerB, d_offset)

    r_triangles = d_triangles.copy_to_host()

    triangles = []
    [getValid(r[x], triangles) for x in range(0, 16) for r in r_triangles]

    print(triangles.__len__())
    return np.array(triangles)


def construct_surface(points):
    return march_gpu(points)
    #return march_cpu(points)

def compute(file, saved_files):
    name = os.path.basename(file)[:-4]
    print(name)
    points = []
    if name in saved_files:
        # load
        points = np.load(save_dir + name + ".npy")
    else:
        # compute
        points = load_points(file)
        points = construct_surface(points)

        np.save(save_dir + name, points)

    tri_points = list(chain.from_iterable(points))
    X, Y, Z = zip(*tri_points)
    tri_idx = [(3 * i, 3 * i + 1, 3 * i + 2) for i in range(len(points))]

    return name, (X, Y, Z, tri_idx)

def animate(frame, computed, ax, fig, plt, surf):
    X, Y, Z, tri_idx = computed[frame][1]
    fig.suptitle("Marching Cubes {}".format(frame))
    surf[0].remove()
    surf[0] = ax.plot_trisurf(X, Y, Z, color='cadetblue', triangles=tri_idx, edgecolor='none', linewidth=0, antialiased=False)
    return []

data_dir = "./sph_17_40x40x40_mu_3_5/"
save_dir = "./sph_17_40x40x40_mu_3_5_iso/"
def main():
    print("Preparing data ...")

    files = list(filter(os.path.isfile, glob.glob(data_dir + "*.bin")))[0:11]
    files.sort(key=lambda x: os.path.split(x)[1])
    saved_files = [os.path.basename(f)[:-4] for f in list(filter(os.path.isfile, glob.glob(save_dir + "*.npy")))]

    data = [(f, saved_files) for f in files]
    with Pool(11) as p:
        computed = sorted(p.starmap(compute, data), key=lambda x: x[0])

    print("Showing data ...")

    name = "Marching Cubes"
    fig = plt.figure(name)
    fig.suptitle(name)
    ax = fig.gca(projection='3d')
    ax.set_zlim(0, 1)
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    #fig.canvas.draw_idle()
    #plt.pause(0.1)
    #plt.show()

    X, Y, Z, tri_idx = computed[0][1]
    surf = [ax.plot_trisurf(X, Y, Z, color='cadetblue', triangles=tri_idx, edgecolor='none', linewidth=0, antialiased=False)]
    anim = FuncAnimation(fig, animate, fargs=(computed, ax, fig, plt, surf), interval=100, frames=len(computed) - 1, blit=True, repeat=True, cache_frame_data=len(computed))

    plt.show()
    '''
    plt.pause(0.1)

    while True:
        anim.frame_seq = anim.new_frame_seq()
        anim.event_source.start()
        plt.pause(0.1*computed.__len__())
    '''
    '''
    while True:
        for name, points in computed:
            print(name, len(points[0]))
            #draw_points(fig, ax, points)
            draw_points2(fig, ax, *points)
            #fig.canvas.draw()
            #plt.show(block=False)
            plt.pause(0.000001)
    '''

    plt.show()

if __name__ == '__main__':
    main()