import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

# 𝜕𝑇/𝜕𝑟(𝑟, 𝑠) = (𝑠 − 1)(𝒑1 − 𝒑2) + 𝑠(𝒑3 − 𝒑4)
# 𝜕𝑇/𝜕𝑠(𝑟, 𝑠) = (𝑟 − 1)(𝒑1 − 𝒑4) + 𝑟(𝒑3 − 𝒑2)
def J(r, s, p):
    return np.array([
        (s - 1) * (p[0] - p[1]) + s * (p[2] - p[3]),
        (r - 1) * (p[0] - p[3]) + r * (p[2] - p[1])
    ]).transpose()


# T(r, s) = (1 - s) * (1 - r) * p1 + rp2
#              + s  * (1 - r) * p4 + rp3
def T(r, s, p):
    a0 = (1 - r) * p[0] + r * p[1]
    a1 = (1 - r) * p[3] + r * p[2]
    return (1 - s) * a0 + s * a1


# (𝑟^(𝑡+1), 𝑠^(𝑡+1)) = (𝑟^𝑡, 𝑠^𝑡) − J_T^−1(𝑟^𝑡, 𝑠^𝑡) * (𝑇(𝑟, 𝑠) − 𝒑)
def NewtonMethod(r, s, t, v, p):
    rs = np.array((r, s))
    for i in range(t):
        J_inv = np.linalg.pinv(J(*rs, v))
        rs = rs - (J_inv @ (T(*rs, v) - p))

    return T(*rs, v)


# Is point under line
def IsUnder(x1, y1, z1, x2, y2, z2, px, py):
    v1 = np.array((x1 - px, y1 - py))
    v2 = np.array((x2 - px, y2 - py))
    return np.cross(v1, v2) < 0


# Checks if point is in grid
def GetGridPoints(grid, xy, d, start):
    px, py = (start + xy) / d
    px, py = int(px), int(py)

    tl = grid[px + 0, py + 0, :]
    bl = grid[px + 0, py + 1, :]
    tr = grid[px + 1, py + 0, :]
    br = grid[px + 1, py + 1, :]

    fixes = np.array([
        (0, -1) if IsUnder(*tl, *tr, *xy) else (0, 0),
        (1, 0) if IsUnder(*tr, *br, *xy) else (0, 0),
        (0, 1) if IsUnder(*br, *bl, *xy) else (0, 0),
        (-1, 0) if IsUnder(*bl, *tl, *xy) else (0, 0)
    ])
    fx, fy = np.sum(fixes, axis=0)

    # Check if is inside
    if fx != 0 or fy != 0:
        # Fix side
        px, py = px + fx, py + fy

        tl = grid[px + 0, py + 0, :]
        bl = grid[px + 0, py + 1, :]
        tr = grid[px + 1, py + 0, :]
        br = grid[px + 1, py + 1, :]

    return (bl, br, tr, tl)


fce = lambda x, y: np.cos(x) * np.sin(y)


# Prepare plot
name = "F(x)"
fig = plt.figure(name)
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
fig.canvas.draw_idle()

# Prepare function
size = (-7, 7)
detail = 50
x = np.linspace(size[0], size[1], detail)
y = np.linspace(size[0], size[1], detail)
X, Y = np.meshgrid(x, y)
Z = fce(X, Y)

# Draw function
ax1.contourf(X, Y, Z, cmap=plt.get_cmap('summer'), levels=[0.025 * x - 1.0 for x in range(80)])

# Sample grid
chunks = 16
rnd = lambda e: np.random.uniform(-0.2, 0.2) if not e else 0.0
edge = lambda i, j: j == size[0] or i == size[0] or j == size[1] or i == size[1]
grid = []
for i in np.linspace(size[0], size[1], chunks):
    row = []
    for j in np.linspace(size[0], size[1], chunks):
        ii = i + rnd(edge(i, j))
        jj = j + rnd(edge(i, j))
        row.append((ii, jj, fce(ii, jj)))
    grid.append(row)

grid = np.array(grid)

# Draw grid
ax1.plot(grid[:, :, 0], grid[:, :, 1], c='black')
ax1.plot(grid[:, :, 0].T, grid[:, :, 1].T, c='black')
ax1.scatter(grid[:, :, 0], grid[:, :, 1], c="black", s=10)

plt.pause(0.1)

# Compute interpolation
fixed = []
rst = (0.5, 0.5, 5)
d = (size[1] - size[0] + 1) / chunks
start = (-size[0], -size[0])
fx = []
fy = []
fz = []
for px in x:
    for py in y:
        xy = np.array((px, py))
        p = GetGridPoints(grid, xy, d, start)
        s = (px, py, fce(px, py))
        nx, ny, nz = NewtonMethod(*rst, p, s)
        fx.append(nx)
        fy.append(ny)
        fz.append(nz)

print(len(fx))

X, Y = np.meshgrid(np.unique(fx), np.unique(fy))
Z = interpolate.griddata((fx, fy), fz, (X, Y), method='linear')

ax2.contourf(X, Y, Z, cmap=plt.get_cmap('summer'), levels=[0.025 * x - 1.0 for x in range(80)])

print('Showing')
plt.pause(1)

plt.show()
