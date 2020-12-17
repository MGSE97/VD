import numpy as np
from mpl_toolkits.mplot3d import Axes3D
Axes3D = Axes3D  # pycharm auto import
from matplotlib import cm
import matplotlib.pyplot as plt
import cv2


# 𝒙(𝑡 + ∆𝑡) = 𝒙(𝑡) + 𝒗(𝑡, 𝒙(𝑡))∆𝑡
# 𝒙(𝑡0) = 𝒙0
def Euler(v, x, dt):
    return x + v(x)*dt


# 𝒙(𝑡 + ∆𝑡) = 𝒙(𝑡) + ½ (𝐾1 + 𝐾2)
# 𝐾1 = 𝒗(𝒙(𝑡))∆𝑡
# 𝐾2 = 𝒗(𝒙(𝑡) + 𝐾1)∆t
def RungeKutta2(v, x, dt):
    K1 = v(x) * dt
    K2 = v(x + K1) * dt
    return x + 0.5 * (K1 + K2)


# 𝒙(𝑡 + ∆𝑡) = 𝒙(𝑡) + 1/6 (𝐾1 + 2𝐾2 + 2𝐾3 + 𝐾4)
# 𝐾1 = 𝒗(𝒙(𝑡))∆t
# 𝐾2 = 𝒗(𝒙(𝑡) + ½𝐾1)∆𝑡
# 𝐾3 = 𝒗(𝒙(𝑡) + ½𝐾2)∆𝑡
# 𝐾4 = 𝒗(𝒙(𝑡) + 𝐾3)∆𝑡
def RungeKutta4(v, x, dt):
    K1 = v(x) * dt
    K2 = v(x + 0.5 * K1) * dt
    K3 = v(x + 0.5 * K2) * dt
    K4 = v(x + K3) * dt
    return x + (1/6) * (K1 + 2*K2 + 2*K3 + K4)


def Fix(point, size):
    x, y = point
    w, h = size
    x = max(min(x, w-1), 0)
    y = max(min(y, h-1), 0)
    return x, y


def GetData(point, data, size):
    x, y = point
    w, h = size
    x = max(min(x, w-1), 0)
    y = max(min(y, h-1), 0)
    return data[int(y), int(x), :]


# Settings
size = (256, 256)
gsize = (26, 26)
frames = 100
dt = 2
fce = RungeKutta4

# Prepare
points = []
for i in range(1000):
    x, y = np.random.uniform(0, size[0]), np.random.uniform(0, size[1])
    points.append((x, y))

# Load
big_data = []
for i in range(frames):
    file = 'u{:05d}.yml'.format(i)
    print(file)
    fs = cv2.FileStorage('./flow_field/{}'.format(file), cv2.FILE_STORAGE_READ)
    big_data.append(np.array(fs.getNode('flow').mat()))
    fs.release()

# Compute
name = "CV4"
fig = plt.figure(name)
fig.suptitle(name)
ax = fig.gca()
fig.canvas.draw_idle()

fx = np.linspace(0, size[0], size[0])
fy = np.linspace(0, size[1], size[1])
vX, vY = np.meshgrid(fx, fy)

qx = np.linspace(0, size[0], gsize[0])
qy = np.linspace(0, size[1], gsize[1])
qX, qY = np.meshgrid(qx, qy)

for i in range(frames):
    #ax.cla()
    fig.suptitle('Frame {}'.format(i))
    data = big_data[i]

    # Color from velocity
    #Z = [[np.linalg.norm(np.sqrt(u * u + v * v)) for u, v in dx] for dx in data]
    #Z = [[np.linalg.norm(u+v) for u, v in dx] for dx in data]
    Z = np.linalg.norm(data, axis=2)
    c = ax.contourf(vX, vY, Z, 100, cmap=cm.plasma, vmin=0, vmax=10)

    # Arrows
    u = data[0::10, 0::10, 0]
    v = data[0::10, 0::10, 1]
    U, V = np.meshgrid(u, v)
    q = ax.quiver(qX, qY, u, v, color='white', scale=150, alpha=0.5)

    # Points
    points = np.array([Fix(fce(lambda f: GetData(f, data, size), p, dt), size) for p in points])
    dx = points[:, 0]
    dy = points[:, 1]
    d = ax.scatter(dx, dy, c='green')

    # Cleanup
    plt.pause(0.01)
    [r.remove() for r in c.collections]
    q.remove()
    d.remove()

plt.show()