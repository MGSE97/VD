import numpy as np
from mpl_toolkits.mplot3d import Axes3D
Axes3D = Axes3D  # pycharm auto import
from matplotlib import cm
import matplotlib.pyplot as plt
from scipy import interpolate


def GetWeight(p, s, method, r):
    xy = np.array(s[0:2])
    d = np.linalg.norm(p - xy)
    o = method(d)
    r[0] += o * s[2]  # sum(Oi(p)*fi)
    r[1] += o         # sum(Oi(p))


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

plt.pause(0.1)

# Sample
sample_count = 100
samples = np.random.uniform(size[0], size[1], size=(sample_count, 2))
samples = np.array([(x, y, fce(x, y)) for x, y in samples])

# Draw samples
ax1.scatter(samples[:, 0], samples[:, 1], c="black", s=5)
plt.pause(0.1)

# Compute
# Radial Basis Functions:
# e ^ -(εr)^2
gaussian = lambda e, r: np.exp(-(e*r)**2)
# 1 / (1 + (εr)^2)
inverse_quadratic = lambda e, r: 1/(1 + (e*r)**2)

# Selected function
method = lambda d: gaussian(1, d)
#method = lambda d: inverse_quadratic(10, d)

def Sheppard(px , py):
    p = np.array((px, py))
    r = [0, 0]
    [GetWeight(p, s, method, r) for s in samples]
    return (px, py, r[0] / r[1])  # sheppard

f = np.array([Sheppard(px, py) for px in x for py in y])
fx = f[:, 0]
fy = f[:, 1]
fz = f[:, 2]

print(len(fx))

X, Y = np.meshgrid(np.unique(fx), np.unique(fy))
Z = interpolate.griddata((fx, fy), fz, (X, Y), method='linear')

ax2.contourf(X, Y, Z, cmap=plt.get_cmap('summer'), levels=[0.025 * x - 1.0 for x in range(80)])

print('Showing')
plt.pause(1)

plt.show()