import random
import numpy as np


class Settings:
    x = (0, 200)
    y = (0, 200)
    t = (-50, 50)
    count = 50
    save = "cluster2D.txt"


def get_number(setting, double):
    if double:
        return random.uniform(setting[0], setting[1])
    else:
        return random.randrange(setting[0], setting[1])


clusters = []
for i in range(0, Settings.count):
    clusters.append([
        get_number(Settings.x, False),
        get_number(Settings.y, False),
        get_number(Settings.t, True),
    ])

data = []
for x in np.arange(Settings.x[0], Settings.x[1]):
    for y in np.arange(Settings.y[0], Settings.y[1]):
        val = 0.0
        for c_x, c_y, c_s in clusters:
            size = abs(c_s)
            neg = c_s < 0
            dst = np.sqrt((x - c_x)**2 + (y - c_y)**2)
            if dst < size:
                val += -(size - dst) if neg else (size - dst)
        data.append([x, y, val])

file = open(Settings.save, 'w')
file.writelines(["{}\t{}\t{:.2f}\n".format(*d) for d in data])
file.close()
