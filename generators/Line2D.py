import random
import numpy as np


class Settings:
    x = (0, 100)
    y = (-30, 30)
    double = True,
    step = 1
    change = 0.3
    save = "line2D.txt"


def get_number():
    if Settings.double:
        return random.uniform(Settings.y[0], Settings.y[1])
    else:
       return random.randrange(Settings.y[0], Settings.y[1])


data = []
last = get_number()
for x in np.arange(Settings.x[0], Settings.x[1], step=Settings.step):
    last = last + get_number() * Settings.change
    data.append([x, last])

file = open(Settings.save, 'w')
file.writelines(["{}\t{:.2f}\n".format(*d) for d in data])
file.close()
