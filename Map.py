
from Utils import *
import numpy as np
from matplotlib import pyplot as plt

class Map:
    def __init__(self):
        self.size = 256*3
        self.path_width = 75
        self.margin = 10

        self.bg = ImageUtils.make_image(self.size, self.size)
        image_load = self.bg.load()
        for i in range(self.bg.size[0]):
            for j in range(self.bg.size[1]):
                image_load[i, j] = (0, 100, 0)

        for i in range(100):
            for j in range(100):
                image_load[i,j] = (200, 0, 0)

        # self.bg.show()
        self.path_generator()
        print(1)


    def path_generator(self):

        n= 20

        x = np.random.randint(0, 50, n)
        y = np.random.randint(0, 50, n)

        ##computing the (or a) 'center point' of the polygon
        center_point = [np.sum(x) / n, np.sum(y) / n]

        angles = np.arctan2(x - center_point[0], y - center_point[1])

        ##sorting the points:
        sort_tups = sorted([(i, j, k) for i, j, k in zip(x, y, angles)], key=lambda t: t[2])

        ##making sure that there are no duplicates:
        if len(sort_tups) != len(set(sort_tups)):
            raise Exception('two equal coordinates -- exiting')

        x, y, angles = zip(*sort_tups)
        x = list(x)
        y = list(y)

        ##appending first coordinate values to lists:
        x.append(x[0])
        y.append(y[0])

        fig, ax = plt.subplots()

        ax.plot(x, y, label='{}'.format(n))
        fig.show()

        print(1)
        pass





if __name__ == "__main__":
    m = Map()