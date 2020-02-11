
from Utils import *


class Map:
    def __init__(self):
        self.size = 256*3
        self.bg = ImageUtils.make_image(self.size, self.size)
        image_load = self.bg.load()
        for i in range(self.bg.size[0]):
            for j in range(self.bg.size[1]):
                image_load[i, j] = (0, 100, 0)

        for i in range(100):
            for j in range(100):
                image_load[i,j] = (200, 0, 0)

        self.bg.show()
        print(1)





if __name__ == "__main__":
    m = Map()