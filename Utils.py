from PIL import Image, ImageDraw
import math
from Config import Config
import numpy as np
from copy import copy
from copy import deepcopy as dcopy

class RectUtils:
    # useful for wall blocks, but not for the car, cause the dot product of the car's vertices might
    # result in something equals a very small number but not zero
    @staticmethod
    def sort_vertice(vertice):
        def vector(vertice1, vertice2):
            return [vertice2[0] - vertice1[0], vertice2[1] - vertice1[1]]

        def valid_vector(vec):
            return True if vec[0] and vec[1] else False

        def around_null(res):
            return True if abs(res) < 1 else False

        v1, v2, v3, v4 = vertice
        vec1 = vector(v1, v2)
        vec2 = vector(v3, v4)
        if valid_vector(vec1) and valid_vector(vec2) and 0 == vec1[0] * vec2[1] - vec1[1] * vec2[0]:
            vec1 = vector(v1, v2)
            vec2 = vector(v2, v3)
            if valid_vector(vec1) and valid_vector(vec2) and 0 == vec1[0] * vec2[0] + vec1[1] * vec2[1]:
                return [v1, v2, v3, v4]
            else:
                return [v1, v2, v4, v3]
        else:
            vec1 = vector(v1, v3)
            vec2 = vector(v3, v4)
            if valid_vector(vec1) and valid_vector(vec2) and 0 == vec1[0] * vec2[0] + vec1[1] * vec2[1]:
                return [v1, v3, v4, v2]
            else:
                return [v1, v3, v2, v4]

    @staticmethod
    def get_car_vertice_no_rotate(pos, car_size):
        x, y = pos
        car_len, car_width = Config.car_length_base() * car_size, Config.car_width_base() * car_size
        radius = ((car_len ** 2 + car_width ** 2) ** 0.5) / 2
        car_angle = math.atan(car_width / car_len)

        def calc_car_vertice(prefix=1, adjust=0.0):
            return ((x + radius * math.cos(prefix * car_angle + adjust)),
                    (y + radius * math.sin(prefix * car_angle + adjust)))

        front_left = calc_car_vertice()
        front_right = calc_car_vertice(prefix=-1)
        back_left = calc_car_vertice(prefix=-1, adjust=math.pi)
        back_right = calc_car_vertice(prefix=1, adjust=math.pi)
        return front_left, front_right, back_left, back_right

    # the order might be wrong, but, it draws the correct rectangle, just be careful with the radar position
    # I am too lazy to correct the name on this function
    @staticmethod
    def get_car_vertice(pos, angle, car_size=1):
        front_left, front_right, back_left, back_right = RectUtils.get_car_vertice_no_rotate(pos, car_size)
        x, y = pos

        def rotate_vertice(vertice, angle):
            angle = math.radians(angle)
            return ((vertice[0] - x) * math.cos(angle) + (vertice[1] - y) * math.sin(angle) + x,
                    -(vertice[0] - x) * math.sin(angle) + (vertice[1] - y) * math.cos(angle) + y)

        front_left = rotate_vertice(front_left, angle)
        front_right = rotate_vertice(front_right, angle)
        back_left = rotate_vertice(back_left, angle)
        back_right = rotate_vertice(back_right, angle)
        return front_left, front_right, back_right, back_left

    @staticmethod
    def radar_pos(vertice):
        fr_l, fr_r, bc_r, bc_l = vertice
        fr_l, fr_r, bc_r, bc_l = np.array(fr_l), np.array(fr_r), np.array(bc_r), np.array(bc_l)
        r1 = dcopy((fr_l + bc_l)/2)
        r2 = dcopy((r1+fr_l)/2)
        r3 = dcopy(fr_l)
        r4 = dcopy((fr_l+fr_r)/2)
        r5 = dcopy(fr_r)
        r7 = dcopy((fr_r+bc_r)/2)
        r6 = dcopy((fr_r+r7)/2)
        return [r1, r2, r3, r4, r5, r6, r7]


class ImageUtils:

    @staticmethod
    def draw_map(size=2):
        # image = ImageUtils.make_image(size, size)
        # return ImageUtils.draw_rect(image, (169,169,169))
        return Image.new('RGB', (size * Config.map_base(), size * Config.map_base()), Config.gray_rbg())

    @staticmethod
    def make_image(i, j):
        return Image.new("RGB", (i, j), "white")

    @staticmethod
    def draw_rect(image, vertice, color=(200, 200, 200), outline=None):
        # im_ = image.load()
        ImageDraw.Draw(image).polygon(vertice, fill=color, outline=outline)

    @staticmethod
    def draw_radar(image, radars, angle):
        r = 200


    @staticmethod
    def draw_car(image, pos, angle, car_size=1):
        front_left, front_right, back_right, back_left = RectUtils.get_car_vertice(pos, angle, car_size)

        ImageDraw.Draw(image).polygon((front_left, front_right, back_right, back_left), fill='blue',
                                      outline=(200, 0, 0))

        radar_pos = RectUtils.radar_pos((front_left, front_right, back_right, back_left))
        ImageUtils.draw_radar(image, radar_pos, angle)

        # for r in radar_pos:
        #     ra = 1
        #     ImageDraw.Draw(image).ellipse((r[0] - ra, r[1] - ra, r[0] + ra, r[1] + ra), fill=(255, 0, 0, 0))

        r = radar_pos[6]
        ImageDraw.Draw(image).line([(r[0], r[1]), (200,200)], fill=(255,0,0), width=0, joint=None)

        # image.rotate(45)
        # img = Image.open('car.png')
        # img = img.rotate(angle)
        # img = img.resize((50, 50)).convert("RGBA")
        # image.paste(img, (100, 100), mask=img)
        # image.show()


def test_draw_car():
    map = ImageUtils.draw_map()
    # for i in range(10, 200, 50):
    # ImageUtils.draw_car(map, (100, 10), 10)
    # ImageUtils.draw_car(map, (100, 50), 45)
    # ImageUtils.draw_car(map, (100, 90), 90)
    ImageUtils.draw_car(map, (100, 180), 45)
    map.show()


def test_sort_vertice():
    rect1 = [[0, 0], [0, 1], [1, 0], [1, 1]]
    rect2 = [[0, 0], [1, 1], [1, 0], [1, 0]]
    rect3 = [[0, 0], [1, 0], [0, 1], [1, 1]]
    print(RectUtils.sort_vertice(rect1))
    print(RectUtils.sort_vertice(rect2))
    print(RectUtils.sort_vertice(rect3))


def test_draw_rect():
    tmp = ImageUtils.make_image(500, 500)
    ImageUtils.draw_rect(tmp, ((0, 0), (0, 100), (100, 100), (100, 0)), color=(0, 200, 0))
    tmp.show()


if __name__ == "__main__":
    test_draw_car()

    # test_draw_rect()
    # test_sort_vertice()
