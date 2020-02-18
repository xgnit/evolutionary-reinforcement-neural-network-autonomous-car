from PIL import Image, ImageDraw
import math
from Config import Config
import numpy as np
from copy import deepcopy as dcopy
import pyglet
from Map import Map

class ColliderUtils:
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
        return np.array([front_left, front_right, back_right, back_left])

    # the order might be wrong, but, it draws the correct rectangle, just be careful with the radar position
    # I am too lazy to correct the name on this function
    @staticmethod
    def get_car_vertice(pos, angle, car_size=1):
        # front_left, front_right, back_left, back_right = RectUtils.get_car_vertice_no_rotate(pos, car_size)
        vert = ColliderUtils.get_car_vertice_no_rotate(pos, car_size)
        angle = np.full((1, len(vert)), angle)
        # x, y = pos
        x, y = np.full_like(angle, pos[0]), np.full_like(angle, pos[1])
        return np.concatenate(((vert[:,0]-x) * np.cos(angle) + (vert[:,1] - y) * np.sin(angle) + x ,
                         -(vert[:,0]-x) * np.sin(angle) + (vert[:,1] - y) * np.cos(angle) + y ))

    @staticmethod
    def generate_block_vertice(row_start, row_end, col_start, col_end):
        return ((col_start * Config.path_width(), row_start * Config.path_width()),
                 (col_start * Config.path_width(), (row_end+1) * Config.path_width()),
                 ((col_end+1) * Config.path_width(), (row_end+1) * Config.path_width()),
                 ((col_end+1) * Config.path_width(), row_start * Config.path_width()))

    @staticmethod
    def radar_pos(vertice):
        fr_l, fr_r, bc_r, bc_l = np.split(vertice, 4, axis = 1)
        r1 = dcopy((fr_l + bc_l)/2)
        r2 = dcopy(fr_l)
        r3 = dcopy((fr_l+fr_r)/2)
        r4 = dcopy(fr_r)
        r5 = dcopy((fr_r+bc_r)/2)
        return np.concatenate((r1.reshape(1,2), r2.reshape(1,2), r3.reshape(1,2), r4.reshape(1,2), r5.reshape(1,2)))

    @staticmethod
    def collider_lines_from_path_rects(path_rects):
        pathes = np.array(path_rects)
        left_side, bot_side, right_side, top_side = pathes[:,[0,1]], pathes[:,[1,2]], pathes[:,[2,3]], pathes[:,[3,1]]





        print(1)

class ImageUtils:

    @staticmethod
    def draw_map():
        # image = ImageUtils.make_image(size, size)
        # return ImageUtils.draw_rect(image, (169,169,169))
        return Image.new('RGB', (Config.map_size(), Config.map_size()), Config.bg_rbg())

    @staticmethod
    def make_image(i, j):
        return Image.new("RGB", (i, j), "white")

    @staticmethod
    def draw_rect(image, vertice, color=(200, 200, 200), outline=None):
        # im_ = image.load()
        ImageDraw.Draw(image).polygon(vertice, fill=color, outline=outline)

    @classmethod
    def draw_path(cls, image, vertice):
        cls.draw_rect(image, vertice, color=Config.path_gray_rbg())

    @classmethod
    def draw_wall(cls, image, vertice):
        cls.draw_rect(image, vertice, color=Config.wall_rbg())

    @staticmethod
    def draw_laser(image, pos):
        ImageDraw.Draw(image).line(pos, fill=(255,0,0), width=1, joint=None)


    @staticmethod
    def draw_radar(image, radars, angle, collider_lines):
        r = 200
        angles = np.array([-angle + math.pi/2, -angle + math.pi/4, -angle, -angle - math.pi/4, -angle - math.pi/2])
        rx, ry = radars[:, 0] + r * np.cos(angles), radars[:, 1] + r * np.sin(angles)
        for ra, x, y in zip(radars, rx, ry):
            ImageUtils.draw_laser(image, [tuple(ra), (x, y)])


    @staticmethod
    def draw_car(image, pos, angle, collider_lines, car_size=1):
        angle = math.radians(angle)

        # front_left, front_right, back_right, back_left = RectUtils.get_car_vertice(pos, angle, car_size)
        vert = ColliderUtils.get_car_vertice(pos, angle, car_size)

        ImageDraw.Draw(image).polygon((tuple(vert[:,0]), tuple(vert[:,1]), tuple(vert[:,2]), tuple(vert[:,3])), fill='blue',
                                      outline=(200, 0, 0))

        radar_pos = ColliderUtils.radar_pos(vert)
        ImageUtils.draw_radar(image, radar_pos, angle, collider_lines)

        # for r in radar_pos:
        #     ra = 1
        #     ImageDraw.Draw(image).ellipse((r[0] - ra, r[1] - ra, r[0] + ra, r[1] + ra), fill=(255, 0, 0, 0))

        # r = radar_pos[0]
        # ImageDraw.Draw(image).line([(r[0], r[1]), (200,200)], fill=(255,0,0), width=0, joint=None)

        # image.rotate(45)
        # img = Image.open('car.png')
        # img = img.rotate(angle)
        # img = img.resize((50, 50)).convert("RGBA")
        # image.paste(img, (100, 100), mask=img)
        # image.show()

    @staticmethod
    def play_gif(path):
        animation = pyglet.resource.animation(path)
        sprite = pyglet.sprite.Sprite(animation)
        win = pyglet.window.Window(width=sprite.width, height=sprite.height)
        green = 0, 1, 0, 1
        pyglet.gl.glClearColor(*green)

        @win.event
        def on_draw():
            win.clear()
            sprite.draw()

        pyglet.app.run()


    @staticmethod
    def save_img_lst_2_gif(imgs):
        imgs[0].save('out.gif',
                    save_all=True,
                    append_images=out[1::2],
                    duration=1000 * 0.08,
                    loop=0)


class MiscUtils:

    @staticmethod
    def merge_neighbors(np_array):
        if np_array.size > 0:
            spliter = ','
            tmp = np_array.copy()
            res = str(tmp[0])
            for i in range(1, len(tmp)):
                if tmp[i] != tmp[i-1] + 1:
                    res += spliter + str(tmp[i])
                else:   res += str(tmp[i])
            res = res.split(spliter)
            res = [(int(x[0]), int(x[-1])) for x in res]
            return res
        else:   return []


def test_draw_car():

    # map = ImageUtils.draw_map()
    # ImageUtils.draw_car(map, (100, 100), )
    # map.show()

    out = []
    for i in range(150):
        map = ImageUtils.draw_map()
        ImageUtils.draw_car(map, (i, 100), 0)
        out.append(map)
    out[0].save('out.gif',
                   save_all=True,
                   append_images=out[1::2],
                   duration=1000*0.08,
                   loop=0)
    ImageUtils.play_gif('out.gif')



def test_sort_vertice():
    rect1 = [[0, 0], [0, 1], [1, 0], [1, 1]]
    rect2 = [[0, 0], [1, 1], [1, 0], [1, 0]]
    rect3 = [[0, 0], [1, 0], [0, 1], [1, 1]]
    print(ColliderUtils.sort_vertice(rect1))
    print(ColliderUtils.sort_vertice(rect2))
    print(ColliderUtils.sort_vertice(rect3))


def test_draw_rect():
    tmp = ImageUtils.make_image(500, 500)
    ImageUtils.draw_rect(tmp, ((0, 0), (0, 100), (100, 100), (100, 0)), color=(0, 200, 0))
    tmp.show()


if __name__ == "__main__":
    test_draw_car()

    # test_draw_rect()
    # test_sort_vertice()
