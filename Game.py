

from Map import Map
from Utils import ImageUtils, ColliderUtils

class Game:

    def __init__(self):
        self.map = Map()
        self.colliders = self.map.collider_lines
        self.wall_rects = self.map.wall_rects
        self.test_game()

    def test_game(self):

        movie = []
        for i in range(360):
            m = self.map.draw_map_bg()



            ImageUtils.draw_car(m, (100, 30), i, self.colliders)
            movie.append(m)
        ImageUtils.save_img_lst_2_gif(movie, 'out.gif')
        ImageUtils.play_gif('out.gif')


if __name__ == "__main__":
    Game().test_game()

