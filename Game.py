
from Map import Map
from Utils import ImageUtils, ColliderUtils, MiscUtils
import os, neat

class Game:

    def __init__(self):
        self.map = Map()
        self.colliders = self.map.collider_lines
        self.wall_rects = self.map.wall_rects
        self.result_file = 'out.gif'



    def eval_genomes(self):
        pass

    def run(self):
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, 'config-feedforward')
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config_path)
        p = neat.Population(config)
        p.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)
        p.add_reporter(neat.Checkpointer(50))
        winner = p.run(self.eval_genomes, 300)



    def single_drive(self):

        def update_orientation(old_orientation):
            return old_orientation + 1

        movie = []
        pos = (100, 30)
        orientation = -10
        speed = 10

        while 1:

            if ColliderUtils.collision((pos, orientation), self.wall_rects):
                break
            m = self.map.draw_map_bg()
            ImageUtils.draw_car(m, pos, orientation, self.colliders)
            movie.append(m)
            pos = MiscUtils.get_next_pos(pos, orientation, speed)
            orientation = update_orientation(orientation)


        ImageUtils.save_img_lst_2_gif(movie, self.result_file)
        ImageUtils.play_gif(self.result_file)


    def test_game(self):

        movie = []
        for i in range(50):
            m = self.map.draw_map_bg()
            ImageUtils.draw_car(m, (100, 30), i, self.colliders)
            movie.append(m)
        ImageUtils.save_img_lst_2_gif(movie, self.result_file)
        ImageUtils.play_gif(self.result_file)


if __name__ == "__main__":
    Game().single_drive()

