
from Map import Map
from Utils import ImageUtils, ColliderUtils
import os, neat

class Game:

    def __init__(self):
        self.map = Map()
        self.colliders = self.map.collider_lines
        self.wall_rects = self.map.wall_rects
        self.test_game()

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

