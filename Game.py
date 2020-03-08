
from Map import Map
from Utils import ImageUtils, ColliderUtils, MiscUtils
import os, neat
import numpy as np
import time

class Game:

    def __init__(self):
        self.map = Map()
        self.colliders = self.map.collider_lines
        self.wall_rects = self.map.wall_rects
        self.result_file = 'out.gif'
        self.best = 0


    def single_drive_with_nn(self, nn):


        def update_range(range, old_pos, new_pos):
            res = np.array(old_pos) - np.array(new_pos)
            return range + (res[0]**2 + res[1]**2)**0.5

        movie = []
        travel_range = 0
        pos = (100, 30)
        orientation = 0
        speed = 2

        while 1:

            if ColliderUtils.collision((pos, orientation), self.wall_rects):

                if travel_range > self.best:
                    ImageUtils.save_img_lst_2_gif(movie, 'res/' + str(time.time()) + self.result_file)
                    self.best = travel_range
                # ImageUtils.play_gif(self.result_file)
                return travel_range
            m = self.map.draw_map_bg()
            ImageUtils.draw_car(m, pos, orientation, self.colliders)
            movie.append(m)
            radar_data = ImageUtils.radar_data(pos, orientation, self.colliders)

            pos_new = MiscUtils.get_next_pos(pos, orientation, speed)
            travel_range = update_range(travel_range, pos, pos_new)
            pos = pos_new
            orientation += nn.activate(radar_data)[0]


    def run(self):

        def eval_genomes(genomes, config):
            for genome_id, genome in genomes:  # for each individual
                genome.fitness = 0
                net = neat.nn.FeedForwardNetwork.create(genome, config)
                genome.fitness = self.single_drive_with_nn(net)


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
        winner = p.run(eval_genomes, 300)




    def single_drive_single_car(self):

        def update_orientation(old_orientation):
            return old_orientation + 0.3

        movie = []
        pos = (100, 30)
        orientation = -15
        speed = 2

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

    def single_drive_single_multiple_cars(self):
        pass


    def test_game(self):

        movie = []
        for i in range(50):
            m = self.map.draw_map_bg()
            ImageUtils.draw_car(m, (100, 30), i, self.colliders)
            movie.append(m)
        ImageUtils.save_img_lst_2_gif(movie, self.result_file)
        ImageUtils.play_gif(self.result_file)


if __name__ == "__main__":
    Game().run()

