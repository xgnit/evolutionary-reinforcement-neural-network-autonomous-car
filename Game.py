
from Map import Map
from Utils import ImageUtils, ColliderUtils, MiscUtils
import os, neat
import numpy as np
import time
from Config import Config

class Game:

    def __init__(self):
        self.map = Map()
        self.colliders = self.map.collider_lines
        self.wall_rects = self.map.wall_rects
        self.result_file = 'out.gif'
        self.best = 0

    def single_drive_with_whole_population(self, nn_list):

        marker = np.zeros_like(nn_list, dtype=bool)

        def update_range(range, old_pos, new_pos):
            res = np.array(old_pos) - np.array(new_pos)
            return range + (res[0]**2 + res[1]**2)**0.5

        travel_range = np.zeros_like(nn_list)
        movie = []
        # pos = np.full_like(nn_list, (100, 30))
        pos = np.empty_like(nn_list)
        for i in range(len(pos)):
            pos[i] = (100, 30)
        orientation = np.full_like(nn_list, 0)

        while 1:

            if np.all(marker):
                import os
                if not os.path.exists('res'):
                    os.makedirs('res')

                ImageUtils.save_img_lst_2_gif(movie, 'res/' + str(time.time()) + self.result_file)
                return travel_range

            m = self.map.draw_map_bg()
            for i in range(len(nn_list)):

                p, o, nn = pos[i], orientation[i], nn_list[i]
                if marker[i]:
                    ImageUtils.draw_car(m, p, o, self.colliders, draw_radar=False)
                    continue

                if ColliderUtils.collision((p, o), self.wall_rects) or travel_range[i] > 3000:
                    marker[i] = True

                ImageUtils.draw_car(m, p, o, self.colliders)

                # if marker[i]:
                #     continue

                radar_data = ImageUtils.radar_data(p, o, self.colliders)
                pos_new = MiscUtils.get_next_pos(p, o, Config.car_speed())
                travel_range[i] = update_range(travel_range[i], p, pos_new)
                pos[i] = pos_new

                left, right = nn.activate(radar_data)

                clamp = 5
                turning = left - right
                turning = clamp if turning > clamp else turning
                turning = -1 * clamp if turning < -1 * clamp else turning
                orientation[i] += turning

            movie.append(m)


    def eval_genomes(self, genomes, config):
        nn_list = []
        for genome_id, genome in genomes:
            genome.fitness = 0
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            nn_list.append(net)

        pop_fitness = self.single_drive_with_whole_population(nn_list)

        for i, genome in enumerate(genomes):
            genome = genome[1]
            genome.fitness = pop_fitness[i]

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
        winner = p.run(self.eval_genomes, 5000)



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

