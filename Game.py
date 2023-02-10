
from Map import Map
import argparse
from evolutionary_trainer import EvolutionaryTrainer
from reinforcement_trainer import ReinforcementTrainer

class Game:
    evo_auto_play = False

    def __init__(self):
        self.map = Map()
        self.colliders = self.map.collider_lines
        self.wall_rects = self.map.wall_rects
        self.result_file = '.gif'
        self.best = 0

    def run_reinfocement(self):
        trainer = ReinforcementTrainer(self.map)
        trainer.train()

    def run_evo(self):
        trainer = EvolutionaryTrainer(self.map)
        trainer.train()

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rl', type=str2bool, default=False, help='Will train the car with reinforcement learning, \
    otherwise will train the car with genetic algorithm')
    parser.add_argument('--auto_play', type=str2bool, default=False, help='Will play the gif after each generation automatically, \
    only applicable for evolutionary method')

    opt = parser.parse_args()
    if opt.auto_play:
        Game.evo_auto_play = True

    if opt.rl:
        Game().run_reinfocement()
    else:
        Game().run_evo()

if __name__ == "__main__":
    main()

