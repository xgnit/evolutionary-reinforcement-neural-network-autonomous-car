
from Utils import ImageUtils, ColliderUtils, MiscUtils
import os, neat
import numpy as np
from Config import Config
import shutil
import os
from evolutionary_trainer import Trainer
import torch.nn.functional as F

import torch
import torch.nn as nn

LIDAR_NO = 5
GAMMA = 0.9
MEM_CAP = 2000
LR = 0.1
EPSILON = 0.9
BATCH = 32
TARGET_REPLACE_ITER = 100
# ENV_A_SHAPE= None
N_ACTIONS = 5


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        hidden_nodes = 30
        self.hidden = nn.Linear(LIDAR_NO, hidden_nodes)
        self.hidden.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(hidden_nodes, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = torch.relu(self.hidden(x))
        x = self.out(x)
        return x

class ReNet:
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()

        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((MEM_CAP, LIDAR_NO * 2 + 2))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)

        if np.random.uniform() < EPSILON:
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0]
        else:
            action = np.random.randint(0, N_ACTIONS)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % MEM_CAP
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1


        sample_index = np.random.choice(MEM_CAP, BATCH)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :LIDAR_NO])
        b_a = torch.LongTensor(b_memory[:, LIDAR_NO:LIDAR_NO+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, LIDAR_NO+1:LIDAR_NO+2])
        b_s_ = torch.FloatTensor(b_memory[:, -LIDAR_NO:])


        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class DriveSim:
    sim_cnt = 0

    def __init__(self, map):
        self.map = map
        self.pos = Config.start_pos()
        self.orientation = 0
        self.movie = []
        self.result_format = '.gif'
        self.colliders = self.map.collider_lines
        self.wall_rects = self.map.wall_rects
        self.travel_range = 0


    def step(self, action):

        def update_range(range, old_pos, new_pos):
            res = np.array(old_pos) - np.array(new_pos)
            return range + (res[0]**2 + res[1]**2)**0.5

        m = self.map.draw_map_bg()
        ImageUtils.draw_car(m, self.pos, self.orientation, self.colliders)
        self.movie.append(m)
        old_pos = self.pos

        self.pos = MiscUtils.get_next_pos(self.pos, self.orientation, Config.car_speed())
        self.orientation += 4*(action - N_ACTIONS//2)
        self.travel_range = update_range(self.travel_range, old_pos, self.pos)

        radar_data = ImageUtils.radar_data(self.pos, self.orientation, self.colliders)
        done = False
        if ColliderUtils.collision((self.pos, self.orientation), self.wall_rects) or self.travel_range > Config.max_fitness():
            done = True

        l1, l2, l3, l4, l5 = radar_data

        r1 = -3 * abs(l1 - l5)
        r2 = -1 * abs(l2 - l4)
        if l3 < Config.path_width():
            r3 = - Config.path_width()
        else:
            r3 = Config.path_width()
        r = r1 + r2 + r3

        return radar_data, done, r

    def reset(self):
        self.pos = Config.start_pos()
        self.orientation = 0
        self.movie = []
        self.travel_range = 0
        radar_data = ImageUtils.radar_data(self.pos, self.orientation, self.colliders)
        return radar_data

    def save_gif(self):
        if not os.path.exists(Config.result_dir()):
            os.makedirs(Config.result_dir())
        gif_name = 'res/reinforcement_{}{}'.format(DriveSim.sim_cnt, self.result_format)
        ImageUtils.save_img_lst_2_gif(self.movie, gif_name)
        DriveSim.sim_cnt += 1
        self.reset()


class ReinforcementTrainer(Trainer):
    def __init__(self, map):
        super().__init__(map)
        self.renet = ReNet()
        self.sim = DriveSim(map)
        self.best_score = 0

    def train(self):
        MiscUtils.rm_hist()

        for epi in range(20000):
            s = self.sim.reset()
            if self.best_score > Config.max_fitness():
                break

            while 1:
                a = self.renet.choose_action(s)
                s_, done, r = self.sim.step(a)

                self.renet.store_transition(s, a, r, s_)

                if self.renet.memory_counter > MEM_CAP:
                    self.renet.learn()
                    if done:
                        print('episode: {} score: {}'.format(epi, self.sim.travel_range))
                        if self.sim.travel_range > self.best_score:
                            self.best_score = self.sim.travel_range
                            print('*'*20)
                            print('New best score! score: {}'.format(self.best_score))
                            print('*'*20)
                            self.sim.save_gif()

                if done:
                    break
                s = s_

        MiscUtils.finish_info()


