from Utils import *
import numpy as np
from Config import Config
from copy import deepcopy as dcopy

class Map:

    def __init__(self):
        self.margin = 10
        self.grid_size = Config.map_size() // Config.path_width()

        self.wall_colider = None

        # self.bg = ImageUtils.make_image(self.size, self.size)
        # ImageUtils.draw_rect(self.bg, ((0, 0), (0, 100), (100, 100), (100, 0)), color=(0, 200, 0))
        # self.bg.show()
        self.generate_map()

    def map_frame_generator(self):

        grid_no = self.grid_size
        if grid_no < 3:
            raise RuntimeError('This map has less than 3 grids, cannot generate path')
        board = np.zeros([grid_no, grid_no])
        board[:, 0] = 1
        # board[0:grid_no-1,1] = 2
        target_pos = np.random.randint(grid_no - 3, grid_no, 1)[0]
        board[0, 0:target_pos + 1] = 1
        # board[1, 1:target_pos] = 2
        start_col = np.random.randint(3, grid_no, 1)[0]
        board[grid_no - 1, 1: start_col + 1] = 1

        start_pos = (grid_no - 1, start_col)

        def dfs(pos):
            row, col = pos
            if row < 4:
                board[row, min(col, target_pos): max(col, target_pos)] = 1
                board[: row + 1, target_pos] = 1
                return

            if row < grid_no - 1 and 1 == board[row + 1, col]:
                if col > grid_no // 2:
                    end_col = np.random.randint(2, grid_no // 2, 1)[0]
                    board[row, end_col: col + 1] = 1
                else:
                    end_col = np.random.randint(grid_no // 2, grid_no, 1)[0]
                    board[row, col: end_col + 1] = 1
                dfs((row, end_col))
            else:
                end_row = np.random.randint(2, row - 1, 1)[0]
                if row - end_row > grid_no // 2:
                    end_row = row - np.abs(end_row - row) // 2
                board[end_row: row, col] = 1
                dfs((end_row, col))

        dfs(start_pos)
        return board

    def get_tile_rects(self, map_frame, tile_number):
        res = []
        for row in range(len(map_frame)):
            walls = np.where(tile_number == map_frame[row])
            walls = MiscUtils.merge_neighbors(walls[0])
            for w in walls:
                res.append(ColliderUtils.generate_block_vertice(row, row, w[0], w[1]))
        return res

    def get_path_rect(self, map_frame):
        return self.get_tile_rects(map_frame, 1)

    def get_wall_rect(self, map_frame):
        return self.get_tile_rects(map_frame, 0)

    def draw_blocks(self, map, blocks, draw_method):
        for b in blocks:
            draw_method(map, b)

    def get_boundaries(self):
        bottom_wall = (
            (0, self.grid_size * Config.path_width()),
            (0, Config.map_size()),
            (Config.map_size(), Config.map_size()),
            (Config.map_size(), self.grid_size * Config.path_width())
        )
        right_wall = (
            (self.grid_size * Config.path_width(), 0),
            (self.grid_size * Config.path_width(), Config.map_size()),
            (Config.map_size(), Config.map_size()),
            (Config.map_size(), 0)
        )
        top_wall = (
            (0,0),
            (0,0),
            (self.grid_size * Config.path_width(), 0),
            (self.grid_size * Config.path_width(), 0)
        )
        left_wall = (
            (0, 0),
            (0, self.grid_size * Config.path_width()),
            (0, self.grid_size * Config.path_width()),
            (0, 0)
        )
        return bottom_wall, right_wall, top_wall, left_wall


    def generate_map(self):

        map_frame = self.map_frame_generator()
        # path_rects = [RectUtils.generate_block_vertice(0, self.grid_size-1, 0, 0)] + \
        #              self.get_path_rect(map_frame)
        path_rects = self.get_path_rect(map_frame)

        print(map_frame)
        wall_rects = self.get_wall_rect(map_frame)
        self.wall_colider = dcopy(wall_rects)

        bot_wall, right_wall, top_wall, left_wall = self.get_boundaries()
        self.wall_colider.append(bot_wall)
        self.wall_colider.append(right_wall)
        self.wall_colider.append(top_wall)
        self.wall_colider.append(left_wall)


        # self.bg = ImageUtils.make_image(Config.map_size(), Config.map_size())
        map = ImageUtils.draw_map()

        collider_lines = ColliderUtils.collider_lines_from_path_rects(path_rects)

        self.draw_blocks(map, path_rects, ImageUtils.draw_path)
        self.draw_blocks(map, wall_rects, ImageUtils.draw_wall)

        ImageUtils.draw_car(map, (20, 20), -30, collider_lines)

        ImageUtils.draw_rect(map, bot_wall, color='white')
        ImageUtils.draw_rect(map, right_wall, color='white')
        # ImageUtils.draw_rect(map, path_rects[0], color=(0, 200, 0))

        # ImageUtils.draw_rect(self.bg, ((0, 0), (0, 100), (100, 100), (100, 0)), color=(0, 200, 0))
        # self.bg.show()
        map.show()

        print(1)


if __name__ == "__main__":
    m = Map()
