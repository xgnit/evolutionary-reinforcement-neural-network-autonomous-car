
from Utils import *
import numpy as np
from Config import Config

class Map:



    def __init__(self):
        self.margin = 10

        # self.bg = ImageUtils.make_image(self.size, self.size)
        # ImageUtils.draw_rect(self.bg, ((0, 0), (0, 100), (100, 100), (100, 0)), color=(0, 200, 0))
        # self.bg.show()
        self.path_generator()

    def map_frame_generator(self, map_size, path_width):

        grid_no = map_size//path_width
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


    def path_generator(self):

        def cluster_size(map_size, path_width):
            pass


        map_size = Config.map_size()
        path_size = Config.path_width()
        map_frame = self.map_frame_generator(map_size, path_size)
        print(map_frame)






if __name__ == "__main__":
    m = Map()