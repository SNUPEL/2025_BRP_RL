import numpy as np
import torch
from DP_Retrieval import *


class Heuristic():
    def __init__(self, grid_size, TP_type_len, mod):
        super(Heuristic, self).__init__()
        self.grid_size = grid_size  # Grid dimensions (rows, columns)
        self.TP_type_len = TP_type_len  # Number of transporter types (or capacity levels)
        self.mod = mod  # Mode for decision policy ('ASR', 'Random', 'BLF')

    def find_coordinates(self, m, arr):
        # Find indices where the first channel value is less than m and other channels are non-zero
        indices = np.argwhere((arr[:, :, 0] < m) & (arr[:, :, 1:].sum() > 0))
        return indices.tolist()  # Convert to list of [x, y] coordinates

    def find_space(self, arr):
        # Find indices where the first channel is zero (indicating available space)
        indices = np.argwhere((arr[:, :, 0] == 0))
        return indices.tolist()

    def Locate(self, grid_tensor, block_tensor, mask_tensor, ans=None):
        r = self.grid_size[0]
        c = self.grid_size[1]

        # Convert tensors to numpy arrays
        grid = np.array(grid_tensor.cpu()).reshape(r, c, -1)
        grid[:, :, 0] = grid[:, :, 0] * 500.0  # Scale the exit ID
        block = np.array(block_tensor[:, 0, :].cpu()).flatten()
        block[0] = block[0] * 500.0  # Scale the target exit ID
        mask = np.array(mask_tensor.cpu()).reshape(r, c, 1)

        if self.mod == 'ASR':
            target_exit = block[0]
            cal_block = self.find_coordinates(target_exit, grid)  # Blocks with earlier exits
            candidate_space = self.find_space(mask)  # All available placement spaces
            score = []

            if len(cal_block) == 0:
                # If there are no earlier blocks, choose bottom-left (default)
                return torch.tensor(1), torch.tensor(r * c - c)

            TP_capacity_num = int(block[1:].sum())  # Get block's capacity level (one-hot vector sum)

            for space in candidate_space:
                temp_grid = grid.copy()
                temp_grid[space[0], space[1], :] = block  # Tentatively place block in the candidate space
                ave_score = 0
                step = 0

                for bl in cal_block:
                    for TP_capacity in range(self.TP_type_len - TP_capacity_num, self.TP_type_len):
                        count = Count_retrieval(temp_grid, TP_capacity, bl)  # Count retrieval steps for this setting
                        ave_score += count
                        step += 1
                score.append(ave_score / step)  # Average retrieval cost

            min_index = np.argmin(np.array(score))  # Find space with minimum retrieval cost
            return torch.tensor(1), torch.tensor(candidate_space[min_index][0] * c + candidate_space[min_index][1])

        if self.mod == 'Random':
            candidate_space = self.find_space(mask)  # Find all available spaces
            random_index = np.random.randint(0, len(candidate_space))  # Randomly select one
            return torch.tensor(1), torch.tensor(
                candidate_space[random_index][0] * c + candidate_space[random_index][1])

        if self.mod == 'BLF':
            candidate_space = self.find_space(mask)  # Find all available spaces
            # Select the bottom-leftmost coordinate (maximize row, minimize column)
            best_x, best_y = max(candidate_space, key=lambda p: (p[0], -p[1]))
            return torch.tensor(1), torch.tensor(best_x * c + best_y)
