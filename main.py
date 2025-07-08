import numpy as np
from DP_Retrieval import *
from Location import *

from Location_Heuristic import *
from collections import OrderedDict
#import vessl
class Stockyard_simulation:
    def __init__(self, yard_size, initial_block, lam, weight, TP_type, Block_per_Day, mod):
        self.yard_size = yard_size
        self.initial_block = initial_block
        self.lam = lam
        self.weight_distribution = weight
        self.TP_type = TP_type
        self.mod = mod
        self.Block_per_Day = Block_per_Day

    # One-hot encode weight values based on thresholds
    def one_hot_encode(self, weight, thresholds):
        return [1 if weight < t else 0 for t in thresholds]

    # Initialize the yard grid with random block positions, weights, and retrieval times
    def Generate_grid(self, seed):
        if seed is not None:
            np.random.seed(seed)  # Set seed for reproducibility

        grid = np.zeros((self.yard_size[0], self.yard_size[1], len(self.TP_type)+2), dtype=int)
        grid_save = np.zeros((self.yard_size[0], self.yard_size[1], 2), dtype=int)
        positions = np.random.choice(self.yard_size[0] * self.yard_size[1], self.initial_block, replace=False)
        blocks = np.zeros((self.initial_block, 2))

        for e, pos in enumerate(positions):
            x, y = divmod(pos, self.yard_size[0])
            time = np.random.exponential(scale=1/self.lam)
            weight = np.random.randint(100, 501)
            embedded_weight = self.one_hot_encode(weight, self.TP_type)
            grid_save[x, y, 0] = time
            grid_save[x, y, 1] = weight
            grid[x, y, 0] = time
            grid[x, y, 1:-1] = embedded_weight
            blocks[e, 0] = time
            blocks[e, 1] = weight

        return grid, grid_save, blocks

    # Generate new blocks to be added each day
    def Create_blocks(self):
        block_num = np.random.randint(self.Block_per_Day[0], self.Block_per_Day[1])
        blocks = np.zeros((block_num, 2))

        for i in range(block_num):
            blocks[i, 0] = np.random.exponential(scale=1/self.lam)
            blocks[i, 1] = np.random.randint(100, 501)
        return blocks

    # Create a mask to identify feasible positions for placement
    def Create_mask(self, grid, TP_capa):
        r, c, f = grid.shape
        mask = (grid[:, :, 1:1+len(self.TP_type)].sum(axis=2) > 0).astype(np.uint8)
        mask = mask[:, :, np.newaxis].copy().reshape(r, c)
        rows, cols = len(mask), len(mask[0])
        visited = [[False] * cols for _ in range(rows)]
        new_grid = [[1] * cols for _ in range(rows)]
        queue = deque()

        for x in range(cols):
            if mask[0][x] == 0:
                queue.append((0, x))
                visited[0][x] = True
                new_grid[0][x] = 0

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while queue:
            y, x = queue.popleft()
            for dy, dx in directions:
                ny, nx = y + dy, x + dx
                if 0 <= ny < rows and 0 <= nx < cols and not visited[ny][nx] and mask[ny][nx] == 0:
                    queue.append((ny, nx))
                    visited[ny][nx] = True
                    new_grid[ny][nx] = 0

        new_grid = np.array(new_grid)
        need_retrieve = False
        if new_grid[0].sum() == cols:
            need_retrieve = True
            check_list = np.argwhere(mask == 0)
            check_num = np.zeros(len(check_list))

            for e, space in enumerate(check_list):
                count, _ = Count_retrieval(grid, TP_capa, space)
                check_num[e] = count

            min_value = check_num.min()
            index = np.argwhere(check_num == min_value).flatten()
            for i in index:
                new_grid[check_list[i, 0], check_list[i, 1]] = 0
        return new_grid, need_retrieve

    # Encode block retrieval time and weights
    def block_encoding(self, arr, thresholds):
        column1 = arr[:, [0]]
        column2 = arr[:, 1]
        one_hot_matrix = (column2[:, np.newaxis] <= thresholds).astype(int)
        return np.hstack((column1, one_hot_matrix))

    # Find retrievable blocks (based on threshold and type)
    def find_indices(self, grid):
        n = len(self.TP_type)
        condition_1 = grid[:, :, 0] <= 100
        condition_2 = np.any(grid[:, :, 1:n+1] == 1, axis=2)
        indices = np.where(condition_1 & condition_2)
        return indices

    # Main function to run the simulation environment
    def Run_simulation(self, simulation_day, lookahead_num, ppo, grid, total_block, total_block_encoded):
        current_time = 0
        total_rearrangement = 0
        grids, blocks, actions, dones, masks, probs, block_lefts = [], [], [], [], [], [], []
        max_length = len(total_block_encoded)
        rewards = [0]
        step = 0
        block_num = 0
        block_left_num = max_length

        for i in range(simulation_day):
            block_located = total_block_encoded[block_num:block_num+len(total_block[i])]

            for e, row in enumerate(block_located):
                cc = np.where(grid[:, :, 1:1+len(self.TP_type)].sum(axis=2) > 0)
                if np.array(cc).shape[1] == grid.shape[0] * grid.shape[1]:
                    continue

                grids.append(grid.copy())
                block_lefts.append(block_left_num)
                block_left_num -= 1

                blocks_vec = total_block_encoded[block_num+e:int(min(block_num+e+lookahead_num, max_length)), :].copy()
                if len(blocks_vec) < lookahead_num:
                    blocks_vec_temp = np.zeros((lookahead_num, 1+len(self.TP_type)))
                    blocks_vec_temp[:, 0] = 250
                    blocks_vec_temp[:, 1:int(1+(1+len(self.TP_type))/2.0)] = 1
                    blocks_vec_temp[:len(blocks_vec)] = blocks_vec
                    blocks_vec = blocks_vec_temp

                blocks.append(blocks_vec.copy())
                mask, need_retrieval = self.Create_mask(grid.copy(), TP_capa=len(self.TP_type)-1)
                masks.append(mask.reshape(-1, 1).copy())

                grid_tensor = torch.tensor(grid[:, :, :-1].reshape(1, grid.shape[0], grid.shape[1], -1), dtype=torch.float32).to(device)
                grid_tensor[:, :, 0] = grid_tensor[:, :, 0] / 500.0
                block_tensor = torch.tensor(blocks_vec.reshape(1, lookahead_num, -1), dtype=torch.float32).to(device)
                block_tensor[:, :, 0] = block_tensor[:, :, 0] / 500.0
                mask_tensor = torch.tensor(mask.reshape(1, -1, 1), dtype=torch.float32).to(device)

                prob, coord = ppo.Locate(grid_tensor, block_tensor, mask_tensor, ans=None)
                probs.append(prob.item())
                actions.append(coord.item())
                dones.append(0)
                rewards.append(0)

                r = coord.item() // grid.shape[0]
                c = coord.item() % grid.shape[0]
                target_block = [r, c]
                step += 1

                grid[r, c, 0] = total_block_encoded[block_num+e, 0]
                grid[r, c, 1:-1] = total_block_encoded[block_num+e, 1:]
                grid[r, c, -1] = step

                if need_retrieval:
                    ispossible, rearrange_num, end_grid, step, grids, blocks, actions, rewards, dones, masks, probs, block_lefts = Retrieval(grid.copy(), len(self.TP_type)-1, target_block.copy(), ppo, step, grids, blocks, block_lefts, block_left_num, actions, rewards, dones, masks, probs, lookahead_num, len(self.TP_type), 'NOR')
                    total_rearrangement += rearrange_num
                    grid = end_grid.copy()

            while True:
                indices = self.find_indices(grid)
                if len(indices[0]) == 0:
                    break
                values = grid[indices[0], indices[1], 0]
                target_index = np.argmin(values)
                target_r = indices[0][target_index]
                target_c = indices[1][target_index]
                target_block = [target_r, target_c]
                TP_type_len = len(self.TP_type)
                TP_capacity = np.random.randint(TP_type_len - grid[target_r, target_c, 1:-1].sum(), TP_type_len)

                ispossible, rearrange_num, end_grid, step, grids, blocks, actions, rewards, dones, masks, probs, block_lefts = Retrieval(grid, TP_capacity, target_block, ppo, step, grids, blocks, block_lefts, block_left_num, actions, rewards, dones, masks, probs, lookahead_num, TP_type_len, 'OR')
                while not ispossible:
                    TP_capacity = np.random.randint(TP_type_len - grid[target_r, target_c, 1:-1].sum(), TP_type_len)
                    ispossible, rearrange_num, end_grid, step, grids, blocks, actions, rewards, dones, masks, probs, block_lefts = Retrieval(grid, TP_capacity, target_block, ppo, step, grids, blocks, block_lefts, block_left_num, actions, rewards, dones, masks, probs, lookahead_num, TP_type_len, 'OR')

                total_rearrangement += rearrange_num
                grid = end_grid.copy()

            grid[:, :, 0] -= 100
            grid[:, :, 0] = np.maximum(grid[:, :, 0], 0)
            block_num += len(total_block[i])

        dones[-1] = 1
        return total_rearrangement, grids, blocks, actions, rewards, dones, masks, probs, block_lefts

    def Train(self, train_step, eval_step, K, pr_num, batch_num, run_step_num, simulation_day, lookahead_num, ppo,
              model_dir, ASR_1, Random_1, BLF_1):
        eval_set = []  # Evaluation dataset
        history = np.zeros((2010, 2))  # To record rearrangement results during training
        for _ in range(pr_num):
            grid, grid_save, init_blocks = self.Generate_grid(None)
            total_block = []
            for i in range(1, simulation_day + 1):
                Created_blocks = self.Create_blocks()
                total_block.append(Created_blocks)
            block_concat = np.concatenate(total_block, axis=0)
            total_block_encoded = self.block_encoding(block_concat, self.TP_type)
            eval_set.append([grid.copy(), total_block.copy(), total_block_encoded.copy()])

        '''
        for ev_set in eval_set:
            for _____ in range(batch_num):
                total_rearrangement, grids, blocks, actions, rewards, dones, masks, probs, block_lefts = self.Run_simulation(
                    simulation_day, lookahead_num, ASR_1, ev_set[0].copy(), ev_set[1].copy(), ev_set[2].copy())
                ave_rearrangement += total_rearrangement
        print('ASR ', ave_rearrangement / pr_num / batch_num)
        ave_rearrangement = 0
        for ev_set in eval_set:
            for _____ in range(batch_num):
                total_rearrangement, grids, blocks, actions, rewards, dones, masks, probs, block_lefts = self.Run_simulation(
                    simulation_day, lookahead_num, Random_1, ev_set[0].copy(), ev_set[1].copy(), ev_set[2].copy())
                ave_rearrangement += total_rearrangement
        print('Random ', ave_rearrangement / pr_num / batch_num)
        ave_rearrangement = 0
        for ev_set in eval_set:
            for _____ in range(batch_num):
                total_rearrangement, grids, blocks, actions, rewards, dones, masks, probs, block_lefts = self.Run_simulation(
                    simulation_day, lookahead_num, BLF_1, ev_set[0].copy(), ev_set[1].copy(), ev_set[2].copy())
                ave_rearrangement += total_rearrangement
        print('BLF', ave_rearrangement / pr_num / batch_num)
        '''
        # Generate evaluation set for validation

        # Start training loop
        for tr_step in range(train_step):
            problem_set = []
            history = np.zeros((2001, 2))
            # Generate training problem set
            for _ in range(pr_num):
                grid, grid_save, init_blocks = self.Generate_grid(None)
                total_block = []
                for i in range(1, simulation_day + 1):
                    Created_blocks = self.Create_blocks()
                    total_block.append(Created_blocks)
                block_concat = np.concatenate(total_block, axis=0)
                total_block_encoded = self.block_encoding(block_concat, self.TP_type)
                problem_set.append([grid.copy(), total_block.copy(), total_block_encoded.copy()])

            # For each run step, collect data and train
            for run_step in range(run_step_num):
                ave_rearrangement = 0
                gridss, blockss, actionss = [], [], []
                rewardss, doness, maskss = [], [], []
                probss, block_leftss = [], []

                for pr_set in problem_set:
                    for ___ in range(batch_num):
                        # Run simulation and collect data
                        total_rearrangement, grids, blocks, actions, rewards, dones, masks, probs, block_lefts = self.Run_simulation(
                            simulation_day, lookahead_num, ppo, pr_set[0].copy(), pr_set[1].copy(), pr_set[2].copy())
                        gridss.append(grids)
                        blockss.append(blocks)
                        actionss.append(actions)
                        rewardss.append(rewards[1:])  # Skip first reward (padding)
                        doness.append(dones)
                        maskss.append(masks)
                        probss.append(probs)
                        block_leftss.append(block_lefts)
                        ave_rearrangement += total_rearrangement

                # Normalize
                ave_rearrangement /= (pr_num * batch_num)

                # Collect episode lengths
                ep_len = [len(gr) for gr in gridss]

                # Concatenate all collected data
                gridss = np.concatenate(gridss, axis=0)
                blockss = np.concatenate(blockss, axis=0)
                actionss = np.concatenate(actionss, axis=0)
                rewardss = np.concatenate(rewardss, axis=0)
                doness = np.concatenate(doness, axis=0)
                maskss = np.concatenate(maskss, axis=0)
                probss = np.concatenate(probss, axis=0)
                block_leftss = np.concatenate(block_leftss, axis=0)

                # Train K epochs on collected batch
                for ____ in range(K):
                    ave_loss, v_loss, p_loss = ppo.update(
                        gridss[:, :, :, :-1], blockss, block_leftss, actionss, rewardss,
                        doness, maskss, probss, ep_len, run_step, tr_step, model_dir)

                print(tr_step, ave_rearrangement)
                history[int(tr_step * run_step_num + run_step), 0] = ave_rearrangement

                # Evaluation every 20 steps
                if run_step % eval_step == 0:
                    ave_rearrangement = 0
                    for ev_set in eval_set:
                        for _____ in range(batch_num):
                            total_rearrangement, *_ = self.Run_simulation(
                                simulation_day, lookahead_num, ppo, ev_set[0].copy(), ev_set[1].copy(),
                                ev_set[2].copy())
                            ave_rearrangement += total_rearrangement
                    print('eval', tr_step, ave_rearrangement / (pr_num * batch_num))

        return history

if __name__ == "__main__":
    # Directory setup
    problem_dir = '/output/problem_set/'
    if not os.path.exists(problem_dir): os.makedirs(problem_dir)

    model_dir = '/output/MLP_C_new/'
    if not os.path.exists(model_dir): os.makedirs(model_dir)

    history_dir = '/output/history/'
    if not os.path.exists(history_dir): os.makedirs(history_dir)

    input_dir = '/input/'
    if not os.path.exists(input_dir): os.makedirs(input_dir)

    # Environment and seeds
    device = 'cuda'
    pr_size = (7, 7)
    init_block = 10
    bpd = (10, 14)
    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Instantiate simulation and policies
    ST_sim = Stockyard_simulation(
        yard_size=pr_size, initial_block=init_block, lam=1 / 250,
        weight=(1, 501), TP_type=[300, 400, 550], Block_per_Day=bpd, mod=0)

    ASR_1 = Heuristic(grid_size=pr_size, TP_type_len=3, mod='ASR')
    Random_1 = Heuristic(grid_size=pr_size, TP_type_len=3, mod='Random')
    BLF_1 = Heuristic(grid_size=pr_size, TP_type_len=3, mod='BLF')

    ppo = PPO(
        feature_dim=4, hidden_dim=32, lookahead_block_num=1, grid_size=pr_size,
        learning_rate=0.001, lmbda=0.95, gamma=1, alpha=0.5, beta=0.01,
        epsilon=0.2, mod='GAT'
    ).to(device)

    # Run training
    history = ST_sim.Train(
        train_step=10, eval_step=20, K=2, pr_num=4, batch_num=8, run_step_num=200,
        simulation_day=10, lookahead_num=1, ppo=ppo, model_dir=model_dir,
        ASR_1=ASR_1, Random_1=Random_1, BLF_1=BLF_1)

    # Save history
    history = pd.DataFrame(history)
    history.to_excel('history_GAT_new.xlsx', sheet_name='Sheet', index=False)
