import numpy as np
from collections import deque
device='cuda'
import torch

# Backward Dynamic Programming for path value propagation
def backtracking_dp(grid, goal):
    n, m = len(grid), len(grid[0])
    dp = np.full((n, m), -float('inf'))  # Initialize DP table with -inf
    dp[goal[0]][goal[1]] = 1  # Set goal cell value
    movements = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right

    # Check if the position is within bounds and not blocked
    def is_valid(x, y):
        return 0 <= x < n and 0 <= y < m and grid[x][y] != -2

    queue = deque([goal])  # Start BFS from the goal
    while queue:
        x, y = queue.popleft()
        for dx, dy in movements:
            nx, ny = x + dx, y + dy
            if is_valid(nx, ny):
                step_cost = -0.1 if grid[nx][ny] == -1 else -0.001  # Higher cost for obstacle cells
                new_cost = dp[x][y] + step_cost
                if new_cost > dp[nx][ny]:  # Update if better path found
                    dp[nx][ny] = new_cost
                    queue.append((nx, ny))  # Continue BFS

    # If all values in first row are still -inf, path is impossible
    if np.all(dp[0] == -float('inf')):
        return False, dp
    return True, dp



# Label all connected free space regions starting from the top row
def label_connected_paths(input_grid):
    grid = input_grid.copy()
    rows, cols = len(grid), len(grid[0])

    # Temporarily add an extra top row for entry points
    temp_grid = np.zeros((rows + 1, cols))
    temp_grid[1:, :] = grid
    grid = temp_grid

    visited = [[False for _ in range(cols)] for _ in range(rows + 1)]
    label = 2  # Start labeling from 2
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    label_num = np.zeros(rows * cols + rows)

    # DFS to label connected zero-value regions
    def dfs(x, y, label):
        visited[x][y] = True
        grid[x][y] = label
        if x != 0:
            label_num[label] += 1  # Only count inner grid cells
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows + 1 and 0 <= ny < cols and not visited[nx][ny] and grid[nx][ny] == 0:
                dfs(nx, ny, label)

    # Start DFS from every unvisited 0 cell
    for i in range(rows + 1):
        for j in range(cols):
            if grid[i][j] == 0 and not visited[i][j]:
                dfs(i, j, label)
                label += 1

    grid = grid[1:, :]  # Remove temporary row
    return grid, label_num

# Backward DP that considers buffer areas (free space) around the path
def backtracking_dp_with_free_space(grid, goal, labeled_grid, label_num):
    n, m = len(grid), len(grid[0])
    dp = np.full((n, m), -float('inf'))
    free_space = np.full((n, m), -float('inf'))
    step_space = np.full((n, m), np.inf)

    dp[goal[0]][goal[1]] = 1
    step_space[goal[0], goal[1]] = 0
    free_space[goal[0], goal[1]] = 0

    movements = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def is_valid1(x, y):
        return 0 <= x < n and 0 <= y < m and grid[nx][ny] != -2 and dp[nx][ny] != 1

    def is_valid2(x, y):
        return 0 <= x < n and 0 <= y < m and grid[nx][ny] != -2

    matrix = [[[] for _ in range(m)] for _ in range(n)]
    queue = deque([goal])

    while queue:
        x, y = queue.popleft()
        for dx, dy in movements:
            nx, ny = x + dx, y + dy
            if is_valid1(nx, ny):
                # Find direction with highest DP value among 4 directions
                temp = np.ones(4) * -100
                e = 0
                for ndx, ndy in movements:
                    if is_valid2(nx + ndx, ny + ndy):
                        temp[e] = dp[nx + ndx, ny + ndy]
                    e += 1
                ndx, ndy = movements[np.argmax(temp)]

                step_space[nx][ny] = step_space[nx + ndx][ny + ndy] + 1
                free_space[nx][ny] = free_space[nx + ndx][ny + ndy]

                if grid[nx][ny] == 0:
                    matrix[nx][ny] = matrix[nx + ndx][ny + ndy].copy()
                    if labeled_grid[nx, ny] not in matrix[nx + ndx][ny + ndy]:
                        free_space[nx, ny] += label_num[int(labeled_grid[nx, ny])]
                        matrix[nx][ny].append(labeled_grid[nx, ny].copy())
                    new_cost = dp[nx + ndx][ny + ndy] - 0.001
                if grid[nx][ny] == -1:
                    if step_space[nx, ny] > free_space[nx, ny]:
                        new_cost = dp[nx + ndx][ny + ndy] - 0.2
                    else:
                        new_cost = dp[nx + ndx][ny + ndy] - 0.1
                if new_cost > dp[nx, ny]:
                    dp[nx, ny] = new_cost
                    queue.append((nx, ny))

    if np.all(dp[0] == -float('inf')):
        return False, dp, 0, 0
    return True, dp, free_space, step_space



def search_path(result, grid):
    start = [0, np.argmax(result[0, :])]
    
    rearrange_count = 9 - int(result[start[0], start[1]] * 10)
    if result[start[0], start[1]] * 100 % 10 == 0:
        rearrange_count += 1
    current = start
    path = [current.copy()]
    movements = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while result[current[0], current[1]] != 1:
        temp = np.ones(4)*-100
        step = 0

        def is_valid(x, y):
            return 0 <= x < result.shape[0] and 0 <= y < result.shape[1]

        for dx, dy in (movements):
            nx, ny = current[0] + dx, current[1] + dy
            if is_valid(nx, ny):
                temp[step] = result[nx, ny]
            step += 1
        dx, dy = movements[np.argmax(temp)]
        current[0] = current[0] + dx
        current[1] = current[1] + dy
        if result[current[0], current[1]] != 1:
            path.append(current.copy())

    return path, rearrange_count

# Calculate the amount of free area and obstacle area required along the retrieval path
def cal_area(path, labeled_grid, label_num, input_dp, grid):
    dp = input_dp.copy()
    area_required = 0
    area_able = 0
    path_label = []
    added_label = []
    movements = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    area_left = 0
    before_label = 1

    for i in path:
        if grid[i[0], i[1]] == -1:
            area_required += 2  # Obstacle requires 2 area
            area_able += 1
            for dx, dy in movements:
                if 0 <= i[0] + dx < labeled_grid.shape[0] and 0 <= i[1] + dy < labeled_grid.shape[1]:
                    if [i[0] + dx, i[1] + dy] not in path and labeled_grid[i[0] + dx, i[1] + dy] > 1:
                        if labeled_grid[i[0] + dx, i[1] + dy] not in added_label:
                            added_label.append(labeled_grid[i[0] + dx, i[1] + dy])
                            area_able += label_num[int(labeled_grid[i[0] + dx, i[1] + dy])]

        elif grid[i[0], i[1]] == 0:
            area_required += 1
            if labeled_grid[i[0], i[1]] not in path_label:
                before_label = labeled_grid[i[0], i[1]]
                path_label.append(labeled_grid[i[0], i[1]])
                area_able += label_num[int(labeled_grid[i[0], i[1]])]

    area_left = area_able - area_required
    return area_able, area_required, area_left, dp, path_label, added_label



# Main interface to find path with sufficient free space and estimate rearrangement
def path_finder(grid, goal):
    labeled_grid, label_num = label_connected_paths(grid.copy())
    ispossible, dp, _, _ = backtracking_dp_with_free_space(grid.copy(), goal, labeled_grid, label_num)
    if not ispossible:
        return False, 0, 0, 0, 0, 0, 0, 0

    path, rearrange_count = search_path(dp, grid)
    area_able, area_required, area_left, new_dp, path_label, added_label = cal_area(path, labeled_grid, label_num, dp, grid.copy())
    return True, path, area_left, rearrange_count, path_label, added_label, labeled_grid, label_num

# BFS from a cell to get connected area including obstacles and usable free space
def bfs_area(final_grid, grid, start, path):
    rows, cols = len(grid), len(grid[0])
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    queue = deque([tuple(start)])
    vs = set([tuple(start)])
    area_size = 0

    obstacle = []
    free_space = []

    while queue:
        r, c = queue.popleft()
        area_size += 1
        if grid[r][c] == -1:
            obstacle.append([r, c])
        if grid[r][c] == 0 and [r, c] not in path:
            free_space.append([r, c])

        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in vs and final_grid[nr][nc] == 1:
                vs.add((nr, nc))
                queue.append((nr, nc))

    return obstacle, free_space



# Create binary mask of cells occupied by blocks
def Create_locate_mask(grid, TP_type_len):
    r, c, f = grid.shape
    mask = (grid[:, :, 1:1 + TP_type_len].sum(axis=2) > 0).astype(np.uint8)
    mask = mask[:, :, np.newaxis].copy()
    mask = mask.reshape(r, c)

    rows, cols = len(mask), len(mask[0])
    visited = [[False] * cols for _ in range(rows)]
    new_grid = [[1] * cols for _ in range(rows)]

    queue = deque()

    # Start BFS from top row zeros
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

    return np.array(new_grid)
# Check if there is a valid path (consisting of free spaces only) from start to goal

def bfs_path_exists(grid, start, goal):
    start = start.copy()
    goal = goal.copy()

    # Adjust rows for temporary buffer row in label process
    start[0] += 1
    goal[0] += 1
    rows, cols = grid.shape

    # Add a buffer row at the top (to simulate entrance rows)
    temp_grid = np.zeros((rows + 1, cols))
    temp_grid[1:, :] = grid
    grid = temp_grid.copy()

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    queue = deque([tuple(start)])
    vs = set([tuple(start)])

    while queue:
        r, c = queue.popleft()
        if r == goal[0] and c == goal[1]:
            return True  # Found a valid path to goal

        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in vs and grid[nr][nc] == 0:
                vs.add((nr, nc))
                queue.append((nr, nc))

    return False  # No valid path found

# Create binary mask of shape (r, c, 1) where occupied cells are 1 and empty are 0
def Create_mask(grid, TP_type_len):
    # Sum feature values corresponding to block presence (1:1+TP_type_len)
    mask = (grid[:, :, 1:1+TP_type_len].sum(axis=2) > 0).astype(np.uint8)
    return mask[:, :, np.newaxis]



# Rearrange all obstacles in the retrieval path and relocate them using policy
def retrieval(final_grid, input_grid, state_grid, target_block, path, ppo,
              step, grids, blocks, block_lefts, block_left_num,
              rewards, actions, dones, masks, probs, lookahead_num,
              TP_type_len, rt_mod):
    stock = []  # Temporarily removed blocks
    obstacles_set = []

    # Identify obstacle positions along the path
    for st in path:
        if input_grid[st[0], st[1]] == -1:
            obstacles_set.append([st[0], st[1]])
    rearrange_num = len(obstacles_set)
    goal = target_block

    # Relocate each obstacle one-by-one
    for ob_num in range(rearrange_num):
        obstacles, free_spaces = bfs_area(final_grid, input_grid, goal, path)
        target_obstacle = obstacles_set[ob_num]
        possible_space = []

        # Check if each free space is reachable from the obstacle
        for free_space in free_spaces:
            if bfs_path_exists(input_grid, target_obstacle, free_space):
                possible_space.append(free_space)

        if len(possible_space) == 0:
            # If no space found, temporarily remove block from grid (stock it)
            stock.append(state_grid[target_obstacle[0], target_obstacle[1], :-1].copy())
            input_grid[target_obstacle[0]][target_obstacle[1]] = 0
            change_reward = state_grid[target_obstacle[0], target_obstacle[1], -1]
            rewards[change_reward] -= 0.2
            state_grid[target_obstacle[0], target_obstacle[1]] = 0
        else:
            # Evaluate candidate free spaces to minimize future blockage
            can = []
            for free_space in possible_space:
                test_grid1 = final_grid.copy()
                test_grid1[free_space[0]][free_space[1]] = 0
                test_grid1[target_obstacle[0]][target_obstacle[1]] = 1

                test_grid2 = input_grid.copy()
                test_grid2[target_obstacle[0]][target_obstacle[1]] = 0
                test_grid2[free_space[0]][free_space[1]] = -1

                temp_obstacle, temp_free_space = bfs_area(test_grid1, test_grid2, goal, path)
                can.append(min(len(temp_free_space) - len(temp_obstacle), 0))

            # Choose best candidates for PPO-based relocation
            max_value = max(can)
            max_indices = [i for i, val in enumerate(can) if val == max_value]
            mask = np.ones((input_grid.shape[0], input_grid.shape[1]))
            for ind in max_indices:
                mask[possible_space[ind][0], possible_space[ind][1]] = 0

            # Build block input vector for PPO
            blocks_vec = np.zeros((lookahead_num, TP_type_len+1))
            blocks_vec[:, 0] = 250
            blocks_vec[:, 1:1 + int((1 + TP_type_len) / 2.0)] = 1
            blocks_vec[0] = state_grid[target_obstacle[0], target_obstacle[1], :-1]

            # Log rollout for PPO
            grids.append(state_grid.copy())
            blocks.append(blocks_vec.copy())
            block_lefts.append(block_left_num)
            masks.append(mask.reshape(-1, 1).copy())

            # Convert to tensors and normalize
            mask_tensor = torch.tensor(mask.reshape(1, -1, 1), dtype=torch.float32).to(device)
            state_tensor = torch.tensor(state_grid[:, :, :-1].reshape(1, *state_grid.shape[:2], -1), dtype=torch.float32).to(device)
            blocks_tensor = torch.tensor(blocks_vec.reshape(1, lookahead_num, -1), dtype=torch.float32).to(device)

            state_tensor[:, :, :, 0] /= 500.0
            blocks_tensor[:, :, 0] /= 500.0

            # Use PPO to select target location
            pr, target_space = ppo.Locate(state_tensor, blocks_tensor, mask_tensor, ans=None)
            target_r = target_space.item() // input_grid.shape[0]
            target_c = target_space.item() % input_grid.shape[1]

            # Log PPO results
            probs.append(pr.item())
            actions.append(target_space.item())
            dones.append(0)
            rewards.append(0)

            step += 1
            change_reward = state_grid[target_obstacle[0], target_obstacle[1], -1]
            rewards[change_reward] -= 0.1

            # Relocate obstacle to new location
            state_grid[target_r, target_c] = state_grid[target_obstacle[0], target_obstacle[1]].copy()
            state_grid[target_r, target_c, -1] = step
            state_grid[target_obstacle[0], target_obstacle[1]] = 0

            final_grid[target_r][target_c] = 0
            input_grid[target_r][target_c] = -1
            input_grid[target_obstacle[0]][target_obstacle[1]] = 0

    # Mark goal block as retrieved (if required)
    if rt_mod == 'OR':
        state_grid[goal[0], goal[1], :] = 0
        final_grid[goal[0], goal[0]] = 1
        input_grid[goal[0], goal[1]] = 0

    # Reinsert temporarily removed blocks (from stock) using PPO
    for e, ob in enumerate(stock):
        can = []
        mask = Create_locate_mask(state_grid, TP_type_len)
        zero_positions = np.argwhere(mask == 0)
        obstacle_left = len(stock) - e - 1

        for row_num, col_num in zero_positions:
            temp_grid = state_grid.copy()
            temp_grid[row_num, col_num, :-1] = ob.copy()
            mask = Create_locate_mask(temp_grid, TP_type_len)
            new_zero_positions = np.argwhere(mask == 0)
            can.append(min(len(new_zero_positions) - obstacle_left, 0))

        max_value = max(can)
        max_indices = [i for i, val in enumerate(can) if val == max_value]
        final_mask = np.ones((input_grid.shape[0], input_grid.shape[1]))
        for ind in max_indices:
            final_mask[zero_positions[ind][0], zero_positions[ind][1]] = 0

        blocks_vec = np.zeros((lookahead_num, TP_type_len+1))
        blocks_vec[:, 0] = 250
        blocks_vec[:, 1:1 + int((1 + TP_type_len) / 2.0)] = 1
        max_copy = min(len(stock) - e, lookahead_num)
        blocks_vec[:max_copy] = np.array(stock[e:e+max_copy])

        # Log and run PPO
        block_lefts.append(block_left_num)
        grids.append(state_grid.copy())
        blocks.append(blocks_vec.copy())
        masks.append(final_mask.reshape(-1, 1).copy())

        final_mask = torch.tensor(final_mask.reshape(1, -1, 1), dtype=torch.float32).to(device)
        state_tensor = torch.tensor(state_grid[:, :, :-1].reshape(1, *state_grid.shape[:2], -1), dtype=torch.float32).to(device)
        blocks_tensor = torch.tensor(blocks_vec.reshape(1, lookahead_num, -1), dtype=torch.float32).to(device)

        state_tensor[:, :, :, 0] /= 500.0
        blocks_tensor[:, :, 0] /= 500.0

        pr, target_space = ppo.Locate(state_tensor, blocks_tensor, final_mask,ans=None)
        target_r = target_space.item() // input_grid.shape[0]
        target_c = target_space.item() % input_grid.shape[1]

        probs.append(pr.item())
        actions.append(target_space.item())
        dones.append(0)
        rewards.append(0)

        step += 1
        state_grid[target_r, target_c, :-1] = ob
        state_grid[target_r, target_c, -1] = step

        final_grid[target_r][target_c] = 0
        input_grid[target_r][target_c] = -1

    return rearrange_num, state_grid, step, grids, blocks, actions, rewards, dones, masks, probs, block_lefts


def classify_grid(grid, TP_capacity, goal):  #
    M = TP_capacity
    r = grid.shape[0]
    c = grid.shape[1]
    classified_grid = np.zeros((r, c), dtype=int)

    for i in range(r):
        for j in range(c):
            # If the cell is the target block location
            if grid[i, j, 1 + M] == 1:
                classified_grid[i, j] = -1
            # If the cell contains any other block
            elif grid[i, j, 1 + M:].sum() > 0:
                classified_grid[i, j] = -2
            # If the cell is empty
            else:
                classified_grid[i, j] = 0

    # Mark the goal location (target block) as 1
    classified_grid[goal[0]][goal[1]] = 1

    return classified_grid


def Retrieval(grid, TP_capacity, target_block, ppo, step, grids, blocks, block_lefts, block_left_num,
              actions, rewards, dones, masks, probs, lookahead_num, TP_type_len, rt_mod):
    # grid shape: (r, c, 2 + TP_type)

    # Preprocess the grid to a classified format
    input_grid = classify_grid(grid, TP_capacity, target_block)

    # Use path-finding to check if the target can be retrieved and get path info
    ispossible, path, area_left, count, path_label, added_label, labeled_grid, label_num = path_finder(
        input_grid.copy(), target_block)

    if ispossible == False:
        # If retrieval is not possible, return immediately with no rearrangement
        rearrange_num = 0
        return ispossible, rearrange_num, grid, step, grids, blocks, actions, rewards, dones, masks, probs, block_lefts

    # Collect all cells in path-related regions
    total_area = []
    for label in added_label:
        total_area = list(np.array(np.where(labeled_grid == label)).T) + total_area
    for label in path_label:
        total_area = list(np.array(np.where(labeled_grid == label)).T) + total_area
    total_area = total_area + path

    # Initialize final grid indicating valid movement/retrieval area
    final_grid = np.zeros((grid.shape[0], grid.shape[1]))
    for x, y in total_area:
        final_grid[x][y] = 1  # Mark as part of retrieval area

    # Perform actual retrieval operation and record the result
    rearrange_num, end_grid, step, grids, blocks, actions, rewards, dones, masks, probs, block_lefts = retrieval(
        final_grid, input_grid, grid.copy(), target_block, path, ppo, step, grids, blocks,
        block_lefts, block_left_num, rewards, actions, dones, masks, probs,
        lookahead_num, TP_type_len, rt_mod)

    return ispossible, rearrange_num, end_grid, step, grids, blocks, actions, rewards, dones, masks, probs, block_lefts


def Count_retrieval(grid, TP_capacity, target_block):
    # Classify the grid based on block and transporter information
    input_grid = classify_grid(grid, TP_capacity, target_block)

    # Return the number of blocks to be rearranged and feasibility
    ispossible, path, area_left, count, path_label, added_label, labeled_grid, label_num = path_finder(
        input_grid.copy(), target_block)
    return count, ispossible


