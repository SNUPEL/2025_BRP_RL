import numpy as np
from Retrieval import *
from Placement import *
from Simulation import *

from Placement_heuristic import *
from collections import OrderedDict
#import vessl


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
