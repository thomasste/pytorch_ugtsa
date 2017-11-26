from algorithms.ucb_algorithm import UCBAlgorithm
from algorithms.vertical_lstm_ugts_algorithm import VerticalLSTMUGTSAlgorithm
from games.omringa import OmringaGameState
import numpy as np
import torch

torch.set_num_threads(1)

game_state = OmringaGameState()
ucb_algorithm = UCBAlgorithm(game_state, 5, np.sqrt(2))
ugts_algorithm = VerticalLSTMUGTSAlgorithm(
    game_state=game_state,
    grow_factor=5,
    statistic_size=100,
    update_size=100,
    statistic_kernel_shapes=[(16, 2), (32, 2)],
    statistic_hidden_features=[100],
    update_hidden_features=[100],
    modified_statistic_layers=2,
    modified_update_layers=2,
    move_rate_hidden_features=[100])

for i in range(10):
    print(i)
    ucb_algorithm.improve()

print(ucb_algorithm.tree[:20])

for i in range(3000):
    print(i)
    ugts_algorithm.improve()

print(ugts_algorithm.tree[:20])