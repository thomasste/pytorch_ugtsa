from algorithms.ucb_algorithm import UCBAlgorithm
from algorithms.vertical_lstm_ugts_algorithm import VerticalLSTMUGTSAlgorithm
from games.omringa import OmringaGameState
from torch.autograd import Variable
from torch.nn import functional as F
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
    statistic_kernel_shapes=[(32, 2)],
    statistic_hidden_features=[100],
    update_hidden_features=[100],
    modified_statistic_layers=2,
    modified_update_layers=2,
    move_rate_hidden_features=[100])

ucb_move_rates = []
ugts_move_rates = []

for i in range(3000):
    print(i)
    for j in range(10):
        ucb_algorithm.improve()
    ucb_move_rates.extend(ucb_algorithm.move_rates())

for i in range(3000):
    print(i)
    ugts_algorithm.improve()
    ugts_move_rates.extend(ugts_algorithm.move_rates())

loss = F.binary_cross_entropy_with_logits(
    torch.cat(ugts_move_rates, 0),
    F.sigmoid(Variable(torch.from_numpy(np.array(ucb_move_rates)))))
print(loss)
loss.backward()