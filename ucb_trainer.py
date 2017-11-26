from algorithms.ucb_algorithm import UCBAlgorithm
from algorithms.vertical_lstm_ugts_algorithm import VerticalLSTMUGTSAlgorithm
from copy import copy, deepcopy
from games.omringa import OmringaGameState
from threading import Thread
from multiprocessing import cpu_count
import numpy as np
import torch

torch.set_num_threads(1)

game_state = OmringaGameState()

ucb_algorithm = UCBAlgorithm(game_state, 5, np.sqrt(2))
ucb_algorithm.root = ucb_algorithm.empty_tree()

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
ugts_algorithm.root = ugts_algorithm.empty_tree()

def improve_ugts(iterations):
    ugts_algorithm_copy = copy(ugts_algorithm)
    ugts_algorithm_copy.game_state = deepcopy(game_state)
    for i in range(iterations):
        print(i)
        ugts_algorithm_copy.improve()

# improve_ugts(3000)

threads = [Thread(target=improve_ugts, args=(int(3000 / (cpu_count() / 2)),)) for _ in range(int(cpu_count() / 2))]

for t in threads:
    t.start()

for t in threads:
    t.join()
