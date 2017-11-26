import numpy as np


class Algorithm:
    def __init__(self, game_state):
        self.game_state = game_state

    def improve(self):
        raise NotImplementedError

    def move_rates(self):
        raise NotImplementedError

    def move_rate_as_numpy(self, move_rate):
        raise NotImplementedError

    def best_move(self):
        return np.argmax([
             self.move_rate_as_numpy(r)[self.game_state.player]
             for r in self.move_rates()])
