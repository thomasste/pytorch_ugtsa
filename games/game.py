from algorithms.ucb_algorithm import UCBAlgorithm
from random import randint
import numpy as np


class GameState:
    def __init__(self, player_count, board_shape, info_size, player):
        self.player_count = player_count
        self.board_shape = board_shape
        self.info_size = info_size
        self.player = player

    def light_playout_payoff(self):
        counter = 0

        while not self.is_final():
            move = randint(0, self.move_count() - 1)
            self.apply_move(move)
            counter += 1

        payoff = self.payoff()

        for _ in range(counter):
            self.undo_move()

        return payoff

    def move_to_random_state(self):
        counter = 0

        while not self.is_final():
            if self.player == -1:
                move = randint(0, self.move_count() - 1)
            else:
                ucb_algorithm = UCBAlgorithm(self, 5, np.sqrt(2))
                for _ in range(10000):
                    ucb_algorithm.improve()
                move = ucb_algorithm.best_move()

            self.apply_move(move)
            counter += 1

        undo_times = randint(0, counter)

        for _ in range(undo_times):
            self.undo_move()

    def move_count(self):
        raise NotImplementedError

    def apply_move(self, move):
        raise NotImplementedError

    def undo_move(self):
        raise NotImplementedError

    def is_final(self):
        return self.move_count() == 0

    def board(self):
        raise NotImplementedError

    def info(self):
        raise NotImplementedError

    def payoff(self):
        raise NotImplementedError
