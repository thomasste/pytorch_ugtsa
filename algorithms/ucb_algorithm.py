from recordclass import recordclass
from algorithms.mcts_algorithm import MCTSAlgorithm
import numpy as np


class UCBAlgorithm(MCTSAlgorithm):
    Statistic = recordclass('Statistic', 'number_of_visits wins')

    def __init__(self, game_state, grow_factor, exploration_factor):
        super().__init__(game_state, grow_factor)
        self.exploration_factor = exploration_factor

    def statistic(self):
        return UCBAlgorithm.Statistic(
            number_of_visits=0,
            wins=np.zeros(self.game_state.player_count, np.float32))

    def update(self):
        return self.game_state.light_playout_payoff()

    def modified_statistic(self, statistic, update):
        ws = np.copy(statistic.wins)
        ws[np.argmax(update)] += 1
        return UCBAlgorithm.Statistic(
            number_of_visits=statistic.number_of_visits + 1,
            wins=ws)

    def modified_update(self, update, statistic):
        return update

    def move_rate(self, parent_statistic, child_statistic):
        return (child_statistic.wins / (child_statistic.number_of_visits + 1)) + \
            self.exploration_factor * \
            np.sqrt(np.log(parent_statistic.number_of_visits + 1) / (child_statistic.number_of_visits + 1))

    def move_rate_as_numpy(self, move_rate):
        return move_rate
