from algorithms.mcts_algorithm import MCTSAlgorithm
from torch.autograd import Variable
import torch


class UGTSAlgorithm(MCTSAlgorithm):
    def __init__(self, game_state, grow_factor, pt_statistic, pt_update,
                 pt_modified_statistic, pt_modified_update, pt_move_rate):
        super().__init__(game_state, grow_factor)
        self.pt_statistic = pt_statistic
        self.pt_update = pt_update
        self.pt_modified_statistic = pt_modified_statistic
        self.pt_modified_update = pt_modified_update
        self.pt_move_rate = pt_move_rate

    def statistic(self):
        return self.pt_statistic(
            Variable(torch.from_numpy(self.game_state.board()).unsqueeze(0), requires_grad=False),
            Variable(torch.from_numpy(self.game_state.info()).unsqueeze(0), requires_grad=False))

    def update(self):
        return self.pt_update(
            Variable(torch.from_numpy(self.game_state.light_playout_payoff()).unsqueeze(0), requires_grad=False))

    def modified_statistic(self, statistic, update):
        return self.pt_modified_statistic(statistic, update)

    def modified_update(self, update, statistic):
        return self.pt_modified_update(update, statistic)

    def move_rate(self, parent_statistic, child_statistic):
        return self.pt_move_rate(parent_statistic, child_statistic)

    def move_rate_as_numpy(self, move_rate):
        return move_rate.data.numpy()[0]
