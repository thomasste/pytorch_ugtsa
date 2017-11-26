from algorithms.ugts_algorithm import UGTSAlgorithm
from torch import nn
from torch.nn import functional as F
import torch


class VerticalLSTMUGTSAlgorithm(UGTSAlgorithm):
    class Statistic(nn.Module):
        def __init__(self, board_shape, info_size, statistic_size, kernel_shapes, hidden_features):
            super().__init__()
            self.convs = []
            self.fcs = []
            self.dropouts = [nn.Dropout()] * (len(hidden_features) + 1)

            in_channels = 1
            for out_channels, kernel_size in kernel_shapes:
                self.convs.append(nn.Conv2d(in_channels, out_channels, kernel_size))
                in_channels = out_channels
                board_shape = (board_shape[0] - kernel_size + 1, board_shape[1] - kernel_size + 1)

            self.flat_size = board_shape[0] * board_shape[1] * in_channels

            in_features = self.flat_size + info_size
            for out_features in hidden_features + [statistic_size]:
                self.fcs.append(nn.Linear(in_features, out_features))
                in_features = out_features

        def forward(self, board, info):
            signal = board.unsqueeze(1)
            for conv in self.convs:
                signal = F.relu(conv(signal))

            signal = torch.cat([signal.view(-1, self.flat_size), info], 1)
            for fc, dropout in zip(self.fcs, self.dropouts):
                signal = dropout(F.relu(fc(signal)))

            return signal

    class Update(nn.Module):
        def __init__(self, payoff_size, update_size, hidden_features):
            super().__init__()
            self.fcs = []
            self.dropouts = [nn.Dropout()] * (len(hidden_features) + 1)

            in_features = payoff_size
            for out_features in hidden_features + [update_size]:
                self.fcs.append(nn.Linear(in_features, out_features))
                in_features = out_features

        def forward(self, payoff):
            signal = payoff
            for fc, dropout in zip(self.fcs, self.dropouts):
                signal = dropout(F.relu(fc(signal)))
            return signal

    class ModifiedStatistic(nn.Module):
        def __init__(self, statistic_size, update_size, layers):
            super().__init__()
            self.layers = layers
            self.hidden_size = int(statistic_size / layers / 2)
            self.lstm = nn.LSTM(update_size, self.hidden_size, self.layers)

        def forward(self, statistic, update):
            hs_0, cs_0 = torch.chunk(statistic, 2, 1)
            _, (hs_n, cs_n) = self.lstm(
                update,
                (hs_0.view(self.layers, 1, self.hidden_size),
                 cs_0.view(self.layers, 1, self.hidden_size)))
            return torch.cat([
                hs_n.view(1, self.layers * self.hidden_size),
                cs_n.view(1, self.layers * self.hidden_size)], 1)

    class ModifiedUpdate(nn.Module):
        def __init__(self, statistic_size, update_size, layers):
            super().__init__()
            self.layers = layers
            self.hidden_size = int(update_size / layers / 2)
            self.lstm = nn.LSTM(statistic_size, self.hidden_size, self.layers)

        def forward(self, update, statistic):
            hs_0, cs_0 = torch.chunk(update, 2, 1)
            _, (hs_n, cs_n) = self.lstm(
                statistic,
                (hs_0.view(self.layers, 1, self.hidden_size),
                 cs_0.view(self.layers, 1, self.hidden_size)))
            return torch.cat([
                hs_n.view(1, self.layers * self.hidden_size),
                cs_n.view(1, self.layers * self.hidden_size)], 1)

    class MoveRate(nn.Module):
        def __init__(self, payoff_size, statistic_size, hidden_features):
            super().__init__()
            self.fcs = []
            self.dropouts = [nn.Dropout()] * len(hidden_features)

            in_features = statistic_size * 2
            for out_features in hidden_features + [payoff_size]:
                self.fcs.append(nn.Linear(in_features, out_features))
                in_features = out_features

        def forward(self, parent_statistic, child_statistic):
            signal = torch.cat([parent_statistic, child_statistic], 1)
            for fc, dropout in zip(self.fcs[:-1], self.dropouts):
                signal = dropout(F.relu(fc(signal)))
            return self.fcs[-1](signal)

    def __init__(self, game_state, grow_factor, statistic_size, update_size, statistic_kernel_shapes,
                 statistic_hidden_features, update_hidden_features, modified_statistic_layers, modified_update_layers,
                 move_rate_hidden_features):
        super().__init__(
            game_state, grow_factor,
            VerticalLSTMUGTSAlgorithm.Statistic(
                game_state.board_shape,
                game_state.info_size,
                statistic_size,
                statistic_kernel_shapes,
                statistic_hidden_features),
            VerticalLSTMUGTSAlgorithm.Update(
                game_state.player_count,
                update_size,
                update_hidden_features),
            VerticalLSTMUGTSAlgorithm.ModifiedStatistic(
                statistic_size,
                update_size,
                modified_statistic_layers),
            VerticalLSTMUGTSAlgorithm.ModifiedUpdate(
                statistic_size,
                update_size,
                modified_update_layers),
            VerticalLSTMUGTSAlgorithm.MoveRate(
                game_state.player_count,
                statistic_size,
                move_rate_hidden_features))
