from games.game import GameState
from enum import Enum
from recordclass import recordclass
import numpy as np


class OmringaGameState(GameState):
    class State(Enum):
        BET = 0
        NATURE = 1
        PLACE = 2

    Change = recordclass('Change', 'player state bets passes chosen_player pass_ position index')

    def __init__(self):
        super().__init__(2, (7, 7), 3, 0)
        self.min_bet = 0
        self.max_bet = 9
        self.group_penalty = -5
        self.state = OmringaGameState.State.BET
        self.bets = (None, None)
        self.passes = 0
        self.chosen_player = None
        self.board_ = np.zeros(self.board_shape, np.float32)
        self.empty_positions = [
            (y, x)
            for y in range(self.board_shape[0])
            for x in range(self.board_shape[1])]
        self.changes = []

    def move_count(self):
        if self.state == OmringaGameState.State.BET:
            return self.max_bet - self.min_bet
        elif self.state == OmringaGameState.State.NATURE:
            return 2
        else:
            if self.passes == 2:
                return 0
            else:
                return len(self.empty_positions) + 1

    def apply_move(self, move):
        change = OmringaGameState.Change(
            player=self.player,
            state=self.state,
            bets=self.bets,
            passes=self.passes,
            chosen_player=self.chosen_player,
            pass_=None,
            position=None,
            index=None)
        self.changes.append(change)
        if self.state == OmringaGameState.State.BET:
            self.bets = (
                move if self.player == 0 else self.bets[0],
                move if self.player == 1 else self.bets[1])
            if self.bets[self.player ^ 1] is None:
                self.state = OmringaGameState.State.BET
                self.player ^= 1
            else:
                if self.bets[0] == self.bets[1]:
                    self.state = OmringaGameState.State.NATURE
                    self.player = -1
                else:
                    self.state = OmringaGameState.State.PLACE
                    self.player = 1 if self.bets[0] < self.bets[1] else 0
        elif self.state == OmringaGameState.State.NATURE:
            self.state = OmringaGameState.State.PLACE
            self.player = move
            self.chosen_player = move
        else:
            if move == len(self.empty_positions):
                change.pass_ = True
                self.state = OmringaGameState.State.NATURE
                self.player ^= 1
                self.passes += 1
            else:
                change.pass_ = False
                change.position = self.empty_positions[move]
                change.index = move

                self.board_[change.position[0], change.position[1]] = self.player + 1

                self.empty_positions[move], self.empty_positions[-1] = \
                    self.empty_positions[-1], self.empty_positions[move]
                self.empty_positions.pop()

    def undo_move(self):
        change = self.changes.pop()
        self.player = change.player
        self.state = change.state
        self.bets = change.bets
        self.passes = change.passes
        self.chosen_player = change.chosen_player

        if self.state == OmringaGameState.State.PLACE:
            if not change.pass_:
                self.empty_positions.append(change.position)
                self.empty_positions[change.index], self.empty_positions[-1] = \
                    self.empty_positions[-1], self.empty_positions[change.index]

                self.board_[change.position[0], change.position[1]] = 0

    def board(self):
        return np.copy(self.board_)

    def info(self):
        return np.array([
            self.bets[0] if self.bets[0] is not None else -1,
            self.bets[1] if self.bets[1] is not None else -1,
            self.chosen_player if self.chosen_player is not None else -1], np.float32)

    def group_count(self, id):
        result = 0

        dys = [-1, 0, 0, 1]
        dxs = [0, -1, 1, 0]
        visited = np.zeros(self.board_shape, np.bool)

        for y in range(self.board_shape[0]):
            for x in range(self.board_shape[1]):
                if not visited[y][x] and self.board_[y][x] == id:
                    visited[y][x] = True
                    result += 1

                    stack = [(y, x)]
                    while stack:
                        p = stack.pop()

                        for dx, dy in zip(dys, dxs):
                            dp = (p[0] + dy, p[1] + dx)
                            if 0 <= dp[0] < self.board_shape[0] and \
                                    0 <= dp[1] < self.board_shape[1] and \
                                    not visited[dp] and self.board_[dp] == id:
                                visited[dp] = True
                                stack.append(dp)

        return result

    def payoff(self):
        result = np.array([
            self.group_count(1) * self.group_penalty + (self.board_ == 1).sum(),
            self.group_count(2) * self.group_penalty + (self.board_ == 2).sum()], np.float32)

        if self.bets[0] < self.bets[1]:
            result[0] += self.bets[0] + 0.5
        elif self.bets[0] > self.bets[1]:
            result[1] += self.bets[1] + 0.5
        else:
            result[self.chosen_player ^ 1] += self.bets[self.chosen_player ^ 1] + 0.5

        return result
