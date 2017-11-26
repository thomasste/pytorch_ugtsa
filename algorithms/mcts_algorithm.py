from algorithms.algorithm import Algorithm
from random import randint
from recordclass import recordclass

import numpy as np


class MCTSAlgorithm(Algorithm):
    Node = recordclass('Node', 'number_of_visits children statistic')

    def __init__(self, game_state, grow_factor):
        super().__init__(game_state)
        self.grow_factor = grow_factor
        self.tree = []

    def statistic(self):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError

    def modified_statistic(self, statistic, update):
        raise NotImplementedError

    def modified_update(self, update, statistic):
        raise NotImplementedError

    def move_rate(self, parent_statistic, child_statistic):
        raise NotImplementedError

    def improve(self):
        if not self.tree:
            self.tree.append(MCTSAlgorithm.Node(
                number_of_visits=0,
                children=(1, self.game_state.move_count() + 1),
                statistic=self.statistic()))
            for i in range(self.game_state.move_count()):
                self.game_state.apply_move(i)
                self.tree.append(MCTSAlgorithm.Node(
                    number_of_visits=0,
                    children=None,
                    statistic=self.statistic()))
                self.game_state.undo_move()
        else:
            stack = [0]
            while self.tree[stack[-1]].children:
                node = self.tree[stack[-1]]
                node.number_of_visits += 1

                if self.game_state.player == -1:
                    move = randint(0, node.children[1] - node.children[0] - 1)
                else:
                    parent_statistic = node.statistic
                    children_statistics = [
                        self.tree[c].statistic
                        for c in range(node.children[0], node.children[1])]
                    move = np.argmax([
                         self.move_rate_as_numpy(self.move_rate(parent_statistic, s))[self.game_state.player]
                         for s in children_statistics])

                stack.append(node.children[0] + move)
                self.game_state.apply_move(move)

            leaf = self.tree[stack[-1]]
            leaf.number_of_visits += 1

            if not self.game_state.is_final() and leaf.number_of_visits == self.grow_factor:
                move_count = self.game_state.move_count()
                leaf.children = (len(self.tree), len(self.tree) + move_count)
                for m in range(move_count):
                    self.game_state.apply_move(m)
                    self.tree.append(MCTSAlgorithm.Node(
                        number_of_visits=0,
                        children=None,
                        statistic=self.statistic()))
                    self.game_state.undo_move()

            update = self.update()

            while stack:
                node = self.tree[stack[-1]]
                statistic = node.statistic
                node.statistic = self.modified_statistic(statistic, update)
                stack.pop()

                if stack:
                    update = self.modified_update(update, statistic)
                    self.game_state.undo_move()

    def move_rates(self):
        root = self.tree[0]
        parent_statistic = root.statistic
        return [self.move_rate(parent_statistic, self.tree[c].statistic)
                for c in range(root.children[0], root.children[1])]

    def move_rate_as_numpy(self, move_rate):
        raise NotImplementedError
