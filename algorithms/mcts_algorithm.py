from algorithms.algorithm import Algorithm
from random import randint, choice
from recordclass import recordclass
from threading import Lock

import numpy as np


class MCTSAlgorithm(Algorithm):
    Node = recordclass('Node', 'number_of_visits children statistic lock')

    def __init__(self, game_state, grow_factor):
        super().__init__(game_state)
        self.grow_factor = grow_factor
        self.root = None

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

    def empty_tree(self):
        root = MCTSAlgorithm.Node(
            number_of_visits=0,
            children=[],
            statistic=self.statistic(),
            lock=Lock())
        for i in range(self.game_state.move_count()):
            self.game_state.apply_move(i)
            root.children.append(MCTSAlgorithm.Node(
                number_of_visits=0,
                children=None,
                statistic=self.statistic(),
                lock=Lock()))
            self.game_state.undo_move()
        return root

    def improve(self):
        stack = [self.root]

        while True:
            node = stack[-1]
            with node.lock:
                if not node.children:
                    break
                else:
                    node.number_of_visits += 1

                    if self.game_state.player == -1:
                        move = randint(0, len(node.children) - 1)
                    else:
                        move_rates = [
                             self.move_rate_as_numpy(self.move_rate(node.statistic, child.statistic))[self.game_state.player]
                             for child in node.children]

                        move = choice(sorted([(mr, i) for i, mr in enumerate(move_rates)])[-2:])[1]

                    stack.append(node.children[move])
                    self.game_state.apply_move(move)

        leaf = stack[-1]

        with leaf.lock:
            leaf.number_of_visits += 1

            if not self.game_state.is_final() and leaf.number_of_visits == self.grow_factor:
                leaf.children = []
                for m in range(self.game_state.move_count()):
                    self.game_state.apply_move(m)
                    leaf.children.append(MCTSAlgorithm.Node(
                        number_of_visits=0,
                        children=None,
                        statistic=self.statistic(),
                        lock=Lock()))
                    self.game_state.undo_move()

        update = self.update()

        while stack:
            node = stack[-1]
            with node.lock:
                statistic = node.statistic
                node.statistic = self.modified_statistic(statistic, update)
                stack.pop()

                if stack:
                    update = self.modified_update(update, statistic)
                    self.game_state.undo_move()

    def move_rates(self):
        return [self.move_rate(self.root.statistic, child.statistic)
                for child in self.root.children]

    def move_rate_as_numpy(self, move_rate):
        raise NotImplementedError
