
from sklearn.neural_network import MLPRegressor
from core.board import Board

import numpy as np

# statify this too...
import random

from collections import defaultdict

random_state = np.random.RandomState(42)

class NoMind:
    def __init__(self, size, epsilon, alpha):
        # features = grid + identifier for whose turn it is
        # constant learning rate
        self.q_memory = defaultdict(float)
        self.alpha = alpha
        self.epsilon = epsilon

    # with epsilon probability will select random move
    # returns whether game has concluded or not
    def make_move(self, board, as_player, verbose=True):
        decision_x = None
        decision_y = None
        current_q = self.q(board, as_player)
        assert(as_player == board.player_to_move)
        # find optimal decision
        next_actions = []
        next_qs = []
        # check each move
        for x, y in board.available_moves:
            board.hypothetical_move(x, y)
            next_actions.append((x, y))
            next_qs.append(self.q(board, as_player))
            board.unmove(x, y)

        best_q = np.max(next_qs)
        options = []
        for i in range(len(next_qs)):
            if next_qs[i] == best_q:
                options.append(i)
        random_state.shuffle(options)
        decision_x, decision_y = next_actions[options[0]]

        # occasionally try second best move?
        if random_state.random_sample() < self.epsilon:
            print('random move')
            decision_x, decision_y = random.sample(board.available_moves, 1)[0]

        print(next_qs)

        new_q = (1 - self.alpha) * current_q + self.alpha * np.max(next_qs)
        if abs(self.q_memory[self.feature_vector(board, as_player)]) > 1e-7:
            print('seen before', new_q, current_q)
        self.q_memory[self.feature_vector(board, as_player)] = new_q
        self.q_memory[self.feature_vector(board, -as_player)] = -new_q

        self.q_memory[self.feature_vector(board, as_player)] = max(min(self.q_memory[self.feature_vector(board, as_player)], 100), -100)
        self.q_memory[self.feature_vector(board, -as_player)] = max(min(self.q_memory[self.feature_vector(board, -as_player)], 100), -100)

        result = board.make_move(decision_x, decision_y)
        # game over
        reward = 0
        if result is True:
            reward = 1
            self.q_memory[self.feature_vector(board, as_player)] = 100
            self.q_memory[self.feature_vector(board, -as_player)] = -100

        if result is not None:
            return True

        return False

    def feature_vector(self, board, as_player):
        is_my_move = 1 if board.player_to_move == as_player else -1
        return tuple(np.hstack([np.reshape(board.get_matrix(as_player), (-1)), [is_my_move]]))

    # turn = 1 if my turn, -1 if opponent's
    def q(self, board, as_player):
        return self.q_memory[self.feature_vector(board, as_player)]