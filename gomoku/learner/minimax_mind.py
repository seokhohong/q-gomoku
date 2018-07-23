from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from core.board import Board

import numpy as np

import random

from sklearn.metrics import mean_absolute_error

from collections import defaultdict

from copy import copy
from sklearn.externals import joblib

random_state = np.random.RandomState(42)

MIN_Q = -1
MAX_Q = 1

class MinimaxMind:
    def __init__(self, size, epsilon, alpha):
        # features = grid + identifier for whose turn it is
        # constant learning rate
        #self.est = MLPRegressor(hidden_layer_sizes=(size ** 2 + 1, size ** 2), learning_rate_init=0.001, )
        #self.est = DecisionTreeRegressor('mae')
        #self.est = RandomForestRegressor(n_estimators = 5)
        self.est = GradientBoostingRegressor(loss='lad', n_estimators=5, max_depth=20)
        self.est.fit(random_state.randint(size=(1, size ** 2 + 1), low = -1, high = 2), np.zeros(1))

        # zero out the MLP
        #for i in range(len(self.est.intercepts_)):
        #    self.est.intercepts_[i] = 0
        #for x in self.est.coefs_:
        #    x.fill(0)

        #self.train_vectors = defaultdict(float)
        self.train_vectors = []
        self.train_labels = []
        self.fitted = False

        self.alpha = alpha
        self.epsilon = epsilon

    # with epsilon probability will select random move
    # returns whether game has concluded or not
    def make_move(self, board, as_player, retrain=True, verbose=True):
        decision_x = None
        decision_y = None
        current_q = self.q(board, as_player)
        assert(as_player == board.player_to_move)
        # find optimal decision
        next_actions = []
        next_qs = []
        # check each move
        assert(len(board.available_moves) <= board.size ** 2)
        next_actions, next_qs = self.minimax_q(board, as_player)

        best_q = np.max(next_qs)
        options = []
        for i in range(len(next_qs)):
            if next_qs[i] == best_q:
                options.append(i)
        random_state.shuffle(options)
        decision_x, decision_y = next_actions[options[0]]

        # occasionally try second best move?
        if random_state.random_sample() < self.epsilon:
            if verbose:
                print('random move')
            decision_x, decision_y = random.sample(board.available_moves, 1)[0]

        for i in range(len(next_qs)):
            if verbose:
                print('Move (', next_actions[i][0], ', ', next_actions[i][1], ') : ', next_qs[i])

        new_q = (1 - self.alpha) * current_q + self.alpha * np.max(next_qs)
        self.add_train_example(board, as_player, new_q)

        result = board.make_move(decision_x, decision_y)

        if result is not None:
            if retrain:
                self.update_model()
            return True

        return False

    # adds rotations
    def add_train_example(self, board, as_player, result):
        is_my_move = 1 if board.player_to_move == as_player else -1
        vectors = [
            np.hstack([np.reshape(board.get_matrix(as_player), (-1)), [is_my_move]]),
            np.hstack([np.reshape(board.get_matrix(as_player).transpose(), (-1)), [is_my_move]]),
            np.hstack([np.reshape(np.rot90(board.get_matrix(as_player)), (-1)), [is_my_move]]),
            np.hstack([np.reshape(np.rot90(board.get_matrix(as_player).transpose()), (-1)), [is_my_move]])
        ]

        for vector in vectors:
            clamped_result = max(min(result, MAX_Q), MIN_Q)
            # non-end states
            if clamped_result < 1:
                self.train_vectors.append(tuple(vector))
                self.train_labels.append(clamped_result)

    def update_model(self):

        #self.est.fit(np.vstack(self.train_vectors), self.train_labels, sample_weight = np.abs(self.train_labels) + 0.1)
        self.est.fit(np.vstack(self.train_vectors), self.train_labels)
        #self.est.fit(np.vstack(self.train_vectors), self.train_labels)

        max_vectors = 5000
        while len(self.train_vectors) > max_vectors:
            self.train_vectors = self.train_vectors[100:]
            self.train_labels = self.train_labels[100:]
            #sorted_experience = np.array(sorted(zip(self.train_vectors, self.train_labels), reverse=True, key = lambda x: x[1]))

            #self.train_vectors = list(sorted_experience[: int(max_vectors / 2), 0])
            #self.train_labels = list(sorted_experience[: int(max_vectors / 2), 1])

        print('Num Train Vectors', len(self.train_vectors))
        #if self.fitted:
        #    self.est.partial_fit(np.vstack(train_vectors), np.vstack(train_labels))
        #else:
        #    self.est.fit(np.vstack(train_vectors), np.vstack(train_labels))

        #self.train_vectors = {}



    def feature_vector(self, board, as_player):
        is_my_move = 1 if board.player_to_move == as_player else -1
        return np.hstack([np.reshape(board.get_matrix(as_player), (-1)), [is_my_move]]).reshape(1, -1)

    # non-recursive, 2-layer minimax
    def minimax_q(self, board, as_player):
        next_actions = []
        next_qs = []
        for x, y in copy(board.available_moves):
            board.hypothetical_move(x, y)
            next_actions.append((x, y))

            if board.game_won():
                next_qs.append(MAX_Q)
            else:
                # opponent tries to maximize her q
                opponent_q = MIN_Q
                for x2, y2 in copy(board.available_moves):
                    board.hypothetical_move(x2, y2)
                    opponent_q = max(self.q(board, -as_player), opponent_q)
                    if board.game_won():
                        opponent_q = MAX_Q
                    board.unmove(x2, y2)

                if len(board.available_moves) == 0:
                    next_qs.append(0)
                else:
                    next_qs.append(-opponent_q)

            board.unmove(x, y)

        return next_actions, next_qs

    # turn = 1 if my turn, -1 if opponent's
    def q(self, board, as_player):
        prediction = self.est.predict(self.feature_vector(board, as_player))[0]
        return prediction

    def save(self, file):
        joblib.dump(self.est, file)

    def load(self, file):
        self.est = joblib.load(file)