from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from core.board import Board

import numpy as np

import random

from sklearn.metrics import mean_absolute_error

from collections import defaultdict

random_state = np.random.RandomState(42)

MY_PERSPECTIVE = 1
OPPONENT_PERSPECTIVE = -1


class BasicMind:
    def __init__(self, size, epsilon, alpha):
        # features = grid + identifier for whose turn it is
        # constant learning rate
        #self.est = MLPRegressor(hidden_layer_sizes=(size ** 2 + 1), learning_rate_init=0.001, )
        #self.est = DecisionTreeRegressor('mae')
        #self.est = RandomForestRegressor(n_estimators = 4)
        self.est = GradientBoostingRegressor(loss='lad', n_estimators=5, max_depth=20)
        self.est.fit(random_state.randint(size=(1, size ** 2 + 1), low = -1, high = 2), np.zeros(1))

        # zero out the MLP
        #for i in range(len(self.est.intercepts_)):
        #    self.est.intercepts_[i] = 0
        #for x in self.est.coefs_:
        #    x.fill(0)

        self.train_vectors = defaultdict(float)
        self.fitted = False

        self.alpha = alpha
        self.epsilon = epsilon

    # with epsilon probability will select random move
    # returns whether game has concluded or not
    def make_move(self, board, as_player, epsilon = 0.05, alpha = 0.05):
        decision_x = None
        decision_y = None
        current_q = self.q(board, as_player)
        assert(as_player == board.player_to_move)
        # find optimal decision
        next_actions = []
        next_qs = []
        # check each move
        assert(len(board.available_moves) <= board.size ** 2)
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

        for i in range(len(next_qs)):
            print('Move (', next_actions[i][0], ', ', next_actions[i][1], ') : ', next_qs[i])
        new_q = (1 - alpha) * current_q + alpha * np.max(next_qs)
        self.add_train_example(self.feature_vector(board, as_player), new_q)
        self.add_train_example(self.feature_vector(board, -as_player), -new_q)

        result = board.make_move(decision_x, decision_y)
        # game over
        reward = 0
        if result is True:
            reward = 1
            self.add_train_example(self.feature_vector(board, as_player), reward)
            self.add_train_example(self.feature_vector(board, -as_player), -reward)

        if result is not None:
            return True

        return False

    def add_train_example(self, vector, result):
        self.train_vectors[tuple(vector[0])] = result

    def update_model(self):
        #if self.fitted:
        #    self.est.partial_fit(np.vstack(self.train_vectors), np.vstack(self.train_results))
        #else:
        train_vectors = []
        train_labels = []
        for vector in self.train_vectors.keys():
            train_vectors.append(vector)
            train_labels.append(self.train_vectors[vector])
        self.est.fit(train_vectors, np.log1p(np.abs(train_labels)) * np.sign(train_labels))



    def feature_vector(self, board, as_player):
        is_my_move = 1 if board.player_to_move == as_player else -1
        return np.hstack([np.reshape(board.get_matrix(as_player), (-1)), [is_my_move]]).reshape(1, -1)

    # turn = 1 if my turn, -1 if opponent's
    def q(self, board, as_player):
        prediction = self.est.predict(self.feature_vector(board, as_player))[0]
        return np.sign(prediction) * np.expm1(np.abs(prediction))