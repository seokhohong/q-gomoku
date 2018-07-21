from copy import copy

import numpy as np
from keras.layers import Input, Convolution2D, Dense, Dropout, Flatten, concatenate
from keras.models import Model  # basic class for specifying and training a neural network
import keras

random_state = np.random.RandomState(42)

MIN_Q = -1
MAX_Q = 1

class ConvMind:
    def __init__(self, size, alpha):

        assert(size == 5)
        self.size = size

        height = size
        width = size
        kernel_size = 3
        conv_depth = 16
        pool_size = 2
        drop_prob_1 = 0.2
        hidden_size = 15
        drop_prob_2 = 0.1

        batch_size = 100
        epochs = 10

        inp = Input(shape=(height, width, 1))
        conv_1 = Convolution2D(conv_depth, (kernel_size, kernel_size), padding='valid', activation='relu')(inp)
        #pool_1 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_1)
        drop_1 = Dropout(drop_prob_1)(conv_1)
        # Now flatten to 1D, apply FC -> ReLU
        flat = Flatten()(drop_1)
        turn_input = Input(shape=(1,), name='turn')
        full = concatenate([flat, turn_input])
        hidden = Dense(hidden_size, activation='relu')(full)
        drop_3 = Dropout(drop_prob_2)(hidden)
        out = Dense(1)(drop_3)

        self.est = Model(inputs=[inp, turn_input], outputs=out)
        self.est.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])

        from keras.callbacks import TensorBoard
        # indicate folder to save, plus other options
        tensorboard = TensorBoard(log_dir='/Users/hseokho/PycharmProjects/q-gomoku/logs/', histogram_freq=0,
                                  write_graph=True, write_images=False)
        self.callbacks_list = []

        self.est.fit(x=[
                                random_state.randint(size=(1, size, size), low = -1, high = 2).reshape(-1, size, size, 1),
                                np.ones(1).reshape(-1, 1),
                            ], y=np.zeros(1))

        self.train_vectors = []
        self.train_labels = []
        self.sample_weights = []

        self.fitted = False

        self.alpha = alpha

    # with epsilon probability will select random move
    # returns whether game has concluded or not
    def make_move(self, board, as_player, retrain=True, verbose=True, epsilon=0.1, max_depth=2, max_iters=2):

        current_q = self.q(board, as_player)
        assert(as_player == board.player_to_move)


        # check each move
        assert(len(board.available_moves) <= board.size ** 2)
        next_actions, next_qs = self.minimax_q(board)

        best_q = np.max(next_qs)
        options = []
        for i in range(len(next_qs)):
            if next_qs[i] == best_q:
                options.append(i)
        random_state.shuffle(options)
        decision_x, decision_y = next_actions[options[0]]

        # occasionally try suboptimal move?
        if random_state.random_sample() < epsilon:
            if verbose:
                print('suboptimal move')
            # abs will fix any floating irregularities
            distribution = np.abs(np.array(next_qs) + 1.0) / 2
            distribution = distribution / sum(distribution)
            picked_action = np.random.choice(range(len(next_actions)), 1, p=distribution)[0]
            decision_x, decision_y = next_actions[picked_action]


        for next_action, next_q in sorted(zip(next_actions, next_qs), key=lambda x: x[1], reverse=True):
            if verbose:
                print('Move (', next_action[0], ', ', next_action[1], ') : ', next_q)

        new_q = (1 - self.alpha) * current_q + self.alpha * np.max(next_qs)
        self.add_train_example(board, as_player, new_q)
        self.add_train_example(board, -as_player, -new_q)

        board.make_move(decision_x, decision_y)

        if board.game_over():
            if retrain:
                self.update_model()
            return True

        return False

    # adds rotations
    def add_train_example(self, board, as_player, result):
        is_my_move = 1 if board.player_to_move == as_player else -1
        board_vectors = board.get_rotated_matrices(as_player=as_player)

        weight_shift = 0.1

        for vector in board_vectors:
            clamped_result = max(min(result, MAX_Q), MIN_Q)
            self.train_vectors.append((vector, is_my_move))
            self.train_labels.append(clamped_result)
            self.sample_weights.append(min(abs(clamped_result) + weight_shift, 1))

    def update_model(self):

        train_inputs = [[], []]
        for vector, whose_move in self.train_vectors:
            train_inputs[0].append(vector.reshape(self.size, self.size, 1))
            train_inputs[1].append(whose_move)

        #self.est.fit(np.vstack(train_vectors), np.sign(train_labels) * np.sqrt(np.abs(train_labels)))
        #self.est.fit(np.vstack(train_vectors), train_labels)

        weight_shift = 0.1

        print(len(self.train_vectors))
        if len(self.train_vectors) > 0:
            self.est.fit(x=train_inputs,
                         y=self.train_labels,
                         validation_split=0.1,
                         sample_weight=np.abs(np.array(self.train_labels)) + weight_shift)

        #self.train_vectors = []
        #self.train_labels = []

        max_vectors = 10000
        while len(self.train_vectors) > max_vectors:
            self.train_vectors = self.train_vectors[100:]
            self.train_labels = self.train_labels[100:]
            self.sample_weights = self.sample_weights[100:]



        print('Num Train Vectors', len(self.train_vectors))
        #if self.fitted:
        #    self.est.partial_fit(np.vstack(train_vectors), np.vstack(train_labels))
        #else:
        #    self.est.fit(np.vstack(train_vectors), np.vstack(train_labels))

        #self.train_vectors = {}



    def feature_vector(self, board, as_player):
        is_my_move = 1 if board.player_to_move == as_player else -1
        return board.get_matrix(as_player)

    # non-recursive, 2-layer minimax
    def minimax_q(self, board):
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
                    opponent_q = max(self.q(board, -board.player_to_move), opponent_q)
                    if board.game_won():
                        opponent_q = MAX_Q
                    board.unmove()
                # flip q
                if len(board.available_moves) == 0:
                    next_qs.append(0)
                else:
                    next_qs.append(-opponent_q)

            board.unmove()

        return next_actions, next_qs

    # turn = 1 if my turn, -1 if opponent's
    def q(self, board, as_player):
        prediction = self.est.predict([[board.get_matrix(as_player).reshape(board.size, board.size, -1)], np.array([as_player])])[0][0]
        return prediction

    def save(self, file):
        self.est.save(file)

    def load(self, file):
        self.est = keras.models.load_model(file)