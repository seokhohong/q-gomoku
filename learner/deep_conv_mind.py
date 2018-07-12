import copy

import numpy as np
from keras.layers import Input, Convolution2D, Dense, Dropout, Flatten, concatenate
from keras.models import Model  # basic class for specifying and training a neural networkf
from keras import losses
import keras

import core.board
from core.board import GameState
from core import minimax

random_state = np.random.RandomState(42)

MIN_Q = -1
MAX_Q = 1

class DeepConvMind:
    def __init__(self, size, alpha, turn_input=True):

        assert(size == 5)
        self.size = size

        self.est = self.get_model()

        # initialization
        init_examples = 10

        labels = np.random.random((init_examples))
        self.est.fit(x=[
                                random_state.randint(size=(init_examples, size, size), low = -1, high = 2).reshape(-1, size, size, 1),
                                np.ones(init_examples).reshape(init_examples, -1),
                            ], y=labels)

        self.train_vectors = []
        self.train_labels = []
        self.sample_weights = []

        self.fitted = False

        self.turn_input = turn_input

        self.alpha = alpha

    def get_model(self):
        height = self.size
        width = self.size
        kernel_size = 3
        conv_depth = 16
        pool_size = 2
        drop_prob_1 = 0.2
        hidden_size = 15
        drop_prob_2 = 0.1

        batch_size = 100
        epochs = 10

        inp = Input(shape=(height, width, 1))
        # key difference between this and conv network is padding
        conv_1 = Convolution2D(conv_depth, (3, 3), padding='valid', activation='relu',
                               kernel_initializer='random_uniform')(inp)
        conv_2 = Convolution2D(conv_depth, (3, 3), padding='same', activation='relu',
                               kernel_initializer='random_uniform')(conv_1)
        # pool_1 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_1)
        drop_1 = Dropout(drop_prob_1)(conv_2)
        # Now flatten to 1D, apply FC -> ReLU
        flat = Flatten()(drop_1)
        turn_input = Input(shape=(1,), name='turn')
        #full = concatenate([flat, turn_input])

        hidden = Dense(hidden_size, activation='relu', kernel_initializer='random_uniform')(flat)
        drop_2 = Dropout(drop_prob_2)(hidden)

        out = Dense(1)(drop_2)

        model = Model(inputs=[inp, turn_input], outputs=out)
        model.compile(loss=losses.mean_absolute_error, optimizer='adam', metrics=['mean_absolute_error'])

        return model

    # board perception AND move turn perception will always be from the perspective of Player 1
    # Q will always be from the perspective of Player 1 (Player 1 Wins = Q = 1, Player -1 Wins, Q = -1)

    def pvs_best_moves(self, board, max_iters=10, epsilon=0.01, verbose=True, k=25, max_depth=5):
        root_node = minimax.PVSNode(parent=None,
                                    is_maximizing=True if board.player_to_move == 1 else False,
                                    full_move_list=minimax.MoveList(moves=()))

        principle_variations = [root_node]

        for i in range(max_iters):
            self.pvs_batch_q(board, principle_variations)
            principle_variations = root_node.get_k_principle_variations(k=k, max_depth=max_depth)

            if len(principle_variations) == 0:
                print("Exhausted Search")
                break

            # game winning path
            if abs(principle_variations[0].principle_variation.q - board.player_to_move) < 1E-6:
                if verbose:
                    # if early termination for a deep node, generally means the nnet has a lot to learn
                    print("Early Search termination: Found Win")
                break

            # find best node (highest q)
        possible_moves = root_node.get_sorted_moves()

        for move, q in possible_moves:
            node = root_node.children[move]
            print(str(node.principle_variation))

        return possible_moves

    def pvs(self, board, max_iters=10, epsilon=0.01, verbose=True, max_depth=5, k=25):

        # array of [best_move, best_node]
        possible_moves = self.pvs_best_moves(board,
                                             max_iters=max_iters,
                                             epsilon=epsilon,
                                             verbose=verbose,
                                             k=k,
                                             max_depth=max_depth)

        # best action is 0th index
        picked_action = 0

        # pick a suboptimal move
        if random_state.random_sample() < epsilon:
            if verbose:
                print('suboptimal move')
            # abs is only there to handle floating point problems
            qs = np.array([node.q for _, node in possible_moves])
            if board.player_to_move == 1:
                distribution = np.abs(qs + 1.0) / 2
            else:
                # not sure this is correct
                distribution = sorted(-np.abs(qs - 1.0) / 2)

            if sum(distribution) > 0:
                distribution = (distribution.astype(np.float64) / sum(distribution))
                picked_action = np.random.choice(range(len(possible_moves)), 1, p=distribution)[0]

        best_move, best_node = possible_moves[picked_action]

        # this player flip is super awkward...
        return best_move, best_node.q * board.player_to_move

    def negamax_feature_vector(self, board):
        # essentially looking at "what is the value of my move (I've already made the move so I back up)"
        return board.get_matrix(as_player=-board.player_to_move), -board.player_to_move

    # q_memory is a dict[(move1, move2, move3, ...)] = (q, game over, True/false)
    # q_memory / board should be treated as references
    def pvs_batch_q(self, board, nodes_to_expand):

        for parent in nodes_to_expand:
            for move in parent.full_move_list.moves:
                board.hypothetical_move(move[0], move[1])

            if board.game_won():
                print('game over??!')

            for move in parent.full_move_list.moves:
                board.unmove()

        # each board state is defined by a list of moves
        q_search_nodes = []
        q_search_vectors = []
        q_search_player = []

        validation_matrix = np.copy(board.matrix)

        for parent in nodes_to_expand:
            # for each move except the last, make rapid moves on board
            for move in parent.full_move_list.moves:
                board.hypothetical_move(move[0], move[1])

            for child_move in copy.copy(board.available_moves):
                child = parent.create_child(child_move)
                board.hypothetical_move(child_move[0], child_move[1])
                # if game is over, then we have our q
                if board.game_won():
                    # the player who last move won!
                    #print('Game Won', child.full_move_list.moves, -board.player_to_move)
                    if len(child.full_move_list.moves) == 1:
                        print('win now')
                    #child.assign_q(-1, core.board.GameState.WON)
                    child.assign_q(-board.player_to_move, core.board.GameState.WON)

                elif board.game_drawn():
                    child.assign_q(0, core.board.GameState.DRAW)

                else:
                    q_search_nodes.append(child)
                    vector, player = self.negamax_feature_vector(board)
                    q_search_vectors.append(vector)
                    #q_search_player.append(-board.player_to_move)
                    q_search_player.append(player)

                # unmove for child
                board.unmove()

            # unwind parent
            for i in range(len(parent.full_move_list)):
                board.unmove()

        assert(np.array_equal(board.matrix, validation_matrix))

        if len(q_search_vectors) == 0:
            return False

        board_vectors = np.array(q_search_vectors).reshape(len(q_search_nodes), self.size, self.size, 1)

        # this helps parallelize
        # multiplication is needed to flip the Q (adjust perspective)
        if self.turn_input:
            predictions = np.clip(
                    np.array(q_search_player) * self.est.predict([board_vectors, np.array(q_search_player)],
                                                        batch_size=32).reshape(len(q_search_nodes)),
                    a_max=minimax.TreeNode.MAX_Q - 0.01,
                    a_min=minimax.TreeNode.MIN_Q + 0.01
            )

        else:
            predictions = np.clip(
                    np.array(q_search_player) * self.est.predict([board_vectors],
                                                        batch_size=32).reshape(len(q_search_nodes)),
                    a_max=minimax.TreeNode.MAX_Q - 0.01,
                    a_min=minimax.TreeNode.MIN_Q + 0.01
            )

        for i, leaf in enumerate(q_search_nodes):
            # update with newly computed q's (only an assignment since approx, we'll compute minimax q's later)
            leaf.assign_q(predictions[i], GameState.NOT_OVER)

        # for all the nodes whose leaves' q's are calculated
        for parent in nodes_to_expand:
            parent.recalculate_q()

        return True

    def q(self, board, as_player):
        if self.turn_input:
            prediction = self.est.predict([[board.get_matrix(as_player).reshape(board.size, board.size, -1)], np.array([as_player])])[0][0]
        else:
            prediction = self.est.predict(
                [[board.get_matrix(as_player).reshape(board.size, board.size, -1)]])[0][0]
        return prediction

    # with epsilon probability will select random move
    # returns whether game has concluded or not
    def make_move(self, board, as_player, retrain=True, verbose=True, epsilon=0.1, max_depth=5, max_iters=10, k=25):
        current_q = self.q(board, as_player)
        assert(as_player == board.player_to_move)

        move, best_q = self.pvs(board,
                                epsilon=epsilon,
                                verbose=verbose,
                                max_depth=max_depth,
                                max_iters=max_iters,
                                k=k)

        new_q = (1 - self.alpha) * current_q + self.alpha * best_q
        print(current_q, best_q)
        self.add_train_example(board, as_player, new_q)
        self.add_train_example(board, -as_player, -new_q)

        board.make_move(move[0], move[1])

        if board.game_over():
            if retrain:
                self.update_model()
            return True

        return False

    # adds rotations
    def add_train_example(self, board, as_player, result, invert_board=False):
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

        max_vectors = 10000
        while len(self.train_vectors) > max_vectors:
            self.train_vectors = self.train_vectors[100:]
            self.train_labels = self.train_labels[100:]
            self.sample_weights = self.sample_weights[100:]

        print('Num Train Vectors', len(self.train_vectors))

    def save(self, file):
        self.est.save(file)

    def load(self, file):
        self.est = keras.models.load_model(file)