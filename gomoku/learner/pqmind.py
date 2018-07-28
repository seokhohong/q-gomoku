import copy

import numpy as np
from keras.layers import Input, Convolution2D, Dense, Dropout, Flatten, concatenate, BatchNormalization
from keras.models import Model  # basic class for specifying and training a neural network
from keras import losses
import keras
import tensorflow as tf
from sortedcontainers import SortedSet

from core.board import GameState
from core import optimized_minimax

random_state = np.random.RandomState(42)

MIN_Q = -1
MAX_Q = 1

class PQMind:
    def __init__(self, size, alpha, init=True, channels=1):

        self.size = size
        self.channels = channels

        self.value_est = self.get_value_model()
        self.policy_est = self.get_policy_model()


        # initialization
        init_examples = 10

        #from keras import backend as K
        #K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=4, inter_op_parallelism_threads=4)))
        if init:
            sample_x = [
                                    random_state.random_integers(-1, 1, size=(init_examples, size, size, self.channels))
                                ]
            self.value_est.fit(sample_x, y=np.zeros((init_examples)), epochs=1, batch_size=100)
            self.policy_est.fit(sample_x, y=np.zeros((init_examples, self.size ** 2)))

        self.train_vectors = []
        self.train_q = []
        self.train_p = []

        self.fitted = False

        self.alpha = alpha

    def get_value_model(self):
        inp = Input(shape=(self.size, self.size, self.channels))

        # key difference between this and conv network is padding
        conv_1 = Convolution2D(64, (3, 3), padding='same', activation='relu',
                               kernel_initializer='random_normal', use_bias=False)(inp)
        bn2 = BatchNormalization()(conv_1)
        conv_2 = Convolution2D(32, (3, 3), padding='same', activation='relu',
                               kernel_initializer='random_normal', use_bias=False)(bn2)
        bn3 = BatchNormalization()(conv_2)
        conv_3 = Convolution2D(32, (3, 3), padding='valid', activation='relu',
                               kernel_initializer='random_normal', use_bias=False)(bn3)
        bn4 = BatchNormalization()(conv_3)
        conv_4 = Convolution2D(16, (3, 3), padding='valid', activation='relu',
                               kernel_initializer='random_normal', use_bias=False)(bn4)
        bn5 = BatchNormalization()(conv_4)

        flat = Flatten()(bn5)

        hidden = Dense(10, activation='relu', kernel_initializer='random_normal', use_bias=False)(flat)
        bn_final = BatchNormalization()(hidden)

        out = Dense(1, use_bias=False)(bn_final)

        model = Model(inputs=[inp], outputs=out)
        model.compile(loss=losses.mean_squared_error, optimizer='adam', metrics=['mean_squared_error'])

        return model

    def get_policy_model(self):
        inp = Input(shape=(self.size, self.size, self.channels))

        # key difference between this and conv network is padding
        conv_1 = Convolution2D(128, (3, 3), padding='same', activation='relu',
                               kernel_initializer='random_normal', use_bias=False)(inp)
        bn2 = BatchNormalization()(conv_1)
        conv_2 = Convolution2D(64, (3, 3), padding='same', activation='relu',
                               kernel_initializer='random_normal', use_bias=False)(bn2)
        bn3 = BatchNormalization()(conv_2)
        conv_3 = Convolution2D(32, (3, 3), padding='same', activation='relu',
                               kernel_initializer='random_normal', use_bias=False)(bn3)
        bn4 = BatchNormalization()(conv_3)
        conv_4 = Convolution2D(16, (3, 3), padding='same', activation='relu',
                               kernel_initializer='random_normal', use_bias=False)(bn4)
        bn5 = BatchNormalization()(conv_4)

        flat = Flatten()(bn5)

        hidden = Dense(10, activation='relu', kernel_initializer='random_normal', use_bias=False)(flat)
        bn_final = BatchNormalization()(hidden)

        out = Dense(self.size ** 2, activation='softmax')(bn_final)

        model = Model(inputs=[inp], outputs=out)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model

    # after parents are expanded, this method recomputes all leaves
    def pvs_catch_leaves(self, leaf_nodes, new_parents, max_depth=5):
        for parent in new_parents:
            if parent in leaf_nodes:
                leaf_nodes.remove(parent)
            # if parent is not too deep and isn't a game ending state
            if len(parent.full_move_list) < max_depth and abs(parent.q) < 1:
                leaf_nodes.update([node for node in parent.children.values() if node.game_status == GameState.NOT_OVER])

    def pvs_k_principle_variations(self, root_node, leaf_nodes, k=5):
        # include the best move according to q
        q_best = root_node.principle_variation
        # draw
        if q_best is None:
            return None

        # include k-1 most likely moves according to p
        if q_best.game_status == GameState.NOT_OVER:
            return [q_best] + list(leaf_nodes.islice(0, k - 1))
        else:
            return list(leaf_nodes.islice(0, k))

    # board perception AND move turn perception will always be from the perspective of Player 1
    # Q will always be from the perspective of Player 1 (Player 1 Wins = Q = 1, Player -1 Wins, Q = -1)

    def pvs_best_moves(self, board, max_iters=10, epsilon=0.01, verbose=True, k=25, max_depth=5):
        root_node = optimized_minimax.PVSNode(parent=None,
                                    is_maximizing=True if board.player_to_move == 1 else False,
                                    full_move_list=optimized_minimax.MoveList(moves=()))

        principle_variations = [root_node]
        leaf_nodes = SortedSet(principle_variations, key=lambda x: -x.log_total_p)

        for i in range(max_iters):
            self.pvs_batch(board, principle_variations)
            self.pvs_catch_leaves(leaf_nodes, principle_variations, max_depth=max_depth)
            principle_variations = self.pvs_k_principle_variations(root_node, leaf_nodes, k=k)
            if not principle_variations:
                print("Exhausted Search")
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

        return best_move, best_node.q

    def pvs_batch(self, board, nodes_to_expand):

        for parent in nodes_to_expand:
            for move in parent.full_move_list.moves:
                board.move(move[0], move[1])

            if board.game_won():
                print('game over??!')

            for move in parent.full_move_list.moves:
                board.unmove()

        # each board state is defined by a list of moves
        q_search_nodes = []
        q_search_vectors = []
        q_search_player = []

        p_search_vectors = []
        p_search_players = []

        validation_matrix = np.copy(board.matrix)

        for parent in nodes_to_expand:
            # for each move except the last, make rapid moves on board
            for move in parent.full_move_list.moves:
                board.move(move[0], move[1])

            for child_move in copy.copy(board.available_moves):
                child = parent.create_child(child_move)
                board.move(child_move[0], child_move[1])
                # if game is over, then we have our q
                if board.game_won():
                    # the player who last move won!
                    if len(child.full_move_list.moves) == 1:
                        print('win now')
                    child.assign_q(-board.player_to_move, GameState.WON)

                elif board.game_drawn():
                    child.assign_q(0, GameState.DRAW)

                else:
                    q_search_nodes.append(child)
                    vector, player = board.get_matrix(as_player=board.player_to_move), board.player_to_move
                    q_search_vectors.append(vector)
                    q_search_player.append(player)

                # unmove for child
                board.unmove()

            # update p
            vector, player = board.get_matrix(as_player=board.player_to_move), board.player_to_move
            p_search_vectors.append(vector)
            p_search_players.append(player)

            # unwind parent
            for i in range(len(parent.full_move_list)):
                board.unmove()

        assert(np.array_equal(board.matrix, validation_matrix))

        if len(q_search_vectors) == 0:
            # recalculate for wins/draws
            for parent in nodes_to_expand:
                parent.recalculate_q()
            return False

        q_board_vectors = np.array(q_search_vectors).reshape(len(q_search_vectors), self.size, self.size, self.channels)
        p_board_vectors = np.array(p_search_vectors).reshape(len(p_search_vectors), self.size, self.size, self.channels)

        # this helps parallelize
        # multiplication is needed to flip the Q (adjust perspective)
        q_predictions = np.clip(
                    self.value_est.predict([q_board_vectors],
                                                        batch_size=len(q_board_vectors)).reshape(len(q_search_vectors)),
                    a_max=optimized_minimax.PVSNode.MAX_Q - 0.01,
                    a_min=optimized_minimax.PVSNode.MIN_Q + 0.01
            )

        log_p_predictions = np.log(self.policy_est.predict([p_board_vectors],
                                                        batch_size=len(q_board_vectors)).reshape((len(p_search_vectors), self.size ** 2)))

        for i, parent in enumerate(nodes_to_expand):
            for move in parent.children.keys():
                move_index = self.move_to_index(move)
                parent.children[move[0], move[1]].assign_p(log_p_predictions[i][move_index])

        for i, leaf in enumerate(q_search_nodes):
            # update with newly computed q's (only an assignment since approx, we'll compute minimax q's later)
            leaf.assign_q(q_predictions[i], GameState.NOT_OVER)

        # for all the nodes whose leaves' q's are calculated
        for parent in nodes_to_expand:
            parent.recalculate_q()

        return True

    def q(self, board, as_player):
        prediction = self.value_est.predict([np.array([board.get_matrix(as_player).reshape(board.size, board.size, -1)])])[0][0]
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

        # ignore learning rate if outcome is guaranteed
        if abs(best_q) > optimized_minimax.PVSNode.MAX_MODEL_Q:
            new_q = best_q
        else:
            new_q = (1 - self.alpha) * current_q + self.alpha * best_q
            
        print(current_q, best_q)
        self.add_train_example(board, as_player, new_q, move)

        board.move(move[0], move[1])

        if board.game_over():
            if retrain:
                self.update_model()
            return True

        return False

    def one_hot_p(self, move_index):
        vector = np.zeros((self.size ** 2))
        vector[move_index] = 1.
        return vector

    def move_to_index(self, move):
        return move[0] * self.size + move[1]

    def index_to_move(self, index):
        return np.int(index / self.size), index % self.size

    # adds rotations
    def add_train_example(self, board, as_player, result, move, invert_board=False):
        board_vectors = board.get_rotated_matrices(as_player=as_player)

        for i, vector in enumerate(board_vectors):
            clamped_result = max(min(result, MAX_Q), MIN_Q)
            self.train_vectors.append((vector, as_player))
            self.train_q.append(clamped_result)
            # get the i'th rotation
            which_rotation = board.get_rotated_point(self.move_to_index(move))[i]
            self.train_p.append(self.one_hot_p(which_rotation))
            #self.train_p.append(which_rotation)

    def update_model(self):

        train_inputs = [[], []]
        for vector, whose_move in self.train_vectors:
            train_inputs[0].append(vector.reshape(self.size, self.size, 1))
            train_inputs[1].append(whose_move)
            
        train_inputs[0] = np.array(train_inputs[0])
        train_inputs[1] = np.array(train_inputs[1])

        print(len(self.train_vectors))
        with tf.device('/gpu:1'):
            if len(self.train_vectors) > 0:
                self.value_est.fit(x=train_inputs,
                                    y=np.array(self.train_q),
                                    shuffle=True,
                                    validation_split=0.1)
                self.policy_est.fit(x=train_inputs,
                                    y=np.array(self.train_p),
                                    shuffle=True,
                                    validation_split=0.1)

        max_vectors = 5000
        while len(self.train_vectors) > max_vectors:
            self.train_vectors = self.train_vectors[100:]
            self.train_p = self.train_p[100:]
            self.train_q = self.train_q[100:]

        print('Num Train Vectors', len(self.train_vectors))

    def save(self, filename):
        self.value_est.save(filename + '_value.net')
        self.policy_est.save(filename + '_policy.net')

    def load_net(self, filename):
        self.value_est = keras.models.load_model(filename + '_value.net')
        self.policy_est = keras.models.load_model(filename + '_policy.net')
       
    def load(self, value_file, policy_file):
        self.value_est = keras.models.load_model(value_file)
        self.policy_est = keras.models.load_model(policy_file)