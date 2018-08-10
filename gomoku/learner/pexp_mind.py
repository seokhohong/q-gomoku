import copy

import numpy as np
from keras.layers import Input, Convolution2D, Dense, Dropout, Flatten, concatenate, BatchNormalization
from keras.models import Model  # basic class for specifying and training a neural network
from keras import losses
import keras
import pandas as pd
import tensorflow as tf

from sortedcontainers import SortedSet
from sortedcontainers import SortedList

from core.board import GameState
from core.board import Board
from core import optimized_minimax
from core.optimized_minimax import PExpNode


class PExpMind:
    def __init__(self, size, alpha, init=True, channels=1, debug=True):

        self.size = size
        self.channels = channels
        self.debug = debug

        if self.size == 7:
            self.value_est = self.value_model_7()
            self.policy_est = self.policy_model_7()
        elif self.size == 9:
            self.value_est = self.value_model_9()
            self.policy_est = self.policy_model_9()
        else:
            self.value_est = self.get_value_model()
            self.policy_est = self.get_policy_model()

        # initialization
        init_examples = 10

        if init:
            sample_x = [
                            np.random.randint(-1, 1, size=(init_examples, size, size, self.channels))
                        ]
            self.value_est.fit(sample_x, y=np.zeros(init_examples), epochs=1, batch_size=100)
            self.policy_est.fit(sample_x, y=np.zeros((init_examples, self.size ** 2)))

        self.train_vectors = []
        self.train_q = []
        self.train_p = []

        self.fitted = False

        self.alpha = alpha

        self.memory_root = None

    def value_model_7(self):
        inp = Input(shape=(self.size, self.size, self.channels))

        # key difference between this and conv network is padding
        conv_1 = Convolution2D(64, (3, 3), padding='valid', activation='relu',
                               kernel_initializer='random_normal', use_bias=False)(inp)
        bn2 = BatchNormalization()(conv_1)
        conv_2 = Convolution2D(32, (3, 3), padding='valid', activation='relu',
                               kernel_initializer='random_normal', use_bias=False)(bn2)
        bn3 = BatchNormalization()(conv_2)

        flat = Flatten()(bn3)

        hidden = Dense(10, activation='relu', kernel_initializer='random_normal', use_bias=False)(flat)
        bn_final = BatchNormalization()(hidden)

        out = Dense(1, use_bias=False)(bn_final)

        model = Model(inputs=[inp], outputs=out)
        model.compile(loss=losses.mean_squared_error, optimizer='adam', metrics=['mean_squared_error'])

        return model

    def value_model_9(self):
        inp = Input(shape=(self.size, self.size, self.channels))

        # key difference between this and conv network is padding
        conv_1 = Convolution2D(64, (3, 3), padding='valid', activation='relu',
                               kernel_initializer='random_normal', use_bias=False)(inp)
        bn2 = BatchNormalization()(conv_1)
        conv_2 = Convolution2D(32, (3, 3), padding='valid', activation='relu',
                               kernel_initializer='random_normal', use_bias=False)(bn2)
        bn3 = BatchNormalization()(conv_2)

        flat = Flatten()(bn3)

        hidden = Dense(10, activation='relu', kernel_initializer='random_normal', use_bias=False)(flat)
        bn_final = BatchNormalization()(hidden)

        out = Dense(1, use_bias=False)(bn_final)

        model = Model(inputs=[inp], outputs=out)
        model.compile(loss=losses.mean_squared_error, optimizer='adam', metrics=['mean_squared_error'])

        return model

    def policy_model_7(self):
        inp = Input(shape=(self.size, self.size, self.channels))

        # key difference between this and conv network is padding
        conv_1 = Convolution2D(64, (3, 3), padding='same', activation='relu',
                               kernel_initializer='random_normal', use_bias=False)(inp)
        bn2 = BatchNormalization()(conv_1)
        conv_2 = Convolution2D(32, (3, 3), padding='same', activation='relu',
                               kernel_initializer='random_normal', use_bias=False)(bn2)
        bn3 = BatchNormalization()(conv_2)
        conv_3 = Convolution2D(16, (3, 3), padding='same', activation='relu',
                               kernel_initializer='random_normal', use_bias=False)(bn3)
        bn4 = BatchNormalization()(conv_3)
        conv_4 = Convolution2D(8, (3, 3), padding='same', activation='relu',
                               kernel_initializer='random_normal', use_bias=False)(bn4)
        bn5 = BatchNormalization()(conv_4)

        flat = Flatten()(bn5)

        hidden = Dense(self.size ** 2, activation='relu', kernel_initializer='random_normal', use_bias=False)(flat)
        bn_final = BatchNormalization()(hidden)

        out = Dense(self.size ** 2, activation='softmax')(bn_final)

        model = Model(inputs=[inp], outputs=out)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model


    def policy_model_9(self):
        inp = Input(shape=(self.size, self.size, self.channels))

        # key difference between this and conv network is padding
        conv_1 = Convolution2D(64, (3, 3), padding='same', activation='relu',
                               kernel_initializer='random_normal', use_bias=False)(inp)
        bn2 = BatchNormalization()(conv_1)
        conv_2 = Convolution2D(64, (3, 3), padding='same', activation='relu',
                               kernel_initializer='random_normal', use_bias=False)(bn2)
        bn3 = BatchNormalization()(conv_2)
        conv_3 = Convolution2D(64, (3, 3), padding='same', activation='relu',
                               kernel_initializer='random_normal', use_bias=False)(bn3)
        bn4 = BatchNormalization()(conv_3)
        conv_4 = Convolution2D(32, (3, 3), padding='same', activation='relu',
                               kernel_initializer='random_normal', use_bias=False)(bn4)
        bn5 = BatchNormalization()(conv_4)
        conv_5 = Convolution2D(16, (3, 3), padding='same', activation='relu',
                               kernel_initializer='random_normal', use_bias=False)(bn5)
        bn6 = BatchNormalization()(conv_5)

        flat = Flatten()(bn6)

        hidden = Dense(self.size ** 2, activation='relu', kernel_initializer='random_normal', use_bias=False)(flat)
        bn_final = BatchNormalization()(hidden)

        out = Dense(self.size ** 2, activation='softmax')(bn_final)

        model = Model(inputs=[inp], outputs=out)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model

    def get_value_model(self):
        inp = Input(shape=(self.size, self.size, self.channels))

        # key difference between this and conv network is padding
        conv_1 = Convolution2D(64, (3, 3), padding='valid', activation='relu',
                               kernel_initializer='random_normal', use_bias=False)(inp)
        bn2 = BatchNormalization()(conv_1)
        conv_2 = Convolution2D(32, (3, 3), padding='valid', activation='relu',
                               kernel_initializer='random_normal', use_bias=False)(bn2)
        bn3 = BatchNormalization()(conv_2)
        #conv_3 = Convolution2D(32, (3, 3), padding='valid', activation='relu',
        #                       kernel_initializer='random_normal', use_bias=False)(bn3)
        #bn4 = BatchNormalization()(conv_3)
        #conv_4 = Convolution2D(16, (3, 3), padding='valid', activation='relu',
        #                       kernel_initializer='random_normal', use_bias=False)(bn4)
        #bn5 = BatchNormalization()(conv_4)

        flat = Flatten()(bn3)

        hidden = Dense(10, activation='relu', kernel_initializer='random_normal', use_bias=False)(flat)
        bn_final = BatchNormalization()(hidden)

        out = Dense(1, use_bias=False)(bn_final)

        model = Model(inputs=[inp], outputs=out)
        model.compile(loss=losses.mean_squared_error, optimizer='adam', metrics=['mean_squared_error'])

        return model

    def get_policy_model(self):
        inp = Input(shape=(self.size, self.size, self.channels))

        # key difference between this and conv network is padding
        conv_1 = Convolution2D(64, (3, 3), padding='same', activation='relu',
                               kernel_initializer='random_normal', use_bias=False)(inp)
        bn2 = BatchNormalization()(conv_1)
        conv_2 = Convolution2D(32, (3, 3), padding='same', activation='relu',
                               kernel_initializer='random_normal', use_bias=False)(bn2)
        bn3 = BatchNormalization()(conv_2)
        conv_3 = Convolution2D(16, (3, 3), padding='same', activation='relu',
                               kernel_initializer='random_normal', use_bias=False)(bn3)
        bn4 = BatchNormalization()(conv_3)
        #conv_4 = Convolution2D(8, (3, 3), padding='same', activation='relu',
        #                       kernel_initializer='random_normal', use_bias=False)(bn4)
        #bn5 = BatchNormalization()(conv_4)

        flat = Flatten()(bn4)

        hidden = Dense(10, activation='relu', kernel_initializer='random_normal', use_bias=False)(flat)
        bn_final = BatchNormalization()(hidden)

        out = Dense(self.size ** 2, activation='softmax')(bn_final)

        model = Model(inputs=[inp], outputs=out)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model

    def pvs_k_principal_variations(self, leaf_nodes, k=5):
        # include the best move according to q
        principal_variations = []
        for item in leaf_nodes:
            if item.game_status == GameState.NOT_OVER:
                principal_variations.append(item)
            if len(principal_variations) == k:
                break
        return principal_variations

    # board perception will always be from the perspective of Player 1
    # Q will always be from the perspective of Player 1 (Player 1 Wins = Q = 1, Player -1 Wins, Q = -1)

    def pvs_best_moves(self, board, max_iters=10, k=25, required_depth=5, root_node=None, p_threshold=1):
        is_maximizing = True if board.player_to_move == 1 else False

        if root_node is None:
            root_node = optimized_minimax.PExpNode(parent=None,
                                                is_maximizing=is_maximizing,
                                                full_move_list=optimized_minimax.MoveList(moves=(), position_hash=[]))

        principal_variations = [root_node]

        transposition_table = {}

        # all nodes at the leaves of the search tree
        leaf_nodes = SortedList(principal_variations, key=lambda x: x.p_comparator)

        #assert(len(leaf_nodes) == len(set(leaf_nodes)))

        explored_states = 1
        i = 0
        MAXED_DURATION = 5

        previously_removed = set()

        principal_qs = []
        while i < max_iters or len(root_node.principal_variation.full_move_list) < required_depth:
            i += 1

            # eternal loop just in case
            if i > max_iters * 5:
                break

            # searching doesn't get us anywhere for awhile
            if root_node.principal_variation and root_node.principal_variation.q:
                principal_qs.append(root_node.principal_variation.q)
                if len(principal_qs) > MAXED_DURATION and np.abs(np.mean(principal_qs[-MAXED_DURATION : -1])) == 1:
                    print('Maxed Duration')
                    break

            #assert(len(leaf_nodes) == len(set(root_node.recursive_children())))

            # p search
            self.p_expand(board, principal_variations, leaf_nodes, transposition_table, previously_removed, p_threshold)

            # don't need to prepare for next iteration

            principal_variations = self.pvs_k_principal_variations(leaf_nodes, k=k)
            # nothing left
            if not principal_variations:
                print("Exhausted Search")
                break

            # P will already have been expanded so do a Q eval
            self.q_eval_top_leaves(leaf_nodes, min_eval_q=k * 2, max_eval_q=k * 2)

            next_pvs = self.highest_leaf_qs(leaf_nodes, is_maximizing, max_p_eval=k ** 2, num_leaves=k)

            principal_variations.extend(next_pvs)

            #root_node.consistent_pv()
            if root_node.principal_variation:
                # each of these nodes will be a leaf with P=0, making them evaluated next time around
                while root_node.principal_variation.has_children():
                    print('pv Q extension')
                    self.q_eval([node for node in root_node.principal_variation.children.values()
                                 if node.game_status == GameState.NOT_OVER and not node.has_children()])

            # this check doesn't hold if we do PV Q extensions
            #if root_node.principal_variation:
            #    root_node.top_down_q()
            #    assert(abs(root_node.principal_variation.q - root_node.negamax()[0]) < 1E-4)

            # if we have a PV, add it to expand
            if root_node.principal_variation and root_node.principal_variation.game_status == GameState.NOT_OVER:
                assert(root_node.principal_variation not in previously_removed)
                assert(not root_node.principal_variation.has_children())
                principal_variations.append(root_node.principal_variation)

            principal_variations = set(principal_variations)

            print('Root', root_node.principal_variation)

        q_stats, p_stats = root_node.recursive_stats()
        print('Explored ' + str(p_stats) + " States (Q evals: " + str(q_stats) + ") in " + str(i) + " Iterations")

        possible_moves = root_node.get_sorted_moves()

        for move, q in possible_moves:
            node = root_node.children[move]
            print('Move', move, str(node.principal_variation))

        return possible_moves, root_node


    def highest_leaf_qs(self, leaf_nodes, is_maximizing, max_p_eval=100, num_leaves=10):
        num_eval = min(max_p_eval, len(leaf_nodes))
        best_leaves = sorted([leaf for leaf in leaf_nodes.islice(0, num_eval) if leaf.assigned_q], key=lambda x: x.q, reverse=not is_maximizing)
        return best_leaves[:num_leaves]

    def q_eval(self, nodes):
        parents_to_update = set()
        board_matrices = []
        nodes = [node for node in nodes if not node.assigned_q]
        for leaf in nodes:
            # normally not, but it can be if nodes = PV's children
            assert(leaf.game_status == GameState.NOT_OVER)
            assert(not leaf.has_children())
            parents_to_update.update(leaf.parents)
            board_matrices.append(leaf.get_matrix())

        # if nothing to eval, get out
        if len(board_matrices) == 0:
            return

        # attach q predictions to all leaves and compute q tree at once
        q_predictions = np.clip(
                    self.value_est.predict(np.array(board_matrices),
                                                        batch_size=1000).reshape(len(board_matrices)),
                    a_max=optimized_minimax.PExpNode.MAX_MODEL_Q,
                    a_min=optimized_minimax.PExpNode.MIN_MODEL_Q
            )

        for i, leaf in enumerate(nodes):
            # it's possible to have transposition assign q's even if we filtered above
            if not leaf.assigned_q:
                leaf.assign_q(q_predictions[i], GameState.NOT_OVER)

        for parent in parents_to_update:
            parent.recalculate_q()

    def q_eval_top_leaves(self, leaf_nodes, min_eval_q=10, max_eval_q=1000):
        # will only contain leaves of games not being over
        num_eval = min(max(min_eval_q, len(leaf_nodes)), max_eval_q)
        best_leaves = leaf_nodes.islice(0, num_eval)
        self.q_eval(best_leaves)

    accessed_transposition = 0
    # use
    def p_expand(self, board, nodes_to_expand, leaf_nodes, transposition_table, previously_removed, p_threshold = -5):
        for parent in nodes_to_expand:
            assert(parent not in previously_removed)
            # should NOT be expanding any non-leaf node
            assert(parent in leaf_nodes)
            leaf_nodes.remove(parent)
            previously_removed.add(parent)

        prev_board = board.get_matrix()

        # each board state is defined by a list of moves
        p_search_vectors = []
        p_search_nodes = list(nodes_to_expand)

        for parent in nodes_to_expand:
            # for each move except the last, make rapid moves on board
            for move in parent.full_move_list.moves:
                board.blind_move(move[0], move[1])

            p_search_vectors.append(board.get_matrix())

            for _ in parent.full_move_list.moves:
                board.unmove()

        p_board_vectors = np.array(p_search_vectors).reshape(len(p_search_vectors), self.size, self.size,
                                                             self.channels)

        log_p_predictions = np.log(self.policy_est.predict([p_board_vectors], batch_size=len(p_board_vectors))).reshape(-1)

        parent_index = np.hstack([np.full(shape=(self.size ** 2), fill_value=i) for i in range(len(nodes_to_expand))])
        move_index = np.array([list(range(self.size ** 2)) * len(nodes_to_expand)])
        zipped_predictions = np.vstack([parent_index, move_index, log_p_predictions]).transpose()
        filtered_predictions = zipped_predictions[zipped_predictions[:, 2] > p_threshold]

        print('Num P over threshold', len(filtered_predictions), 'out of', (len(nodes_to_expand) * self.size ** 2))

        new_leaves = []
        q_update = set()

        # optimize for parent
        for parent_index, move_index, log_prediction in filtered_predictions:
            parent = p_search_nodes[int(parent_index)]

            for move in parent.full_move_list.moves:
                board.blind_move(move[0], move[1])

            predicted_move = self.index_to_move(move_index)
            if predicted_move in board.available_moves:
                child, made_new_child = parent.create_child(predicted_move, transposition_table)
                if made_new_child:
                    board.move(predicted_move[0], predicted_move[1])
                    child.assign_p(log_prediction)
                    # if game is over, then we have our q
                    if board.game_won():
                        # the player who last move won!
                        if len(child.full_move_list.moves) == 1:
                            print('win now')
                        winning_q = PExpNode.MIN_Q if board.player_to_move == Board.FIRST_PLAYER else PExpNode.MAX_Q
                        child.assign_q(winning_q, GameState.WON)
                        q_update.add(parent)

                    elif board.game_drawn():
                        child.assign_q(0, GameState.DRAW)
                        q_update.add(parent)

                    else:
                        assert (child not in previously_removed)
                        new_leaves.append(child)
                        child.set_matrix(board.get_matrix())

                    # unmove for child
                    board.unmove()
                else:
                    PExpMind.accessed_transposition += 1
                    q_update.add(parent)

            for _ in parent.full_move_list.moves:
                board.unmove()

        for parent in q_update:
            parent.recalculate_q()

        leaf_nodes += new_leaves

        assert(np.alltrue(np.equal(prev_board, board.get_matrix())))

        return len(new_leaves) > 0

    def q(self, board):
        prediction = self.value_est.predict([np.array([board.get_matrix().reshape(board.size, board.size, -1)])])[0][0]
        return prediction

    def pick_random_move(self, board, possible_moves):
        picked_action = 0

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

        return picked_action

    # used to pick a branch and fast forward it to use it as root node
    def cleanse_memory_tree(self, root_node, moves):
        root_node.parents = []
        root_node.cleanse_memory(moves)

    # with epsilon probability will select random move
    # returns whether game has concluded or not
    def make_move(self, board, as_player, verbose=True, epsilon=0.1, required_depth=5, max_iters=10, k=25, p_threshold=-10):
        current_q = self.q(board)
        assert(as_player == board.player_to_move)

        # incomplete
        starting_root = None
        if self.memory_root is not None:
            # self-play
            if np.alltrue(np.equal(self.memory_root.get_matrix(), board.get_matrix())):
                starting_root = self.memory_root
            for child in self.memory_root.children.values():
                if np.alltrue(np.equal(child.get_matrix(), board.get_matrix())):
                    #print('Found Previous Child')
                    starting_root = child
                    break

        # array of [best_move, best_node], root node of move calculations
        possible_moves, root_node = self.pvs_best_moves(board,
                                             max_iters=max_iters,
                                             k=k,
                                             required_depth=required_depth,
                                             p_threshold=p_threshold,
                                             root_node=None)

        # best action is 0th index
        picked_action = 0

        # pick a suboptimal move randomly
        if np.random.random_sample() < epsilon:
            if verbose:
                print('suboptimal move')
            picked_action = self.pick_random_move(board, possible_moves)

        picked_move, picked_node = possible_moves[picked_action]
        # add training example assuming best move
        best_move, best_node = possible_moves[0]
        best_q = best_node.q

        new_best_q = (1 - self.alpha) * current_q + self.alpha * best_q
        # compress to model valid range
        new_best_q = max(min(new_best_q, optimized_minimax.PExpNode.MAX_MODEL_Q), optimized_minimax.PExpNode.MIN_MODEL_Q)

        print(current_q, best_q)
        self.add_train_example(board, new_best_q, best_move)

        # picked move may not equal best move if we're making a suboptimal one
        board.move(picked_move[0], picked_move[1])

        self.memory_root = root_node.children[picked_move[0], picked_move[1]]

        return board.game_over()

    def one_hot_p(self, move_index):
        vector = np.zeros((self.size ** 2))
        vector[move_index] = 1.
        return vector

    def move_to_index(self, move):
        return move[0] * self.size + move[1]

    def index_to_move(self, index):
        return np.int(index / self.size), np.int(index % self.size)

    # adds rotations
    def add_train_example(self, board, result, move):
        board_vectors = board.get_rotated_matrices()

        for i, vector in enumerate(board_vectors):
            if abs(result) > 0.999:
                print('won')
            self.train_vectors.append(vector)
            self.train_q.append(result)
            # get the i'th rotation
            which_rotation = board.get_rotated_point(self.move_to_index(move))[i]
            self.train_p.append(self.one_hot_p(which_rotation))


    def save(self, filename):
        self.value_est.save(filename + '_value.net')
        self.policy_est.save(filename + '_policy.net')

    def load_net(self, filename):
        self.value_est = keras.models.load_model(filename + '_value.net')
        self.policy_est = keras.models.load_model(filename + '_policy.net')
       
    def load(self, value_file, policy_file):
        self.value_est = keras.models.load_model(value_file)
        self.policy_est = keras.models.load_model(policy_file)