import pickle

import keras
import numpy as np
from qgomoku.core.board import BitBoard, GameState, BoardTransform, BitBoardCache, Player
from qgomoku.learner.pexp_node_v3 import PExpNodeV3
from qgomoku.core import minimax
from keras import losses
from keras.layers import Input, Convolution2D, Dense, Flatten, BatchNormalization
from keras.models import Model  # basic class for specifying and training a neural network
from sortedcontainers import SortedList
import copy

from qgomoku.learner.game_to_features import FeatureSet_v1_1, FeatureBoard_v1_1

class TranspositionTable:
    def __init__(self):
        self._transposition_dict = {}
        self._transposition_hits = 0

    def get(self, key):
        if key in self._transposition_dict:
            self._transposition_hits += 1
            return self._transposition_dict[key]
        return None

    def put(self, key, node):
        self._transposition_dict[key] = node

    def get_num_hits(self):
        return self._transposition_hits

# the board that considers hypothetical moves all over the place
class ThoughtBoard:
    def __init__(self, board):
        self.board = board
        self.feature_system = FeatureBoard_v1_1(board)
        # root is used for queries that have a movelist (class functions stateless-ly)
        self.root_board = copy.deepcopy(board)
        self.root_feature_system = FeatureBoard_v1_1(board)
        self.delta = 0

    def make_move(self, move):
        self.board.move(move)
        self.feature_system.move(move)
        self.delta += 1

    def get_hard_q(self):
        winning_player = self.board.get_winning_player()
        if winning_player == Player.FIRST:
            return 1
        elif winning_player == Player.SECOND:
            return -1
        return 0

    def get_hard_q_after(self, move_list):
        return self._call_function_with_board_state(self.root_board.get_winning_player, move_list)

    def get_game_status(self):
        return self.board.game_status()

    def game_over(self):
        return self.board.game_over()

    def game_over_after(self, move_list):
        return self._call_function_with_board_state(self.root_board.game_over, move_list)

    def get_available_move_vector_after(self, move_list):
        move_vector = self.root_feature_system.get_init_available_move_vector()
        for move in move_list:
            move_vector[move] = 0
        return move_vector

    # puts the internal featureboard into a state as defined by move_list, then operates this function
    def _call_function_with_feature_state(self, func, move_list):
        for move in move_list:
            self.root_feature_system.move(move)

        value = func()

        for i in range(len(move_list)):
            self.root_feature_system.unmove()

        return value

    def _call_function_with_board_state(self, func, move_list):
        for move in move_list:
            self.root_board.blind_move(move)

        self.root_board.compute_game_state()
        value = func()

        for i in range(len(move_list)):
            self.root_board.unmove()

        return value

    def unmove(self):
        self.board.unmove()
        self.feature_system.unmove()
        self.delta -= 1
        assert self.delta >= 0

    def get_q_features(self):
        return self.feature_system.get_q_features()

    # does not check if move_list is a valid set of moves
    # works right now because p_features = q_features
    def get_q_features_after(self, move_list):
        return self._call_function_with_feature_state(self.root_feature_system.get_q_features, move_list)

    def get_p_features_after(self, move_list):
        return self._call_function_with_feature_state(self.root_feature_system.get_p_features, move_list)

    def make_moves(self, move_list):
        for move in move_list:
            self.make_move(move)

    # returns a copy
    def available_moves(self):
        return set(self.board.get_available_moves())

    # not sure if faster to undo or copy
    def reset(self):
        for i in range(self.delta):
            self.unmove()

class PEvenSearch:
    def __init__(self,
                 board,
                 policy_est,
                 value_est,
                 max_iterations=10,
                 p_batch_size=2 ** 13,
                 fraction_q=0.2,
                 min_child_p=-6, # minimum local p threshold at which a child will be created
                 num_pv_expand=100,
                 search_params=None,
                 verbose=True,
                 validations=False):
        self.board = board

        self.policy_est = policy_est
        self.value_est = value_est

        self._verbose = verbose
        self.is_maximizing = True if board.get_player_to_move() == Player.FIRST else False
        self.max_iterations = max_iterations
        self.p_batch_size = p_batch_size
        self.fraction_q = fraction_q
        self.min_child_p = min_child_p
        self._validations = validations

        # search_params will override any default set parameters
        self.search_params = search_params
        if search_params:
            self.set_search_params(search_params)

        self.root_node = PExpNodeV3(parent=None, move=None, is_maximizing=self.is_maximizing)

        self.thought_board = ThoughtBoard(board)

        self.transformer = BoardTransform(board.get_size())

        # transposition hash, Node
        self.transposition_table = TranspositionTable()

        # all nodes at the leaves of the search tree
        #self.expandable_nodes = SortedList([self.root_node], key=lambda x: x.p_comparator)
        #self.expandable_set = set([self.root_node])
        self.expandable_nodes = {self.root_node}
        # we'll always be expanding the top k principal variations
        self._num_pv_expand = num_pv_expand
        self._top_all_p = [self.root_node]

        self.history = {}

    def set_search_params(self, params):
        for k, v in params.items():
            self.__setattr__(k, v)

    def get_pv(self):
        return self.root_node.get_principal_variation()

    def add_expand_nodes(self, nodes):
        self.expandable_nodes.update(nodes)
        if self._validations:
            for node in nodes:
                assert not node.game_over()
            self.history.update(nodes)

    def remove_expand_node(self, parent):
        if parent in self.expandable_nodes:
            self.expandable_nodes.remove(parent)

    def add_expand_node(self, node):
        assert not node.game_over()
        self.expandable_nodes.add(node)

    # makes no calls to models, just a board expansion
    def build_children(self, node):
        self.thought_board.make_moves(node.get_move_chain())

        # make a child for each available position
        for move in self.thought_board.available_moves():
            self.create_child(node, move)

        # remove the parent from expandable nodes
        self.expandable_nodes.remove(node)

        self.thought_board.reset()

    def create_child(self, parent, move):

        # build the hash of the transposition to check whether the position has been visited before
        transposition_hash = parent.get_transposition_hash_after(move)

        child = self.transposition_table.get(transposition_hash)
        # transposed position exists
        if child:
            # children can have multiple parents
            child.add_parent(parent, move)
        else:
            child = PExpNodeV3(parent=parent, move=move)
            self.transposition_table.put(transposition_hash, child)

            self.thought_board.make_move(move)
            if self._validations:
                assert child.depth() == self.thought_board.delta
            # if the game is over, mark it
            if self.thought_board.game_over():
                child.assign_leaf_q(self.thought_board.get_hard_q(), self.thought_board.get_game_status())
                # since we've updated q, we should recalculate it, since we're not going to recalculate for every parent
                # this is probably suboptimal here
                parent.recalculate_q()
            else:
                # this node can continue to expand
                self.add_expand_node(child)
            self.thought_board.unmove()

        # uh, does this have to happen on each iteration??
        parent.add_child(child, int(move))

        return child

    def compute_p(self, parents):
        print('Compute P', len(parents))
        p_features = []

        for node in parents:
            assert node.log_total_p != PExpNodeV3.UNASSIGNED_P
            assert node.game_status == GameState.NOT_OVER
            p_features.append(self.thought_board.get_p_features_after(node.get_move_chain()))

        # returns (len(parents), size**2) matrix
        log_p_predictions = np.log(self.policy_est.predict([p_features], batch_size=len(p_features)))

        # create and assign
        for prediction_set, parent in zip(log_p_predictions, parents):
            child_creation_vector = np.logical_and(prediction_set > self.min_child_p,
                                                   self.thought_board.get_available_move_vector_after(parent.get_move_chain()))
            self.thought_board.make_moves(parent.get_move_chain())
            for move in child_creation_vector.nonzero()[0]:
                child = self.create_child(parent, int(move))
                # may have already been assigned p if we're using transposition table
                if not child.is_assigned_p():
                    child.assign_p(prediction_set[move])

            self.thought_board.reset()
            # remove the parents from possible expanding nodes
            self.remove_expand_node(parent)

    def validate_whole_tree(self):
        assert self._validations
        self.validate_recursively(self.root_node)

    def validate_recursively(self, node):
        for child in node.get_children():
            child.integrity_check()
            self.validate_recursively(child)

    def pv_expansion(self, to_p_expand):
        # if we don't have anything to expand
        if len(to_p_expand) == 0:
            return []

        if to_p_expand[0].p_comparator > self._top_all_p[-1].p_comparator or len(self._top_all_p) < self._num_pv_expand:
            combined_top_p = list(set(self._top_all_p).union(set(to_p_expand[:self._num_pv_expand])))
            self._top_all_p = sorted(combined_top_p, key=lambda x: x.p_comparator)
            self._top_all_p = self._top_all_p[:max(len(self._top_all_p), 0)]

        top_pvs = set()
        for node in self._top_all_p:
            pv = node.get_principal_variation()
            # node has a pv, which is not already counted, and is not already going to be expanded
            # and if the pv does not point at a game end state
            if pv and pv not in top_pvs and pv not in to_p_expand and not pv.game_over():
                top_pvs.add(pv)

        if self._verbose:
            print('PV Expansion', len(top_pvs))
        return list(top_pvs)

    def p_expand(self):
        highest_p = self.highest_p(k=self.p_batch_size)
        top_pvs = self.pv_expansion(highest_p)
        self.compute_p(highest_p + top_pvs)
        return top_pvs

    def compute_q(self, candidates):
        # we compute only for nodes that haven't finished the game
        q_features = []
        # recalculate qs only on parents
        parents = set()
        original_candidates = len(candidates)
        assert len(candidates) == len(set(candidates))
        candidates = [node for node in candidates if not node.is_assigned_q()]

        q_nodes = []
        for node in candidates:
            parents.update(node.get_parents())
            self.thought_board.make_moves(node.get_move_chain())
            if not self.thought_board.game_over():
                q_features.append(self.thought_board.get_q_features())
                q_nodes.append(node)
            else:
                hard_q = self.thought_board.get_hard_q()
                node.assign_leaf_q(hard_q, GameState.WON if abs(hard_q) > 0 else GameState.DRAW)
            self.thought_board.reset()

        print('Candidate Q', original_candidates, 'Compute Q', len(q_features))

        # if we don't have any q's to expand after hard_q assessment
        if len(q_features) == 0:
            return

        q_predictions = np.clip(self.value_est.predict(np.array(q_features), batch_size=len(q_features)).reshape(-1),
                                a_min=PExpNodeV3.MIN_MODEL_Q, a_max=PExpNodeV3.MAX_MODEL_Q)

        assert len(q_predictions) == len(q_nodes)
        for i, node in enumerate(q_nodes):
            node.assign_leaf_q(q_predictions[i], GameState.NOT_OVER)

        for parent in parents:
            parent.recalculate_q()

    # this is necessary for learning data, regardless of root pv or whatnot
    def make_root_q(self):
        self.thought_board.reset()
        q_features = [self.thought_board.get_q_features()]
        root_q = np.clip(self.value_est.predict(np.array(q_features), batch_size=len(q_features)).reshape(-1),
                            a_min=PExpNodeV3.MIN_MODEL_Q, a_max=PExpNodeV3.MAX_MODEL_Q)[0]

        self.root_node.self_q = root_q

    def highest_p(self, k):
        return sorted(list(self.expandable_nodes), key=lambda x: x.p_comparator)[:k]
        #return [leaf for leaf in self.expandable_nodes.islice(0, k)]

    def q_eval(self):

        highest_p = self.highest_p(k=int(self.p_batch_size * self.fraction_q))

        to_eval = set(highest_p)

        # look at the top pv's and evaluate all of their children
        # since they've gone through a p expansion but don't necessarily have q's
        top_pvs = self.pv_expansion(highest_p)
        for top_pv in top_pvs:
            # should be a parameter
            to_eval.update(top_pv.get_children_highest_p(3))

        # we may not have anything to evaluate
        if len(to_eval) > 0:
            self.compute_q(to_eval)

    def validate_expandable_integrity(self):
        # make sure we're not expanding any game-over nodes
        if self._validations:
            for node in self.expandable_nodes:
                assert not self.thought_board.game_over_after(node.get_move_chain())

    def run_iteration(self):
        # no moves to evaluate
        if len(self.expandable_nodes) == 0:
            return

        self.p_expand()

        if self._verbose:
            print('Num Leaf Nodes', len(self.expandable_nodes), 'Transposition', self.transposition_table.get_num_hits())
        #self.validate_whole_tree()

        self.q_eval()

        if self._verbose:
            print('PV: ', self.root_node.get_principal_variation())

        # this is necessary for training data
        if self.root_node.self_q is None:
            self.make_root_q()

    # forces a split evenly among remaining moves
    def fill_null_pv(self):
        assert self.get_pv() is None
        self.thought_board.reset()
        available_moves = self.thought_board.available_moves()
        for move in available_moves:
            self.create_child(self.root_node, move)
            # state doesn't matter, this is just returning some random move
            self.root_node.get_child(move).assign_p(np.log(1.0 / len(available_moves)))
            self.root_node.get_child(move).assign_leaf_q(0, GameState.NOT_OVER)
        self.root_node.recalculate_q()

    def run(self, num_iterations=None):
        if num_iterations is None:
            num_iterations = self.max_iterations
        for i in range(num_iterations):
            self.run_iteration()

        # no moves to make according to p expansion
        if self.get_pv() is None:
            self.fill_null_pv()

        pv = self.get_pv()

        assert pv.get_q() is not None
        assert self.root_node.self_q is not None
        return self


class PExpMind_v3:
    def __init__(self, size, search_params, init=True,
                 verbose=True,
                 validation=False):

        self.size = size
        self.channels = FeatureSet_v1_1.CHANNELS

        assert size == 9
        self.value_est = self.value_model_9()
        self.policy_est = self.policy_model_9()

        # search parameters
        self.verbose = verbose
        self.validation = validation
        self.search_params = search_params

        # initialization with random examples so we can immediately predict
        init_examples = 10

        if init:
            sample_x = [np.random.randint(-1, 1, size=(init_examples, size, size, self.channels))]
            self.value_est.fit(sample_x, y=np.zeros(init_examples), epochs=1, batch_size=100)
            self.policy_est.fit(sample_x, y=np.zeros((init_examples, self.size ** 2)))

        self.fitted = False

        self.memory_root = None

    def value_model_9(self):
        inp = Input(shape=(self.size, self.size, self.channels))

        # key difference between this and conv network is padding
        conv_1 = Convolution2D(64, (3, 3), padding='valid', activation='relu',
                               kernel_initializer='random_normal', use_bias=False)(inp)
        bn2 = BatchNormalization()(conv_1)
        conv_2 = Convolution2D(32, (3, 3), padding='valid', activation='relu',
                               kernel_initializer='random_normal', use_bias=False)(bn2)
        bn3 = BatchNormalization()(conv_2)
        conv_3 = Convolution2D(16, (3, 3), padding='valid', activation='relu',
                               kernel_initializer='random_normal', use_bias=False)(bn3)
        bn4 = BatchNormalization()(conv_3)

        flat = Flatten()(bn4)

        hidden = Dense(20, activation='relu', kernel_initializer='random_normal', use_bias=False)(flat)
        bn_final = BatchNormalization()(hidden)

        out = Dense(1, use_bias=False)(bn_final)

        model = Model(inputs=[inp], outputs=out)
        model.compile(loss=losses.mean_squared_error, optimizer='adam', metrics=['mean_squared_error'])

        return model


    def policy_model_9(self):
        inp = Input(shape=(self.size, self.size, self.channels))

        # key difference between this and value network
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

    # p_threshold(depth, p) returns a boolean vector filter
    # max_expansion(depth) returns an integer
    def define_policies(self, p_threshold, max_expansion, convergence_count,
                        alpha=0.2, p_exp_batch_size=25, q_exp_batch_size=25,
                        required_depth=4, max_iters=20):
        self._p_threshold = p_threshold
        self._max_expansion = max_expansion
        self._move_convergence_count = convergence_count

        self.p_exp_batch_size = p_exp_batch_size
        self.q_exp_batch_size = q_exp_batch_size
        self.alpha = alpha
        self.required_depth = required_depth
        self.max_iters = max_iters


    def pick_random_move(self, board, possible_moves):
        picked_action = 0

        # abs is only there to handle floating point problems
        qs = np.array([node.q for _, node in possible_moves])
        if board.get_player_to_move() == 1:
            distribution = np.abs(qs + 1.0) / 2
        else:
            # not sure this is correct
            distribution = sorted(-np.abs(qs - 1.0) / 2)

        if sum(distribution) > 0:
            distribution = (distribution.astype(np.float64) / sum(distribution))
            picked_action = np.random.choice(range(len(possible_moves)), 1, p=distribution)[0]

        return picked_action

    def make_search(self, board):
        return PEvenSearch(board, self.policy_est, self.value_est,
                               verbose=self.verbose, validations=self.validation,
                                search_params=self.search_params,
                                num_pv_expand=1000)

    def make_move(self, board, searcher=None):
        if searcher is None:
            searcher = self.make_search(board)
        searcher.run()
        pv = searcher.get_pv()

        return pv.calculate_pv_order()[0], searcher.root_node.self_q, pv.get_q()

    def pickle_root(self, board, root_node):
        # save for debugging
        import sys
        sys.setrecursionlimit(100000)
        saving_hash = str(hash(tuple(board.get_matrix().reshape(-1))))
        with open('qgomoku/logs/' + saving_hash + '.pkl', 'wb') as f:
            root_node.set_matrix(board.get_matrix())
            pickle.dump(root_node, f)
            print('Root saved at ', saving_hash)


    def save(self, filename):
        self.value_est.save(filename + '_value.net')
        self.policy_est.save(filename + '_policy.net')

    def load_net(self, filename):
        self.value_est = keras.models.load_model(filename + '_value.net')
        self.policy_est = keras.models.load_model(filename + '_policy.net')
       
    def load(self, value_file, policy_file):
        self.value_est = keras.models.load_model(value_file)
        self.policy_est = keras.models.load_model(policy_file)