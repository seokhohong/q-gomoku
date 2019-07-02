import numpy as np

from qgomoku.core.board import GameState, BoardTransform
from qgomoku.core.minimax import MoveList


# search tree where we search strictly according to the likelihood function
class PExpNodeV3:
    MAX_Q = 1.0
    # largest Q allowed by model prediction (MAX_Q is a minimax certified win)
    MAX_MODEL_Q = 1.0 - 1E-4
    MIN_Q = -MAX_Q
    MIN_MODEL_Q = -MAX_MODEL_Q
    UNASSIGNED_Q = None
    UNASSIGNED_P = None

    # if parent is None, then we assume the node is root
    # we assume that maximizing is true to make it simpler to create child nodes
    def __init__(self, parent, move, is_maximizing=True):

        # log of probability of playing this move given root board state (used for PVS search)
        self.log_total_p = 0 if parent is None else PExpNodeV3.UNASSIGNED_P
        self._principal_variation = None

        self.self_q = PExpNodeV3.UNASSIGNED_Q

        # with transposition, can have multiple parents
        self._parents = set()
        if parent:
            self.add_parent(parent, move)

        # add the newest move made
        if parent is None:
            self._full_move_list = MoveList(moves=(), position_hash=[])
            self.is_maximizing = is_maximizing
        else:
            assert type(parent) == PExpNodeV3
            self._full_move_list = parent._full_move_list.append(move)
            self.is_maximizing = not parent.is_maximizing

        # is the game over with this move list
        self.game_status = GameState.NOT_OVER
        # for debugging
        # key is move index (just an integer)
        self._children = {}

        self.children_with_q = []
        # current best child
        self.best_child = None

        # log_total_p converted to integer that's easy to sort
        self.p_comparator = 0

        self._p_features = None
        self._q_features = None

        self._move_goodness = None

    def set_q_features(self, q_features):
        self._q_features = q_features

    def set_p_features(self, p_features):
        self._p_features = p_features

    def get_q_features(self):
        return self._q_features

    def get_p_features(self):
        return self._p_features

    def has_q_features(self):
        return self._q_features is not None

    def has_p_features(self):
        return self._p_features is not None

    def has_children(self):
        return len(self._children) > 0

    def get_move_relationship(self, parent):
        assert len(parent.get_children()) > 0
        for move, child in parent.get_children_dict().items():
            if self == child:
                return move
        raise ValueError('Node not in child list of parents')

    def has_child(self, move):
        assert type(move) == int
        return move in self._children

    def get_child(self, move):
        return self._children[move]

    def get_children(self):
        return self._children.values()

    def get_children_highest_p(self, k):
        top_children = sorted(list(self._children.values()), key=lambda x: x.p_comparator)
        return top_children[:min(len(top_children), k)]

    def get_children_moves(self):
        return self._children.keys()

    def get_children_dict(self):
        return dict(self._children)

    def get_principal_variation(self):
        return self._principal_variation

    def has_parents(self):
        return len(self._parents) > 0

    def add_parent(self, parent, move):
        assert type(parent) == PExpNodeV3
        self._parents.add(parent)
        parent._children[move] = self
        self.add_child_to_parent_qs(parent)

    def first_parent(self):
        return next(iter(self._parents))

    def get_parents(self):
        for parent in self._parents:
            yield parent

    def add_child(self, child, move):
        assert type(move) == int
        child.add_parent(self, move)

    def get_sorted_moves(self):
        valid_children = [tup for tup in self._children.items() if tup[1].q is not PExpNodeV3.UNASSIGNED_Q]
        return sorted(valid_children, key=lambda x: x[1].move_goodness(), reverse=self.is_maximizing)

    def get_q(self):
        if self.get_principal_variation() == self:
            return self.self_q
        return self.get_principal_variation().get_q()

    def integrity_check(self):
        for child in self._children.values():
            for parent in child._parents:
                if child.is_assigned_q() and child not in parent.children_with_q \
                        or not child.is_assigned_q() and child in parent.children_with_q:
                    print('why')
                assert child.is_assigned_q() and child in parent.children_with_q
                assert not child.is_assigned_q() and child not in parent.children_with_q
            assert len(self.children_with_q) == len(set(self.children_with_q))

    # used ONLY for leaf assignment
    def assign_leaf_q(self, q, game_status):
        assert not self.is_assigned_q()

        # self_q is the value evaluated by the network
        self.self_q = q

        self.game_status = game_status
        # for a leaf node, the principal variation is itself
        self._update_pv(self)

    def game_over(self):
        return self.game_status != GameState.NOT_OVER

    def add_child_to_parent_qs(self, parent):
        if self.is_assigned_q() and self not in parent.children_with_q:
            parent.children_with_q.append(self)

    def _update_pv(self, pv):
        self._principal_variation = pv
        assert (PExpNodeV3.MIN_Q <= pv.get_q() <= PExpNodeV3.MAX_Q)

        # note to parent that we have a q in this leaf
        for parent in self._parents:
            self.add_child_to_parent_qs(parent)

        self.compute_move_goodness()
        # we could recalculate q here, but we optimize to actually recalculate only at parent level

    def is_assigned_p(self):
        return self.log_total_p != PExpNodeV3.UNASSIGNED_P

    def get_p_comparator(self):
        return self.p_comparator

    def assign_p(self, self_log_p):
        self.log_total_p = self.first_parent().log_total_p + self_log_p

        # needs an integer-based sorting mechanism
        self.p_comparator = int((min(-self.log_total_p, 100)) * 1E6)

    def get_move_chain(self):
        return self._full_move_list.moves

    def get_transposition_hash(self):
        return self._full_move_list.transposition_hash()

    def get_transposition_hash_after(self, with_move):
        return self._full_move_list.append(with_move).transposition_hash()

    def depth(self):
        return len(self._full_move_list.moves)

    # returns if this node is better than another, comparing Q value
    def better_q(self, other):
        return self._move_goodness > other._move_goodness

    def compute_move_goodness(self):
        # if positive, we dislike length
        if self.get_q() > 0:
            length_penalty = -1E-6 * self._principal_variation.depth()
        else:
            length_penalty = 1E-6 * self._principal_variation.depth()
        self._move_goodness = self.get_q() + length_penalty

    def update_best_child(self):
        # we are going to have to frequently search for the max, so might as well sort while we're at it
        self.children_with_q.sort(key=lambda x: x._move_goodness, reverse=self.is_maximizing)
        self.best_child = self.children_with_q[0]

    # Computes a bottom-up recomputation of Q
    def recalculate_q(self):
        # this should only be called on a parent node
        assert len(self.children_with_q) > 0

        self.update_best_child()

        self._update_pv(self.best_child.get_principal_variation())

        # update parents
        for parent in self._parents:
            # if this node is in the PV line
            if parent.best_child == self or parent.best_child is None or (
                    (parent.is_maximizing and self.better_q(parent.best_child)) or
                    (not parent.is_maximizing and not self.better_q(parent.best_child))
            ):
                parent.recalculate_q()

    # because with transpositions, it is not guaranteed that the pv's move ordering is the most likely one
    def calculate_pv_order(self):
        chain = []
        current = self
        while current and current.has_parents():
            largest_p = -np.inf
            best_move = None
            best_parent = None
            for parent in current.get_parents():
                if parent.log_total_p > largest_p:
                    best_move = current.get_move_relationship(parent)
                    best_parent = parent
                    largest_p = parent.log_total_p
            chain.append(best_move)
            current = best_parent
        chain.reverse()
        return chain

    def has_self_q(self):
        return self.self_q is not None

    def is_assigned_q(self):
        return self._principal_variation is not None

    def __str__(self):
        transformer = BoardTransform(size=9)

        if self._full_move_list.moves:
            coord_moves = ', '.join([str(transformer.index_to_coordinate(move)) for move in self.calculate_pv_order()])
        else:
            coord_moves = '(ROOT)'
        if self._principal_variation:
            # super bad code
            return ("PV: " + coord_moves + " Q: {0:.4f} P: {1:.4f}").format(self.get_q(),
                                                                            self._principal_variation.log_total_p)
        elif self.log_total_p:
            return "Position: " + coord_moves + " P: {0:.4f}".format(self.log_total_p)
        else:
            return "Position: " + coord_moves
