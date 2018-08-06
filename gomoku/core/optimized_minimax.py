import numpy as np

from core.board import GameState


class MoveList:
    # moves should be a tuple
    def __init__(self, moves):
        self.moves = moves
        # for move in moves:
        #    assert(len(move) == 2)

    def append(self, new_move):
        return MoveList((self.moves + (new_move,)))

    def __eq__(self, other):
        return self.moves == other.moves

    def __hash__(self):
        return hash(self.moves)

    def __len__(self):
        return len(self.moves)

    def transposition_hash(self):
        return tuple(sorted([(i % 2, move) for i, move in enumerate(self.moves)]))

# P expansion node
class PExpNode:
    MAX_Q = 1
    # largest Q allowed by model prediction (MAX_Q is a minimax certified win)
    MAX_MODEL_Q = 0.99
    MIN_Q = -1
    MIN_MODEL_Q = -0.99
    UNASSIGNED_Q = None

    def __init__(self, parent, is_maximizing, full_move_list):

        # parent TreeNode
        self.is_root = parent is None

        # with transposition, can have multiple parents
        self.parents = [parent]

        # full move tree
        self.full_move_list = full_move_list

        self.is_maximizing = is_maximizing
        # is the game over with this move list
        self.game_status = GameState.NOT_OVER
        # for debugging
        # key is move
        self.children = {}

        self.principal_variation = None
        self.q = PExpNode.UNASSIGNED_Q
        self.assigned_q = False

        # log likelihood of playing this move given parent state
        self.log_local_p = 0
        # log likelihood of playing this move given root board state (used for PVS search)
        self.log_total_p = 0
        # log_total_p converted to integer that's easy to sort
        self.p_comparator = 0

        # only allowed to set once
        self.p_assigned = False

        # the quality of this move
        self.move_goodness = 0

        self._matrix = None

    def set_matrix(self, matrix):
        self._matrix = matrix

    def get_matrix(self):
        return self._matrix

    def has_children(self):
        return len(self.children) > 0

    # def __lt__(self, other):
    #     return self.log_total_p > other.log_total_p
    #
    # def __gt__(self, other):
    #     return self.log_total_p < other.log_total_p


    transposition_access = 0

    # note this does NOT compute q for child
    def create_child(self, move, transposition_table):
        new_move_list = self.full_move_list.append(move)

        transposition_hash = new_move_list.transposition_hash()
        child_found = transposition_hash in transposition_table
        child = None
        if child_found:
            PExpNode.transposition_access += 1
            child = transposition_table[transposition_hash]
            child.parents.append(self)
        else:
            child = PExpNode(parent=self,
                            is_maximizing=not self.is_maximizing,
                            full_move_list=new_move_list)
            transposition_table[transposition_hash] = child

        # only build if not in transposition (trying otherwise)
        self.children[move] = child
        return child, not child_found

    def get_sorted_moves(self):
        valid_children = [tup for tup in self.children.items() if tup[1].q is not PExpNode.UNASSIGNED_Q]
        return sorted(valid_children, key=lambda x: x[1].move_goodness, reverse=self.is_maximizing)

    # used ONLY for leaf assignment
    def assign_q(self, q, game_status):
        assert (len(self.children) == 0)

        # mark game status
        self.game_status = game_status

        # for a leaf node, the principal variation is itself
        self.q = q
        assert(q <= PExpNode.MAX_Q and q >= PExpNode.MIN_Q)
        self.principal_variation = self
        assert(not self.assigned_q)
        self.assigned_q = True

        #self.recalculate_q()

        self.assign_move_goodness()

    def assign_move_goodness(self):
        # play shorter sequences if advantageous, otherwise play longer sequences
        if self.q > 0:
            self.move_goodness = self.q - len(self.principal_variation.full_move_list) * 1E-6
        else:
            self.move_goodness = self.q + len(self.principal_variation.full_move_list) * 1E-6

    # ab cutoff for when we know we can
    # Not sure if it applies regardless of parent
    def ab_valid(self, epsilon=1E-7):
        if not self.is_root and self.parents[0].q:
            return self.parents[0].is_maximizing and self.parents[0].q < PExpNode.MAX_Q \
                    or not self.parents[0].is_maximizing and self.parents[0].q > PExpNode.MIN_Q
        return True

    def assign_p(self, log_p):
        assert(self.log_local_p >= 0)
        assert(not self.p_assigned)
        self.log_local_p = log_p
        self.log_total_p = self.parents[0].log_total_p + self.log_local_p
        self.p_assigned = True
        self.p_comparator = int((min(-self.log_total_p, 100)) * 1E6)

    def recalculate_q(self, verbose=False):
        # take the largest (or smallest) q across all seen moves

        # if this node is still a leaf, break
        if len(self.children) == 0:
            return

        valid_children = [tup for tup in self.children.items() if tup[1].assigned_q]

        if len(valid_children) == 0:
            return

        if self.is_maximizing:
            best_move = max(valid_children, key=lambda x: x[1].move_goodness)[0]
        else:
            best_move = min(valid_children, key=lambda x: x[1].move_goodness)[0]

        if verbose:
            print('Recalculate Q for', self.full_move_list.moves)
            print('Previous', self.principal_variation)

        # using negamax framework
        self.principal_variation = self.children[best_move].principal_variation

        if verbose:
            print("To", self.children[best_move].principal_variation)

        self.q = self.children[best_move].q

        self.assign_move_goodness()

        if not self.is_root:
            # update pvs for parent
            for parent in self.parents:
                parent.recalculate_q(verbose=verbose)

    # DEBUGGING METHODS
    def recursive_children(self, include_game_end=False):
        leaves = []
        for child in self.children.values():
            if child.has_children():
                leaves.extend(child.recursive_children())
            else:
                if not include_game_end and child.game_status == GameState.NOT_OVER\
                        or include_game_end:
                    leaves.append(child)
        return leaves

    # Assert consistency
    def consistent_pv(self):
        if self.has_children():
            valid_children = [tup for tup in self.children.items() if tup[1].q is not PExpNode.UNASSIGNED_Q]
            if len(valid_children) == 0:
                return
            if self.is_maximizing:
                best_move = max(valid_children, key=lambda x: x[1].move_goodness)[0]
            else:
                best_move = min(valid_children, key=lambda x: x[1].move_goodness)[0]

            # using negamax framework
            if self.principal_variation != self.children[best_move].principal_variation:
                self.recalculate_q()
            assert(self.principal_variation.log_total_p < 0)
            assert (self.principal_variation.log_local_p < 0)

            for child in self.children.values():
                child.consistent_pv()

    def top_down_q(self):
        valid_children = [tup for tup in self.children.items() if tup[1].q]
        if len(valid_children) == 0:
            return

        for _, child in valid_children:
            child.top_down_q()

        if self.is_maximizing:
            best_move = max(valid_children, key=lambda x: x[1].move_goodness)[0]
        else:
            best_move = min(valid_children, key=lambda x: x[1].move_goodness)[0]

        self.principal_variation = self.children[best_move].principal_variation
        self.q = self.children[best_move].q

    def negamax(self):
        valid_children = [tup for tup in self.children.items() if tup[1].q]
        if len(valid_children) == 0:
            return self.q, self

        best_child = None
        if self.is_maximizing:
            value = -np.inf
            for move, child in valid_children:
                child_value, next_child = child.negamax()
                assert(abs(child_value - next_child.principal_variation.q) < 1E-4)
                if child_value > value:
                    best_child = next_child
                    value = child_value
        else:
            value = np.inf
            for move, child in valid_children:
                child_value, next_child = child.negamax()
                assert (abs(child_value - next_child.principal_variation.q) < 1E-4)
                if child_value < value:
                    best_child = next_child
                    value = child_value

        return value, best_child

    @classmethod
    # whether the q is from a game result (as opposed to an approximation of game state)
    def is_result_q(cls, q, epsilon=1E-7):
        return abs(q) > PExpNode.MAX_Q - epsilon or abs(q) < epsilon

    def __str__(self):
        if self.principal_variation:
            return ("PV: " + str(self.principal_variation.full_move_list.moves) + " Q: {0:.4f} P: {1:.4f}").format(self.q, self.log_total_p)
        else:
            return "Position: " + str(self.full_move_list.moves) + " P: {0:.4f}".format(self.log_total_p)