import numpy as np

from gomoku.core.board import GameState


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

# principle value search
class PVSNode:
    MAX_Q = 1
    MIN_Q = -1

    def __init__(self, parent, is_maximizing, full_move_list):

        # parent TreeNode
        self.parent = parent
        # full move tree
        self.full_move_list = full_move_list

        self.is_maximizing = is_maximizing
        # is the game over with this move list
        self.game_status = GameState.NOT_OVER
        # for debugging
        # key is move
        self.children = {}

        # tuple of PVSNode, q
        self.principle_variation = None
        self.q = None

        # log likelihood of playing this move given parent state
        self.log_local_p = 0
        # log likelihood of playing this move given root board state (used for PVS search)
        self.log_total_p = 0

        # the quality of this move
        self.move_goodness = 0

    def has_children(self):
        return len(self.children) > 0

    # note this does NOT compute q for child
    def create_child(self, move):
        child = PVSNode(parent=self,
                        is_maximizing=not self.is_maximizing,
                        full_move_list=self.full_move_list.append(move))
        self.children[move] = child
        return child

    def get_sorted_moves(self):
        return sorted(self.children.items(), key=lambda x: x[1].move_goodness, reverse=self.is_maximizing)

    # used ONLY for leaf assignment
    def assign_q(self, q, game_status):
        assert (len(self.children) == 0)

        # mark game status
        self.game_status = game_status

        # for a leaf node, the principle variation is itself
        self.principle_variation = self
        self.q = q

        self.assign_move_goodness()

    def assign_move_goodness(self):
        # play shorter sequences if advantageous, otherwise play longer sequences
        if self.q > 0:
            self.move_goodness = self.q - len(self.principle_variation.full_move_list) * 0.001
        else:
            self.move_goodness = self.q + len(self.principle_variation.full_move_list) * 0.001

    def assign_p(self, log_p):
        self.log_local_p = log_p
        self.log_total_p = self.parent.log_total_p + self.log_local_p

    def recalculate_q(self):
        # take the largest (or smallest) q across all seen moves

        # if this node is still a leaf, break
        if len(self.children) == 0:
            return

        if self.principle_variation:
            prev_q = self.q
        else:
            prev_q = np.inf

        if self.is_maximizing:
            best_move = max(self.children.items(), key=lambda x : x[1].move_goodness)[0]
        else:
            best_move = min(self.children.items(), key=lambda x : x[1].move_goodness)[0]

        # using negamax framework
        self.principle_variation = self.children[best_move].principle_variation
        self.q = self.children[best_move].q

        self.assign_move_goodness()

        if self.parent and abs(prev_q - self.q) > 1E-6:
            # update pvs for parent
            self.parent.recalculate_q()

    def __str__(self):
        if self.principle_variation:
            return ("PV: " + str(self.principle_variation.full_move_list.moves) + " Q: {0:.4f} P: {1:.4f}").format(self.q, self.log_total_p)
        else:
            return "Position: " + str(self.full_move_list.moves)
