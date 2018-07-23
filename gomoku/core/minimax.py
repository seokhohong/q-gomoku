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


class TreeNode:
    MIN_Q = -1
    MAX_Q = 1
    # is maximizing?
    def __init__(self, parent, is_maximizing, full_move_list):
        # this node's q
        self.q = 0
        # all known q values that are one move from this state
        # dict[TreeNode]: q (float)
        self.local_q_table = {}
        # parent TreeNode
        self.parent = parent
        # full move tree
        self.full_move_list = full_move_list
        # is this a maximizing node for our player? (minimizing for opponent)
        assert (type(is_maximizing) == bool)
        self.is_maximizing = is_maximizing
        # is the game over with this move list
        self.game_status = GameState.NOT_OVER
        # for debugging
        # key is move
        self.children = {}
        self.max_depth = 0

    # used for identifying which nodes are the most important to go after
    def get_q_importance(self):
        if self.parent.is_maximizing:
            return self.q
        else:
            return -self.q

    def get_sorted_moves(self):
        return sorted(list(self.local_q_table.items()), key=lambda x: x[1], reverse=True)

    def set_max_depth(self, depth):
        self.max_depth = max(self.max_depth, depth)
        if self.parent:
            self.parent.set_max_depth(depth)

    # note this does NOT compute q for child
    def create_child(self, move):
        child = TreeNode(parent=self,
                                is_maximizing=not self.is_maximizing,
                                full_move_list=self.full_move_list.append(move))

        if len(self.children) == 0:
            child.max_depth = self.max_depth + 1
            self.max_depth = child.max_depth
        else:
            child.max_depth = self.max_depth
            self.set_max_depth(max([this_child.max_depth for this_child in self.children.values()]))

        self.children[move] = child
        return child

    def __str__(self):
        return str(self.full_move_list.moves) + " Q: " + str(self.q) + " Max Depth: " + str(self.max_depth)

class BFNode(TreeNode):
    def __init__(self, *args, **kwargs):
        super(BFNode, self).__init__(*args, **kwargs)

    # used ONLY for leaf assignment
    def assign_q(self, q, game_status):
        assert (len(self.children) == 0)
        self.q = q

        # mark game status
        self.game_status = game_status

        # recursively update
        if self.parent:
            # note that the move that produced this node gives this q
            self.parent.local_q_table[self] = self.q
            self.parent.recalculate_q()

    # minimax q assessment
    def recalculate_q(self):
        # take the largest (or smallest) q across all seen moves
        prev_q = self.q
        if self.is_maximizing:
            self.q = max(self.local_q_table.values())
        else:
            self.q = min(self.local_q_table.values())

        if self.parent and prev_q != self.q:
            self.parent.local_q_table[self] = self.q
            self.parent.recalculate_q()

# principle value search
class PVSNode(TreeNode):

    def __init__(self, *args, **kwargs):
        super(PVSNode, self).__init__(*args, **kwargs)
        # tuple of PVSNode, q
        self.principle_variation = None
        self.q = None

        # log likelihood of playing this move given parent state
        self.log_local_p = 0
        # log likelihood of playing this move given root board state (used for PVS search)
        self.log_total_p = 0

    def find_pvs(self):
        # found child
        if self.has_children():
            return self.full_move_list, self.q
        return

    def has_children(self):
        return len(self.children) > 0

    # note this does NOT compute q for child
    def create_child(self, move):
        child = PVSNode(parent=self,
                        is_maximizing=not self.is_maximizing,
                        full_move_list=self.full_move_list.append(move))

        if len(self.children) == 0:
            child.max_depth = self.max_depth + 1
            self.max_depth = child.max_depth
        else:
            child.max_depth = self.max_depth
            self.set_max_depth(max([this_child.max_depth for this_child in self.children.values()]))

        self.children[move] = child
        return child

    # play shorter sequences if advantageous, otherwise play longer sequences
    def move_goodness(self):
        if self.q > 0:
            return self.q - len(self.principle_variation.full_move_list) * 0.001
        else:
            return self.q + len(self.principle_variation.full_move_list) * 0.001

    def get_sorted_moves(self):
        return sorted(self.children.items(), key= lambda x : x[1].move_goodness(), reverse=self.is_maximizing)

    def get_k_principle_variations(self, leaf_nodes, k=5):
        # include the best move according to q
        q_best = self.principle_variation
        # draw
        if q_best is None:
            return []
        # include k-1 most likely moves according to p
        candidates = [node for node in list(leaf_nodes) if node.game_status == GameState.NOT_OVER
                                                                    and node != q_best]
        if q_best.game_status == GameState.NOT_OVER:
            return [q_best] + sorted(candidates, key=lambda x: x.log_total_p, reverse=True)[:k - 1]
        else:
            return sorted(candidates, key=lambda x: x.log_total_p, reverse=True)[:k]

    # used ONLY for leaf assignment
    def assign_q(self, q, game_status):
        assert (len(self.children) == 0)

        # mark game status
        self.game_status = game_status

        # for a leaf node, the principle variation is itself
        self.principle_variation = self
        self.q = q

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
            best_move = max(self.children.items(), key=lambda x : x[1].move_goodness())[0]
        else:
            best_move = min(self.children.items(), key=lambda x : x[1].move_goodness())[0]

        # using negamax framework
        self.principle_variation = self.children[best_move].principle_variation
        self.q = self.children[best_move].q

        if self.parent and abs(prev_q - self.q) > 1E-6:
            # update pvs for parent
            self.parent.recalculate_q()

    def __str__(self):
        return ("PV: " + str(self.principle_variation.full_move_list.moves) + " Q: {0:.4f} P: {1:.4f} Max Depth: " + str(self.max_depth)).format(self.q, self.log_total_p)
