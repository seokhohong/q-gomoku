
import unittest
from qgomoku.learner.pexp_mind_v3 import PExpMind_v3, PEvenSearch, ThoughtBoard
from qgomoku.learner.pexp_node_v3 import PExpNodeV3
from qgomoku.core.board import Board, BoardTransform, GameState
from qgomoku.learner.game_to_features import FeatureBoard_v1_1
from qgomoku.core.minimax import MoveList

import numpy as np


class TestMind(unittest.TestCase):

    def test_move_list(self):
        base = MoveList(moves=(), position_hash=[])
        base.append(50)
        base.append(40)
        base.append(30)
        base.append(20)

        base_2 = MoveList(moves=(), position_hash=[])
        base_2.append(30)
        base_2.append(20)
        base_2.append(50)
        base_2.append(40)

        self.assertEqual(base.transposition_hash(), base_2.transposition_hash())


    def test_shorter_win(self):
        base = PExpNodeV3(parent=None, move=15)
        base.assign_leaf_q(0.5, GameState.NOT_OVER)

        long_win = PExpNodeV3(base, 16)
        long_win.assign_leaf_q(1.0, GameState.WON)
        long_win.assign_p(-1)
        base.recalculate_q()
        self.assertEqual(base.get_principal_variation(), long_win)
        # check move_goodness looks at length too
        self.assertLess(long_win._move_goodness, long_win.get_q())

        long_win_2 = PExpNodeV3(long_win, 17)
        long_win_2.assign_leaf_q(1.0, GameState.WON)
        long_win_2.assign_p(-0.2)
        long_win.recalculate_q()
        self.assertEqual(base.get_principal_variation(), long_win_2)

        long_win_3 = PExpNodeV3(long_win_2, 18)
        long_win_3.assign_leaf_q(1.0, GameState.WON)
        long_win_3.assign_p(-0.5)
        long_win_2.recalculate_q()
        self.assertEqual(base.get_principal_variation(), long_win_3)

        short_loss = PExpNodeV3(base, 19)
        short_loss.assign_leaf_q(-1.0, GameState.WON)
        short_loss.assign_p(-0.8)
        base.recalculate_q()
        self.assertEqual(base.get_principal_variation(), long_win_3)
        self.assertEqual(list(base.get_principal_variation().get_move_chain()),
                         base.get_principal_variation().calculate_pv_order())

        short_win = PExpNodeV3(base, 20)
        short_win.assign_leaf_q(1.0, GameState.WON)
        base.recalculate_q()
        self.assertEqual(base.get_principal_variation(), short_win)

        self.assertEqual(base.best_child, short_win)
        self.assertEqual(list(base.get_principal_variation().get_move_chain()),
                         base.get_principal_variation().calculate_pv_order())

    def test_longer_loss(self):
        base = PExpNodeV3(parent=None, move=None, is_maximizing=False)
        base.assign_leaf_q(0.5, GameState.NOT_OVER)

        short_loss = PExpNodeV3(base, 20)
        short_loss.assign_leaf_q(1.0, GameState.WON)
        base.recalculate_q()
        self.assertEqual(base.get_principal_variation(), short_loss)

        long_loss = PExpNodeV3(base, 16)
        long_loss.assign_leaf_q(1.0, GameState.WON)
        base.recalculate_q()
        self.assertLess(long_loss._move_goodness, long_loss.get_q())

        long_loss_2 = PExpNodeV3(long_loss, 17)
        long_loss_2.assign_leaf_q(1.0, GameState.WON)
        long_loss.recalculate_q()
        self.assertEqual(base.get_principal_variation(), long_loss_2)

        long_loss_3 = PExpNodeV3(long_loss_2, 18)
        long_loss_3.assign_leaf_q(1.0, GameState.WON)
        long_loss_2.recalculate_q()
        self.assertEqual(base.get_principal_variation(), long_loss_3)

        self.assertEqual(base.best_child, long_loss)

    def test_add_transposition_parent(self):
        base = PExpNodeV3(parent=None, move=None, is_maximizing=False)
        base.assign_leaf_q(0.5, GameState.NOT_OVER)

        left = PExpNodeV3(parent=base, move=15)
        base.add_child(left, 45)
        left.assign_leaf_q(0.6, GameState.NOT_OVER)
        base.recalculate_q()

        right = PExpNodeV3(parent=base, move=15)
        base.add_child(right, 45)
        right.assign_leaf_q(0.4, GameState.NOT_OVER)
        base.recalculate_q()
        self.assertEqual(base.get_principal_variation(), right)
        self.assertEqual(base.get_q(), 0.4)

        # add parent instead of add child, like it would happen if transposed
        down = PExpNodeV3(parent=base, move=15)
        down.add_parent(right, 39)
        down.assign_leaf_q(0.5, GameState.NOT_OVER)
        right.recalculate_q()

        self.assertEqual(base.get_q(), 0.5)

    # we don't use this right now
    @unittest.skip("don't use this right now")
    def test_p_propagation(self):
        # Tree (NOTE this doesn't actually cause transposition in game since moves are by different players)
        #  base
        # / \
        # l r
        # \ /
        #  d
        #  |
        #  p
        null_ml = MoveList(moves=(), position_hash=[])
        base = PExpNodeV3(parent=None, move=4,
                          is_maximizing=True)

        left = PExpNodeV3(parent=base, is_maximizing=False, full_move_list=null_ml)
        base.add_child(left, 45)
        left.assign_p(-2)
        self.assertAlmostEqual(left.log_total_p, -2)

        down = PExpNodeV3(parent=None, is_maximizing=False, full_move_list=null_ml)
        left.add_child(down, 16)
        down.assign_p(-3)

        self.assertAlmostEqual(down.log_total_p, -5)

        # add one more below
        below = PExpNodeV3(parent=down, is_maximizing=False, full_move_list=null_ml)
        below.assign_p(-4)
        down.add_child(below, 16)

        self.assertAlmostEqual(below.log_total_p, -9)

        right = PExpNodeV3(parent=base, is_maximizing=False, full_move_list=null_ml)
        base.add_child(right, 16)
        right.assign_p(-1)
        self.assertAlmostEqual(right.log_total_p, -1)

        # make down a child of right all of a sudden (mimicking transposition table)
        right.add_child(down, 45)
        down.assign_p(-2.5)

        # we're not doing precision p computes...
        # should add probabilities
        # self.assertAlmostEqual(down.log_total_p, PExpNodeV3.add_log_probabilities(-3.5, -5))

        # should propagate probabilities below
        # self.assertAlmostEqual(below.log_total_p, PExpNodeV3.add_log_probabilities(-3.5, -5) - 4)


if __name__ == '__main__':
    unittest.main()
    # TestMind().test_longer_loss()


