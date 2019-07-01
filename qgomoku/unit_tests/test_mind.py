
import unittest
from qgomoku.learner.pexp_mind_v3 import PExpMind_v3, PEvenSearch, ThoughtBoard
from qgomoku.learner.pexp_node_v3 import PExpNodeV3
from qgomoku.core.board import Board, BoardTransform, GameState, BitBoardCache, BitBoard
from qgomoku.learner.game_to_features import FeatureBoard_v1_1
from qgomoku.core.minimax import MoveList

import numpy as np

import os
# bug with python on macos
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class TestMind(unittest.TestCase):

    def test_thoughtboard_consistency(self):
        board = Board(size=9, win_chain_length=5)
        fb = FeatureBoard_v1_1(board)
        tb = ThoughtBoard(board, FeatureBoard_v1_1)

        fb.move(15)

        movelist = MoveList((), []).append(15)

        self.assertTrue(np.equal(fb.get_p_features(), tb.get_p_features_after(movelist)).all())

        movelist_2 = movelist.append(32)
        fb.move(32)
        self.assertTrue(np.equal(fb.get_p_features(), tb.get_p_features_after(movelist_2)).all())

        fb.unmove()
        self.assertTrue(np.equal(fb.get_p_features(), tb.get_p_features_after(movelist)).all())

        fb.move(32)
        self.assertTrue(np.equal(fb.get_p_features(), tb.get_p_features_after(movelist_2)).all())

        tb.make_move(15)
        tb.make_move(32)
        self.assertTrue(np.equal(fb.get_p_features(), tb.get_q_features()).all())

    def test_move_available_vector(self):
        board = Board(size=9, win_chain_length=5)
        fb = FeatureBoard_v1_1(board)
        tb = ThoughtBoard(board, FeatureBoard_v1_1)
        self.assertEqual(np.sum(tb.get_available_move_vector_after([34, 45])), 79)

        fb.move(15)
        self.assertEqual(np.sum(fb.get_init_available_move_vector()), 81)

        self.assertEqual(np.sum(tb.get_available_move_vector_after([18, 25])), 79)

    def test_thoughtboard_root_win(self):
        trivial_board = Board(size=9, win_chain_length=5)
        trivial_board.set_to_one_move_from_win()
        tb = ThoughtBoard(trivial_board, FeatureBoard_v1_1)

        trivial_board.move(36)
        print(trivial_board.pprint())
        move_list = MoveList((), []).append(36)
        self.assertTrue(trivial_board.game_won())
        self.assertTrue(tb.game_over_after(move_list))

    def test_even_search(self):
        mind = PExpMind_v3(size=9, init=False, search_params=None)
        mind.load_net('../../models/voldmaster_' + str(0))

        board = Board(size=9, win_chain_length=5)

        searcher = PEvenSearch(board, mind.policy_est, mind.value_est, max_iterations=2,
                    p_batch_size=1024, verbose=True, validations=True)

        searcher.p_expand()
        self.assertGreater(len(searcher.expandable_nodes), 0)
        for node in searcher.expandable_nodes:
            self.assertEqual(list(node.get_parents())[0], searcher.root_node)
            self.assertLess(node.log_total_p, 0)

        searcher.q_eval()
        self.assertNotEqual(searcher.root_node.get_principal_variation(), None)

    @unittest.skip('Timeconsuming')
    def test_game_closure(self):
        mind = PExpMind_v3(size=9, init=False)
        mind.load_net('../../models/voldmaster_' + str(0))

        cache = BitBoardCache("../cache/9-magics", size=9, win_chain_length=5, force_build_win_checks=False)
        board = BitBoard(cache, size=9, win_chain_length=5)

        board.move_coord(1, 0)
        board.move_coord(1, 1)
        board.move_coord(2, 0)
        board.move_coord(2, 1)
        board.move_coord(3, 0)
        board.move_coord(3, 1)
        board.move_coord(4, 0)
        board.move_coord(4, 1)

        searcher = PEvenSearch(board, mind.policy_est, mind.value_est, max_iterations=4,
                    p_batch_size=1024, verbose=True, validations=True)

        searcher.run(4)
        pv = searcher.get_pv()
        self.assertEqual(len(pv.get_move_chain()), 1)
        self.assertEqual(pv.get_q(), PExpNodeV3.MAX_Q)
        #self.assertEqual(pv.full_move)

    def test_search(self):
        mind = PExpMind_v3(size=9, init=False, verbose=True, search_params={
            'max_iterations': 10,
            'min_child_p': -7,
            'p_batch_size': 1 << 12,
            'q_fraction': 1
        })
        mind.load_net('../../models/voldmaster_' + str(0))

        cache = BitBoardCache("../cache/9-magics", size=9, win_chain_length=5, force_build_win_checks=False)
        board = BitBoard(cache, size=9, win_chain_length=5)

        moves = [
            (6, 0), (7, 0),
            (3, 1), (2, 4),
            (0, 3), (3, 4),
            (2, 6), (4, 4),
            (3, 5), (2, 5),
            (4, 5), (2, 7),
            (5, 5)

        ]
        for move in moves:
            board.move_coord(*move)

        print(board)
        searcher = PEvenSearch(board, mind.policy_est, mind.value_est)

        searcher.run(num_iterations=5)

        pv = searcher.get_pv()
        self.assertEqual(len(pv.get_move_chain()), 3)
        self.assertEqual(pv.get_q(), PExpNodeV3.MIN_Q)

    def test_move_order_search(self):
        mind = PExpMind_v3(size=9, init=False, search_params=None)
        mind.load_net('../../models/voldmaster_' + str(0))

        cache = BitBoardCache("../cache/9-magics", size=9, win_chain_length=5, force_build_win_checks=False)
        board = BitBoard(cache, size=9, win_chain_length=5)

        moves = [
            (7, 1), (0, 1),
            (7, 3), (1, 7),
            (7, 4), (3, 1),
            (3, 2), (4, 1),
            (3, 4), (6, 1),
            (6, 3), (6, 2),
            (5, 2), (8, 7),
            (7, 7), (6, 8),
            (5, 8), (1, 6),
            (2, 7)
        ]
        for move in moves:
            board.move_coord(*move)

        print(board)
        searcher = PEvenSearch(board, mind.policy_est, mind.value_est,
                               search_params={
                                   'max_iterations': 10,
                                   'min_child_p': -7,
                                   'p_batch_size': 1 << 10,
                                   'q_fraction': 1
                               })

        searcher.run(3)
        self.assertEqual(searcher.get_pv().calculate_pv_order(), [19, 46, 10])

    def just_no_crash_from_position(self, mind, board, moves, num_iterations):

        for move in moves:
            board.move(move)

        print(board)
        searcher = PEvenSearch(board, mind.policy_est, mind.value_est,
                               search_params={
                                   'max_iterations': 2,
                                   'min_child_p': -7,
                                   'p_batch_size': 1 << 10,
                                   'q_fraction': 1
                               })

        searcher.run(num_iterations=num_iterations)
        searcher.run()

    def test_able_to_draw(self):
        mind = PExpMind_v3(size=9, init=False, search_params=None)
        mind.load_net('../../models/voldmaster_' + str(0))

        cache = BitBoardCache("../cache/9-magics", size=9, win_chain_length=5, force_build_win_checks=False)
        board = BitBoard(cache, size=9, win_chain_length=5)

        # uninteresting moves for P
        moves = [37, 45, 74, 50, 47, 29, 31, 39, 49, 58, 48, 41, 40, 32, 22, 13, 23, 59, 68, 21, 60, 36, 5, 42, 34, 20,
                 67, 57, 56, 65, 52, 44, 38, 12, 30, 14, 11, 15, 16, 43, 46, 54, 66, 18, 27, 72, 63, 28, 4, 69, 76, 55,
                 75, 77, 25, 7, 24, 26, 53, 61, 78, 73, 19, 8, 64, 35, 62, 79, 51, 6, 17, 3]

        self.just_no_crash_from_position(mind, board, moves, num_iterations=2)

        # last move on the board
        moves = [70, 69, 10, 42, 43, 25, 50, 40, 49, 48, 32, 41, 39, 58, 38, 31, 51, 22, 13, 52, 21, 29, 67, 59, 56, 37,
                 61, 30, 28, 47, 12, 14, 11, 9, 33, 3, 23, 53, 27, 24, 6, 46, 57, 19, 55, 2, 44, 34, 4, 74, 7, 36, 5, 8,
                 54, 62, 16, 65, 66, 78, 75, 68, 1, 15, 76, 77, 26, 35, 45, 63, 18, 60, 79, 20, 0, 64, 80, 72, 73, 17]

        self.just_no_crash_from_position(mind, board, moves, num_iterations=2)

        # instant win
        moves = [23, 66, 44, 80, 35, 12, 33, 30, 31, 39, 40, 48, 75]
        self.just_no_crash_from_position(mind, board, moves, num_iterations=2)

        # no leaf nodes
        moves = [41, 8, 2, 29, 73, 31, 49, 40, 33, 50, 30, 39, 57, 22]
        self.just_no_crash_from_position(mind, board, moves, num_iterations=2)


if __name__ == '__main__':
    unittest.main()
