
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed

set_random_seed(2)

from core.board import Board
from core import minimax
from learner.deep_conv_mind import DeepConvMind
from learner.conv_mind import ConvMind
import numpy as np
from copy import copy

import unittest


class TestBoard(unittest.TestCase):
    # def test_win(self):
    #     mind = DeepConvMind(size=5, alpha=0)
    #
    #     board = Board(size=5, win_chain_length=4)
    #
    #     board.hypothetical_move(0, 0)
    #     board.hypothetical_move(1, 0)
    #     board.hypothetical_move(0, 1)
    #     board.hypothetical_move(1, 1)
    #     board.hypothetical_move(0, 2)
    #     board.hypothetical_move(1, 2)
    #     possible_moves = mind.pvs_best_moves(board, max_iters=1, k=25)
    #     print(possible_moves[0][1])
    #     self.assertAlmostEqual(possible_moves[0][1].principle_variation.q, 1)
    #
    #     board = Board(size=5, win_chain_length=4)
    #
    #     board.hypothetical_move(0, 0)
    #     board.hypothetical_move(1, 0)
    #     board.hypothetical_move(0, 1)
    #     board.hypothetical_move(1, 1)
    #     board.hypothetical_move(0, 2)
    #     board.hypothetical_move(1, 2)
    #     board.hypothetical_move(2, 2)
    #     possible_moves = mind.pvs_best_moves(board, max_iters=1, k=25)
    #     print(possible_moves[0][1])
    #     self.assertAlmostEqual(possible_moves[0][1].principle_variation.q, -1)
    #
    # def test_2(self):
    #     mind = DeepConvMind(size=5, alpha=0)
    #     board = Board(size=5, win_chain_length=4)
    #     board.hypothetical_move(2, 0)
    #     board.hypothetical_move(0, 0)
    #     board.hypothetical_move(1, 3)
    #     board.hypothetical_move(0, 3)
    #     board.hypothetical_move(3, 0)
    #     board.hypothetical_move(1, 0)
    #     board.hypothetical_move(2, 2)
    #     board.hypothetical_move(1, 1)
    #     board.hypothetical_move(2, 3)
    #     board.hypothetical_move(1, 2)
    #     board.hypothetical_move(2, 4)
    #     board.hypothetical_move(3, 3)
    #     board.hypothetical_move(3, 2)
    #     board.hypothetical_move(4, 0)
    #     board.hypothetical_move(3, 4)
    #     board.hypothetical_move(4, 2)
    #     board.hypothetical_move(4, 1)
    #     board.hypothetical_move(4, 4)
    #     board.hypothetical_move(4, 3)
    #     board.hypothetical_move(1, 4)
    #
    #     print(board.pprint())
    #
    #     possible_moves = mind.pvs_best_moves(board, board.player_to_move, max_iters=1, k=25)
    #     print(possible_moves[0][1])
    #     self.assertAlmostEqual(possible_moves[0][1].principle_variation.q, 1)
    #
    # def test_consistency(self):
    #     board = Board(size=5, win_chain_length=4)
    #
    #     as_player = 1
    #     root_node = minimax.PVSNode(parent=None,
    #                                      is_maximizing=True if as_player == 1 else False,
    #                                      full_move_list=minimax.MoveList(moves=()))
    #
    #     mind = DeepConvMind(size=5, alpha=0)
    #
    #     principle_variations = [root_node]
    #     mind.pvs_batch_q(board, principle_variations)
    #
    #     for move in copy(board.available_moves):
    #         board.hypothetical_move(move[0], move[1])
    #         vector, player = mind.negamax_feature_vector(board)
    #         q = mind.est.predict([[vector.reshape(board.size, board.size, 1)], [player]])[0][0]
    #         self.assertAlmostEqual(q, root_node.children[move[0], move[1]].q, places=5)
    #         board.unmove()
    #
    #     mind.pvs_batch_q(board, root_node.children.values())
    #
    #     third_layer = []
    #     for child in root_node.children.values():
    #         third_layer.extend(child.children.values())
    #
    #     mind.pvs_batch_q(board, third_layer)
    #
    #     move_list = [(2, 2), (2, 1), (4, 0)]
    #
    #     for move in move_list:
    #         board.hypothetical_move(move[0], move[1])
    #
    #     vector, player = mind.negamax_feature_vector(board)
    #     q = mind.est.predict([[vector.reshape(board.size, board.size, 1)], [player]])[0][0]
    #     self.assertAlmostEqual(q, root_node.children[move_list[0][0], move_list[0][1]]
    #                                         .children[move_list[1][0], move_list[1][1]]
    #                                         .children[move_list[2][0], move_list[2][1]]
    #                                         .q, places=5)
    #
    #     for i in range(len(move_list)):
    #         board.unmove()
    #
    #
    # def test_batch(self):
    #     board = Board(size=5, win_chain_length=4)
    #
    #     as_player = 1
    #     max_depth = 3
    #     max_iters = 1
    #     k = 5
    #     root_node = minimax.PVSNode(parent=None,
    #                                 is_maximizing=True if as_player == 1 else False,
    #                                 full_move_list=minimax.MoveList(moves=()))
    #
    #     principle_variations = [root_node]
    #
    #     # identical ply-2 architecture check
    #     mind = DeepConvMind(size=5, alpha=0)
    #     mind.load('conv_mind_50.pkl')
    #
    #     conv = ConvMind(size=5, alpha=0)
    #     conv.load('conv_mind_50.pkl')
    #
    #     self.assertAlmostEqual(mind.q(board, as_player), conv.q(board, as_player))
    #
    #     board.hypothetical_move(2, 2)
    #
    #     self.assertAlmostEqual(mind.q(board, as_player), conv.q(board, as_player))
    #
    #     board.unmove()
    #
    #     mind.pvs_batch_q(board, principle_variations)
    #
    #     board.hypothetical_move(2, 2)
    #     vector, to_move = mind.negamax_feature_vector(board)
    #     self.assertAlmostEqual(mind.est.predict([vector.reshape(1, mind.size, mind.size, 1), np.array([to_move])])[0][0],
    #                                     root_node.children[(2, 2)].q, places=4)
    #     board.unmove()
    #
    #     print('After 1 Iter')
    #     for node in list(root_node.children.values()):
    #         print(node)
    #
    #     mind.pvs_batch_q(board, root_node.children.values())
    #
    #     board.hypothetical_move(2, 2)
    #     board.hypothetical_move(3, 2)
    #     vector, to_move = mind.negamax_feature_vector(board)
    #     self.assertAlmostEqual(mind.est.predict([vector.reshape(1, mind.size, mind.size, 1), np.array([to_move])])[0][0],
    #                                     -root_node.children[(2, 2)].children[(3, 2)].q, places=4)
    #     board.unmove()
    #     board.unmove()
    #
    #     print('After 2 Iters')
    #
    #     moves, qs = conv.minimax_q(board)
    #     for move, q in zip(moves, qs):
    #         print(move, q)
    #
    #     for move in moves:
    #         print(root_node.children[move].q)
    #
    #     for move, q in zip(moves, qs):
    #         self.assertAlmostEqual(root_node.children[move].q, q, places=4)
    #
    # def test_pvs(self):
    #     mind = DeepConvMind(size=5, alpha=0)
    #     board = Board(size=5, win_chain_length=4)
    #
    #     possible_moves = mind.pvs_best_moves(board,
    #                                          max_depth=2)
    #
    #     best_move, best_q = mind.pvs(board, max_depth=2, epsilon=0)
    #
    #     self.assertEqual(possible_moves[0][0], best_move)
    #     self.assertAlmostEqual(possible_moves[0][1].q, best_q, places=5)
    #
    #     board.make_move(2, 2)
    #
    #     possible_moves = mind.pvs_best_moves(board,
    #                                          max_depth=2)
    #
    #     best_move, best_q = mind.pvs(board, max_depth=2, epsilon=0)
    #
    #     self.assertEqual(possible_moves[0][0], best_move)
    #     self.assertAlmostEqual(possible_moves[0][1].q, best_q, places=5)

    # def test_train_vectors(self):
    #
    #     mind = DeepConvMind(size=5, alpha=0.5)
    #     mind.load('conv_mind_50.pkl')
    #     mind.make_move(Board(size=5, win_chain_length=4), 1, max_depth=2, epsilon=0)
    #
    #     conv = ConvMind(size=5, alpha=0.5)
    #     conv.load('conv_mind_50.pkl')
    #     conv.make_move(Board(size=5, win_chain_length=4), 1, epsilon=0)
    #
    #     self.assertEqual(len(mind.train_vectors), len(conv.train_vectors))
    #     for i in range(len(mind.train_vectors)):
    #         self.assertTrue(np.array_equal(mind.train_vectors[i][0].reshape(-1), conv.train_vectors[i][0].reshape(-1)))
    #
    #     for elem1, elem2 in zip(mind.train_labels, conv.train_labels):
    #         self.assertAlmostEqual(elem1, elem2)
    #
    #     mind.update_model()
    #     conv.update_model()
    #
    #     for weights1, weights2 in zip(mind.est.get_weights(), conv.est.get_weights()):
    #         for elem1, elem2 in zip(weights1, weights2):
    #             if type(elem1) != np.double:
    #                 if np.mean(elem1 - elem2) > 1E-6 or np.mean(elem1 - elem2) < -1E-6:
    #                     print(elem1, elem2)
    #                 self.assertAlmostEqual(np.mean(elem1 - elem2), 0, places=3)
    #             else:
    #                 self.assertAlmostEqual(elem1, elem2, places=3)
    #
    #     # reset brains
    #     mind.load('conv_mind_50.pkl')
    #     conv.load('conv_mind_50.pkl')
    #
    #     board = Board(size=5, win_chain_length=4)
    #     board.make_move(2, 2)
    #     mind.make_move(board, -1, max_depth=2, epsilon=0)
    #
    #     board = Board(size=5, win_chain_length=4)
    #     board.make_move(2, 2)
    #     conv.make_move(board, -1)
    #
    #     self.assertEqual(len(mind.train_vectors), len(conv.train_vectors))
    #     for i in range(len(mind.train_vectors)):
    #         self.assertTrue(np.array_equal(mind.train_vectors[i][0].reshape(-1), conv.train_vectors[i][0].reshape(-1)))
    #
    #     for elem1, elem2 in zip(mind.train_labels, conv.train_labels):
    #         self.assertAlmostEqual(elem1, elem2)
    #
    #     mind.update_model()
    #     conv.update_model()
    #
    #     for weights1, weights2 in zip(mind.est.get_weights(), conv.est.get_weights()):
    #         for elem1, elem2 in zip(weights1, weights2):
    #             if type(elem1) != np.double:
    #                 if np.mean(elem1 - elem2) > 1E-6 or np.mean(elem1 - elem2) < -1E-6:
    #                     print(elem1, elem2)
    #                 self.assertAlmostEqual(np.mean(elem1 - elem2), 0, places=2)
    #             else:
    #                 self.assertAlmostEqual(elem1, elem2, places=3)

    def test_converged(self):

        mind = DeepConvMind(size=5, alpha=0.5)
        mind.load('deep_conv_2500.pkl')

        board = Board(size=5, win_chain_length=4)


        board.make_move(2, 2)
        board.make_move(3, 3)
        board.make_move(1, 1)
        board.make_move(1, 3)
        board.make_move(2, 3)
        print(board.pprint())
        #mind.make_move(board, -1, max_depth=7, epsilon=0)

        print(mind.q(board, -1))
        print(mind.q(board, 1))

        board.make_move(1, 2)

        print(mind.q(board, 1))
        print(mind.q(board, -1))

if __name__ == '__main__':
    unittest.main()

