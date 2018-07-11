from core.board import Board
from core import minimax
from learner.deep_conv_mind import DeepConvMind
from learner.conv_mind import ConvMind
import numpy as np
from copy import copy

import unittest


class TestBoard(unittest.TestCase):
    # def test_treenode(self):
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
    #     possible_moves = mind.pvs_best_moves(board, board.player_to_move, max_iters=1, k=25)
    #     print(possible_moves[0][1])
    #     self.assertAlmostEqual(possible_moves[0][1].principle_variation.q, 1)
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
    #         self.assertAlmostEqual(mind.q(board, board.player_to_move), root_node.children[move[0], move[1]].principle_variation.q)
    #         board.unmove()
    #
    #     mind.pvs_batch_q(board, root_node.children.values())
    #
    #     depth1_move = move
    #
    #     board.make_move(depth1_move[0], depth1_move[1])
    #
    #     for move in copy(board.available_moves):
    #         board.hypothetical_move(move[0], move[1])
    #         self.assertAlmostEqual(mind.q(board, board.player_to_move), root_node.children[depth1_move].children[move[0], move[1]].principle_variation.q)
    #         board.unmove()
    #
    #     original_board = Board(size=5, win_chain_length=4)
    #     principle_variations = [root_node]
    #     mind.pvs_batch_q(board, principle_variations)


    def test_batch(self):
        board = Board(size=5, win_chain_length=4)

        as_player = 1
        max_depth = 3
        max_iters = 1
        k = 5
        root_node = minimax.PVSNode(parent=None,
                                    is_maximizing=True if as_player == 1 else False,
                                    full_move_list=minimax.MoveList(moves=()))

        principle_variations = [root_node]

        # identical ply-2 architecture check
        mind = DeepConvMind(size=5, alpha=0)
        mind.load('conv_mind_50.pkl')

        conv = ConvMind(size=5, alpha=0)
        conv.load('conv_mind_50.pkl')

        self.assertAlmostEqual(mind.q(board, as_player), conv.q(board, as_player))

        mind.pvs_batch_q(board, principle_variations)

        board.hypothetical_move(2, 2)
        vector, to_move = mind.negamax_feature_vector(board)
        self.assertAlmostEqual(mind.est.predict([vector.reshape(1, mind.size, mind.size, 1), np.array([to_move])])[0][0],
                                        root_node.children[(2, 2)].principle_variation.q)
        board.unmove()

        print('After 1 Iter')
        for node in list(root_node.children.values()):
            print(node)

        mind.pvs_batch_q(board, root_node.children.values())

        board.hypothetical_move(2, 2)
        board.hypothetical_move(3, 2)
        vector, to_move = mind.negamax_feature_vector(board)
        self.assertAlmostEqual(mind.est.predict([vector.reshape(1, mind.size, mind.size, 1), np.array([to_move])])[0][0],
                                        root_node.children[(2, 2)].children[(3, 2)].principle_variation.q)
        board.unmove()
        board.unmove()

        print('After 2 Iters')
        for node in list(root_node.children.values()):
            print(node)

        moves, qs = conv.minimax_q(board, as_player=1)
        for move, q in zip(moves, qs):
            print(move, q)

        for move, q in zip(moves, qs):
            self.assertAlmostEqual(root_node.children[move].principle_variation.q, q)


if __name__ == '__main__':
    unittest.main()

