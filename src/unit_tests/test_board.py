
from src.core.board import Board, BoardTransform

import json
import unittest
import numpy as np

class TestStringMethods(unittest.TestCase):
    def test_chain_length(self):
        board = Board(size=5, win_chain_length=4)
        self.assertEqual(board._matrix[0, 0, Board.NO_PLAYER], Board.STONE_PRESENT)
        board.move(0, 0)
        self.assertEqual(board._matrix[0, 0, Board.NO_PLAYER], Board.STONE_ABSENT)
        self.assertFalse(board.is_move_available(0, 0))
        self.assertEqual(board.get_spot(0, 0), Board.FIRST_PLAYER)
        board.move(0, 1)
        board.move(1, 1)
        self.assertEqual(board.get_spot(1, 1), Board.FIRST_PLAYER)
        assert (board.chain_length(1, 1, -1, 0) == 1)
        assert (board.chain_length(1, 1, -1, -1) == 2)
        assert (board.chain_length(0, 0, 1, 1) == 2)
        board.move(1, 0)
        board.move(2, 2)
        board.move(2, 3)
        board.move(3, 3)
        board.unmove()
        board.move(3, 3)
        assert (board.chain_length(3, 3, 1, 1) == 1)
        print(board.pprint())
        self.assertEqual(board._matrix[2, 2, Board.FIRST_PLAYER], Board.STONE_PRESENT)
        self.assertEqual(board._matrix[2, 2, Board.SECOND_PLAYER], Board.STONE_ABSENT)
        assert (board.game_won())

    def test_export_parse(self):
        board = Board(size=9, win_chain_length=5)
        for i in range(10):
            board.make_random_move()
            parsed_board = Board.load(board.export())
            self.assertEqual(board.pprint(lastmove_highlight=False), parsed_board.pprint(lastmove_highlight=False))
            self.assertFalse(parsed_board.game_over())
            self.assertFalse(parsed_board.game_won())

        for i in range(board._matrix.shape[0]):
            for j in range(board._matrix.shape[1]):
                for k in range(board._matrix.shape[2]):
                    self.assertEqual(parsed_board._matrix[i, j, k], board._matrix[i, j, k])

    def test_double_serialize(self):
        board = Board(size=9, win_chain_length=5)
        json.loads(json.dumps(board.export()))

    def test_rotator(self):
        rot = BoardTransform(size=9)

        for i in range(9 * 9):
            x, y = rot.index_to_coordinate(i)
            #print(x, y, i)
            self.assertEqual(i, rot.coordinate_to_index(x, y))

        board = Board(size=9, win_chain_length=5)

        moves = [(0, 0), (2, 1), (5, 5), (4, 4)]
        for i, move in enumerate(moves):
            board.move(move[0], move[1])
            print(board.pprint())
            index = rot.coordinate_to_index(move[0], move[1])
            rotated_matrices = rot.get_rotated_matrices(board._matrix)
            self.assertFalse(np.equal(rotated_matrices[0], rotated_matrices[2]).all())
            for point, mat in zip(rot.get_rotated_points(index), rotated_matrices):
                x, y = rot.index_to_coordinate(point)
                #if mat[x, y, Board.NO_PLAYER] != Board.STONE_ABSENT:
                #print(x, y, point)
                #print(mat[:, :, Board.NO_PLAYER])
                #print(mat[x, y, Board.NO_PLAYER])
                self.assertEqual(mat[x, y, Board.NO_PLAYER], Board.STONE_ABSENT)
                self.assertEqual(mat[x, y, board.get_player_last_move()], Board.STONE_PRESENT)

if __name__ == '__main__':
    unittest.main()


