
from qgomoku.core.board import Board, BoardTransform, Player

import json
import unittest
import numpy as np

class TestStringMethods(unittest.TestCase):
    def test_chain_length(self):
        board = Board(size=9, win_chain_length=4)
        self.assertEqual(board.get_spot_coord(0, 0), Player.NONE)
        board.move_coord(0, 0)
        self.assertEqual(board.get_spot_coord(0, 0), Player.FIRST)
        self.assertFalse(board.is_move_available(0))
        board.move_coord(0, 1)
        board.move_coord(1, 1)
        self.assertEqual(board.get_spot_coord(1, 1), Player.FIRST)
        index00 = board._transformer.coordinate_to_index(0, 0)
        index11 = board._transformer.coordinate_to_index(1, 1)
        assert (board.chain_length(index11, -1, 0) == 1)
        assert (board.chain_length(index11, -1, -1) == 2)
        assert (board.chain_length(index00, 1, 1) == 2)
        board.move_coord(1, 0)
        board.move_coord(2, 2)
        board.move_coord(2, 3)
        board.move_coord(3, 3)
        board.unmove()
        board.move_coord(3, 3)
        index33 = board._transformer.coordinate_to_index(3, 3)
        self.assertEqual(board.chain_length(index33, 1, 1), 1)
        self.assertEqual(board.get_spot_coord(2, 2), Player.FIRST)
        assert (board.game_won())

    def test_precomputed_checklengths(self):
        board = Board(size=9, win_chain_length=5)
        self.assertEqual(board._check_locations[(0, -1, 0)], ())
        self.assertEqual(board._check_locations[(0, 1, 0)], (1, 2, 3, 4))
        self.assertEqual(board._check_locations[(0, 0, 1)], (9, 18, 27, 36))

    def test_odd_board_bug(self):
        board = Board(size=9, win_chain_length=5)

        board.move_coord(4, 0)
        board.move_coord(4, 1)
        board.move_coord(3, 0)
        board.move_coord(3, 1)
        board.move_coord(2, 0)
        board.move_coord(2, 1)
        board.move_coord(1, 0)
        board.move_coord(1, 1)
        board.move_coord(0, 0)
        self.assertTrue(board.game_won())


    def test_export_parse(self):
        board = Board(size=9, win_chain_length=5)
        for i in range(10):
            board.make_random_move()
            parsed_board = Board.load(board.export())
            self.assertEqual(board.pprint(lastmove_highlight=False), parsed_board.pprint(lastmove_highlight=False))
            self.assertFalse(parsed_board.game_over())
            self.assertFalse(parsed_board.game_won())

        self.assertTrue(np.equal(parsed_board._matrix, board._matrix).all())

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
            board.move_coord(move[0], move[1])
            print(board.pprint())
            index = rot.coordinate_to_index(move[0], move[1])
            rotated_matrices = rot.get_rotated_matrices(board._matrix.reshape((board.get_size(), board.get_size(), 1)))
            self.assertFalse(np.equal(rotated_matrices[0], rotated_matrices[2]).all())
            for point, mat in zip(rot.get_rotated_points(index), rotated_matrices):
                x, y = rot.index_to_coordinate(point)
                self.assertEqual(mat[x][y][0], board.get_player_last_move().value)

    def test_mass_play(self):
        for i in range(1000):
            board = Board(size=7, win_chain_length=4)
            board.make_random_move()
            if board.game_over():
                self.assertTrue(board.game_won() or board.game_assume_drawn())
            if board.game_assume_drawn():
                self.assertTrue(board.game_over())

    def test_deepcopy(self):
        board = Board(size=7, win_chain_length=4)
        import copy
        copied_board = copy.deepcopy(board)
        board.move(22)
        self.assertNotEqual(board.get_spot(22), copied_board.get_spot(22))


if __name__ == '__main__':
    unittest.main()


