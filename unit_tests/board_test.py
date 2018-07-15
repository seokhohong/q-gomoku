from core.board import Board

import unittest


class TestBoard(unittest.TestCase):
    def test_chainLength(self):
        board = Board(size=15, win_chain_length=5)
        board.make_move(7, 7)
        board.make_move(6, 7)
        board.make_move(8, 7)
        board.make_move(9, 7)
        self.assertEqual(len(board.available_moves), board.size ** 2 - 4)
        self.assertEqual(board.chain_length(7, 7, 1, 0), 2)
        self.assertEqual(board.chain_length(8, 7, -1, 0), 2)
        self.assertEqual(board.chain_length(8, 7, 1, 0), 1)
        self.assertEqual(board.chain_length(8, 7, 0, 1), 1)
        board.make_move(10, 7)
        self.assertEqual(board.chain_length(7, 7, 1, 0), 2)
        self.assertEqual(board.chain_length(8, 8, 1, 0), 0)
        board.make_move(7, 8)
        board.make_move(8, 8)
        board.make_move(7, 9)
        board.make_move(8, 9)
        board.make_move(7, 10)
        board.make_move(8, 10)
        board.make_move(7, 11)
        print(board.pprint())
        board.make_move(8, 11)

        self.assertTrue(board.game_won())

        self.assertEqual(board.in_bounds(16, 5), False)
        self.assertEqual(board.in_bounds(5, 5), True)
        self.assertEqual(board.in_bounds(-5, 2), False)

    def test_tictac(self):
        board = Board(size=3, win_chain_length=3)
        board.make_move(0, 0)
        board.make_move(1, 2)
        board.make_move(1, 1)
        board.make_move(2, 1)
        board.make_move(2, 2)
        self.assertTrue(board.game_won())

    def test_tictac2(self):
        board = Board(size=3, win_chain_length=3)
        board.make_move(0, 1)
        board.make_move(1, 2)
        board.make_move(0, 0)
        board.make_move(2, 0)
        board.make_move(1, 1)
        board.make_move(1, 0)
        board.make_move(0, 2)
        self.assertTrue(board.game_won())

    def test_5x5(self):
        board = Board(size=5, win_chain_length=4)
        board.make_move(3, 0)
        board.make_move(2, 0)
        board.make_move(3, 1)
        board.make_move(2, 1)
        board.make_move(3, 2)
        board.make_move(2, 2)
        board.make_move(3, 4)
        board.make_move(2, 4)
        board.make_move(3, 3)
        print(board.pprint())
        self.assertTrue(board.game_won())
        self.assertTrue(board.game_over())

    def test_move_unmove(self):
        board = Board(size=3, win_chain_length=3)
        board.make_move(0, 0)
        board.make_move(1, 0)
        board.make_move(0, 1)
        board.make_move(1, 1)
        board.make_move(0, 2)
        self.assertTrue(board.game_won())
        board.unmove()
        self.assertFalse(board.game_won())
        board.make_move(0, 2)
        self.assertTrue(board.game_won())

    def test_draw(self):
        board = Board(size=3, win_chain_length=3)
        board.make_move(0, 0)
        board.make_move(0, 2)
        board.make_move(0, 1)
        board.make_move(1, 0)
        board.make_move(1, 1)
        board.make_move(2, 2)
        board.make_move(2, 0)
        board.make_move(2, 1)
        board.make_move(1, 2)
        print(board.pprint())
        self.assertTrue(board.game_drawn())
        self.assertFalse(board.game_won())

if __name__ == '__main__':
    unittest.main()

