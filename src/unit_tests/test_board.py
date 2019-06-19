
from src.core.board import Board
import unittest

class TestStringMethods(unittest.TestCase):
    def test_chain_length(self):
        board = Board(size=5, win_chain_length=4)
        board.move(0, 0)
        assert (board.get_matrix()[0, 0, 0] == 0)
        self.assertFalse(board.is_move_available(0, 0))
        board.move(0, 1)
        board.move(1, 1)
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
        assert (board.get_matrix()[2, 2, Board.TURN_INFO_INDEX] == -1)
        assert (board.game_won())

    def test_export_parse(self):
        board = Board(size=9, win_chain_length=5)
        for i in range(10):
            board.make_random_move()
            self.assertEqual(board.pprint(lastmove_highlight=False), Board.parse_string(board.export_string()).pprint(lastmove_highlight=False))

if __name__ == '__main__':
    unittest.main()


