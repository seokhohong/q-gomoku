

import unittest
from src.core.board import Board
from src.core.game_record import GameRecord

class TestStringMethods(unittest.TestCase):
    def test_parse(self):
        board = Board(size=9)
        board.move(5, 5)
        record = GameRecord.create(board)
        record.add_move((3, 5))
        record.set_winner(1)
        with self.assertRaises(ValueError):
            record.add_move((5, 5))
        board = Board(size=9)
        record2 = GameRecord.create(board)
        record2.add_move((5, 5))
        record2.add_move((3, 5))
        record2.set_winner(1)


if __name__ == '__main__':
    unittest.main()


