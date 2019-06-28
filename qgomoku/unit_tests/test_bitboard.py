import unittest

from qgomoku.core.board import *
from qgomoku.util.bitops import *

class TestStringMethods(unittest.TestCase):

    def test_bitops(self):
        self.assertFalse(get_bit(64, 5))
        self.assertTrue(get_bit(64, 6))

        self.assertTrue(has_consecutive_bits(15, 4))
        self.assertTrue(has_consecutive_bits(31, 4))
        self.assertTrue(has_consecutive_bits(30, 4))
        self.assertFalse(has_consecutive_bits(7, 4))

        self.assertEqual(bitops.array_of_set_bits(bitops.bitstring_with([37, 47, 57, 67])), [37, 47, 57, 67])

        for i in range(1000):
            a_number = random.randint(1 << 50, 1 << 90)
            self.assertEqual(a_number, bitops.bitstring_with(bitops.array_of_set_bits(a_number)))

    def test_bitboard(self):
        cache = BitBoardCache("../cache/9-magics", size=9, win_chain_length=5, force_build_win_checks=False)
        bitboard = BitBoard(cache, size=9, win_chain_length=5)
        bitboard.move_coord(4, 0)
        self.assertTrue(bitboard.get_winning_player() == Player.NONE)
        bitboard.move_coord(4, 1)
        bitboard.move_coord(1, 0)
        bitboard.move_coord(1, 1)
        self.assertTrue(bitboard.get_winning_player() == Player.NONE)
        bitboard.move_coord(2, 0)
        bitboard.move_coord(2, 1)
        self.assertTrue(bitboard.get_winning_player() == Player.NONE)
        bitboard.move_coord(3, 0)
        bitboard.move_coord(3, 1)
        self.assertTrue(bitboard.get_winning_player() == Player.NONE)
        bitboard.move_coord(0, 0)
        self.assertEqual(bitboard.get_spot(0), Player.FIRST)
        self.assertTrue(bitboard.get_winning_player() == Player.FIRST)
        # test game over when undo
        bitboard.unmove()
        self.assertFalse(bitboard.game_over())

    def test_board_equivalency(self):
        cache = BitBoardCache("../cache/9-magics", size=9, win_chain_length=5, force_build_win_checks=False)

        for i in range(1000):
            bitboard = BitBoard(cache, size=9, win_chain_length=5, draw_point=50)
            board = Board(size=9, win_chain_length=5, draw_point=50)
            while not board.game_over():
                available_moves = board.get_available_moves()
                self.assertEqual(available_moves, bitboard.get_available_moves())
                self.assertEqual(len(board._ops), len(bitboard._ops))
                self.assertEqual(board.game_status(), bitboard.game_status())
                random_move = random.choice(list(available_moves))
                #print('Print', board.pprint())
                board.move(random_move)

                if board.game_won() and not bitboard.is_winning_move(random_move) or \
                        not board.game_won() and bitboard.is_winning_move(random_move):
                    bitboard.is_winning_move(random_move)
                bitboard.move(random_move)
                self.assertNotEqual(np.sum(board._matrix), 0)
                #self.assertEqual(int(board.get_player_to_move()), bitboard.get_player_to_move().value)

                if board.game_over() != bitboard.game_over():
                    print(i, board.pprint())
                    print(bitboard.pprint())
                    bitboard.move(random_move)
                self.assertEqual(board.game_over(), bitboard.game_over())





if __name__ == '__main__':
    unittest.main()


