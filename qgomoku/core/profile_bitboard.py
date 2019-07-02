import cProfile
import pstats
from io import StringIO

from qgomoku.core.board import *
from qgomoku.util.bitops import *


def build():
    bitboard = BitBoardCache("../cache/9-magics", size=9, win_chain_length=5, force_build_magics=True,
                             force_build_win_checks=True)


def play_games():
    cache = BitBoardCache("../cache/9-magics", size=9, win_chain_length=5, force_build_magics=True,
                          force_build_win_checks=True)
    for i in range(100):
        bitboard = BitBoard(cache, size=9, win_chain_length=5, draw_point=50)
        board = Board(size=9, win_chain_length=5, draw_point=50)
        print(board.get_player_to_move())
        while not board.game_over():
            available_moves = board.get_available_moves()
            random_move = random.choice(list(available_moves))
            for j in range(100):
                board.move_index(random_move)
                bitboard.move(random_move)
                board.unmove()
                bitboard.unmove()
            board.move_index(random_move)
            bitboard.move(random_move)

            if board.game_over() != bitboard.game_over():
                bitboard.move(random_move)


def profile():
    pr = cProfile.Profile()
    pr.enable()
    play_games()
    pr.disable()
    s = StringIO()
    sortby = 'tottime'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())


if __name__ == "__main__":
    profile()
