from learner import pexp_mind
from core.board import Board

from numpy.random import RandomState
import numpy as np

import cProfile

minds = []

SIZE = 9

def run():
    mind = pexp_mind.PExpMind(size=SIZE, init=False, channels=4)
    mind.load_net('../models/9_4_4')

    rs = RandomState(42)

    for i in range(50):
        board = Board(size=SIZE, win_chain_length=5, draw_point=50)

        print('Game', i)

        # randomize the board a bit
        for j in range(rs.randint(0, 10)):
            board.make_random_move()
        #board.move(2, 2)
        #board.move(0, 1)
        #board.move(2, 3)
        #board.move(2, 1)
        #board.move(2, 4)
        #board.move(3, 1)
        #board.move(4, 3)
        #board.move(3, 4)
        #board.move(6, 6)
        #board.move(4, 1)

        # board.move(0, 0)
        # board.move(1, 0)
        # board.move(0, 4)
        # board.move(2, 2)
        # board.move(0, 7)
        # board.move(3, 0)
        # board.move(1, 2)
        # board.move(3, 7)
        # board.move(1, 6)
        # board.move(4, 4)
        # board.move(1, 8)
        # board.move(4, 5)
        # board.move(2, 0)
        # board.move(4, 6)
        # board.move(2, 4)
        # board.move(4, 8)
        # board.move(2, 8)
        # board.move(5, 0)
        # board.move(3, 1)
        # board.move(5, 4)
        # board.move(3, 6)
        # board.move(5, 8)
        # board.move(4, 0)
        # board.move(6, 3)
        # board.move(4, 3)
        # board.move(7, 2)
        # board.move(4, 7)
        # board.move(7, 6)
        # board.move(6, 6)
        # board.move(8, 0)
        # board.move(8, 1)
        # board.move(8, 8)
        # board.move(8, 7)

        print(board)
        current_player = board.get_player_to_move()

        def expanding_p(depth, p):
            return np.logical_or.reduce([
                np.logical_and(depth < 2, p > -6),
                np.logical_and(depth < 4, p > -4),
                np.logical_and(depth < 6, p > -4),
                np.logical_and(depth < np.inf, p > -3)
            ])

        def permissive_expansion(depth):
            if depth < 2:
                return np.inf
            if depth < 8:
                return 5
            return 3

        mind.define_policies(expanding_p, permissive_expansion, convergence_count=5,
                             alpha=0.2, q_exp_batch_size=SIZE ** 2,
                             p_exp_batch_size=SIZE ** 3, required_depth=6, max_iters=20)

        while True:
            result = mind.make_move(board,
                                    as_player=current_player,
                                    epsilon=0.1,
                                    consistency_check=False,
                                    verbose=True)
            print(board.pprint())
            if current_player == Board.FIRST_PLAYER:
                current_player = Board.SECOND_PLAYER
            else:
                current_player = Board.FIRST_PLAYER
            if result:
                break

        print('done')
if __name__ == "__main__":
    #run()

    import cProfile, pstats
    from io import StringIO

    pr = cProfile.Profile()
    pr.enable()
    run()
    pr.disable()
    s = StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())
