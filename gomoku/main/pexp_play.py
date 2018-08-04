from learner import pexp_mind
from learner import deep_conv_mind
from core import detail_board
from core.board import Board
from copy import deepcopy
from collections import defaultdict
from learner import conv_mind
import random

from numpy.random import RandomState
from sklearn.metrics import mean_absolute_error
import numpy as np
import keras

import cProfile

minds = []

SIZE = 9

def depth_function(i):
    if i < 500:
        return 1
    elif i < 1000:
        return 2
    return 15

def iter_function(i):
    if i < 500:
        return 1
    elif i < 1000:
        return 10
    elif i < 2000:
        return 20

def run():
    mind = pexp_mind.PExpMind(size=SIZE, alpha=0.2, init=False, channels=4)
    mind.load_net('../models/9_4_2')
    #c_mind = conv_mind.ConvMind(size=5, alpha=0.9)
    #c_mind.load('conv_mind_200.pkl')

    rs = RandomState(42)

    for i in range(50000):
        round_board = Board(size=SIZE, win_chain_length=5)

        print('Game', i)

        # randomize the board a bit
        #for j in range(rs.randint(0, 10)):
        #    round_board.make_random_move()
        round_board.move(2, 2)
        round_board.move(0, 1)
        round_board.move(2, 3)
        round_board.move(2, 1)
        round_board.move(2, 4)
        round_board.move(3, 1)
        round_board.move(4, 3)
        round_board.move(3, 4)
        round_board.move(6, 6)
        #round_board.move(4, 1)

        print(round_board)
        current_player = round_board.player_to_move

        #    versus(c_mind, mind)
        #if i % 50 == 0 and i > 0:
        #    mind.save('../models/pq_r1')
        #if i % 100 == 1:
        #    with open('../models/train_vectors_13_13.npz', 'wb') as f:
        #        np.savez(f, train_vectors=mind.train_vectors, train_p=mind.train_p, train_q=mind.train_q)
        while True:
            result = mind.make_move(round_board,
                                    as_player=current_player,
                                    epsilon=0.1,
                                    required_depth=4,
                                    k=SIZE ** 2,
                                    max_iters=30,
                                    )
            print(round_board.pprint())
            if current_player == Board.FIRST_PLAYER:
                current_player = Board.SECOND_PLAYER
            else:
                current_player = Board.FIRST_PLAYER
            if result:
                break
            #return

        print('done')
if __name__ == "__main__":
    run()

    # import cProfile, pstats
    # from io import StringIO
    #
    # pr = cProfile.Profile()
    # pr.enable()
    # run()
    # pr.disable()
    # s = StringIO()
    # sortby = 'cumulative'
    # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    # ps.print_stats()
    # print(s.getvalue())
