from learner import pexp_mind
from learner import deep_conv_mind
from core import detail_board
from core.board import Board
from copy import deepcopy
from collections import defaultdict
from learner import conv_mind
import random

from sklearn.metrics import mean_absolute_error
import numpy as np
import keras

import cProfile

minds = []

SIZE = 7

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
    #mind.load_net('../models/7_20')
    #c_mind = conv_mind.ConvMind(size=5, alpha=0.9)
    #c_mind.load('conv_mind_200.pkl')

    for i in range(50000):
        round_board = Board(size=SIZE, win_chain_length=5)

        print('Game', i)

        # randomize the board a bit
        for j in range(random.randint(0, 5)):
            round_board.make_random_move()

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
                                    max_depth=10,
                                    k=SIZE ** 3,
                                    max_iters=10,
                                    max_eval_q=500,
                                    )
            print(round_board.pprint())
            current_player = -current_player
            if result:
                break
            return

        print('done')
if __name__ == "__main__":
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