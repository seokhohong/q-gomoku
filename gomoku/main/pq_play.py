from learner import pqmind
from learner import deep_conv_mind
from core import detail_board
from copy import deepcopy
from collections import defaultdict
from learner import conv_mind
import random

from sklearn.metrics import mean_absolute_error
import numpy as np
import keras

import cProfile

minds = []

SIZE = 5

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
    mind = pqmind.PQMind(size=SIZE, alpha=0.2, init=True, channels=20)
    #mind.load('../models/pq_r1')
    #c_mind = conv_mind.ConvMind(size=5, alpha=0.9)
    #c_mind.load('conv_mind_200.pkl')

    for i in range(1, 50000):
        round_board = detail_board.Board(size=SIZE, win_chain_length=5)

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
                                    max_depth=depth_function(i),
                                    k=SIZE ** 2,
                                    max_iters=iter_function(i),
                                    )
            print(round_board.pprint())
            current_player = -current_player
            if result:
                break

        print('done')
if __name__ == "__main__":
    run()