from learner import basic_mind
from learner import no_mind
from learner import minimax_mind
from learner import deep_conv_mind
from core import board
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
        return 2
    return 15

def run():
    mind = deep_conv_mind.DeepConvMind(size=SIZE, alpha=0.2, turn_input=True)
    mind.load('../models/pq_net_32_32')
    #mind.load('conv_mind_50.pkl')
    #c_mind = conv_mind.ConvMind(size=5, alpha=0.9)
    #c_mind.load('conv_mind_200.pkl')

    for i in range(50000):
        round_board = board.Board(size=SIZE, win_chain_length=4)

        print('Game', i)

        # randomize the board a bit
        for i in range(random.randint(0, 3)):
            round_board.make_random_move()

        current_player = round_board.player_to_move

        #    versus(c_mind, mind)
        #if i % 50 == 0 and i > 0:
        #    mind.save('../models/pq_bn_64_64_64')
        #if i % 100 == 1:
        #    with open('../models/train_vectors.npz', 'wb') as f:
        #        np.savez(f, train_vectors=mind.train_vectors, train_p=mind.train_p, train_q=mind.train_q)
        while True:
            result = mind.make_move(round_board,
                                    as_player=current_player,
                                    retrain=True,
                                    epsilon=0.1,
                                    max_depth=depth_function(i),
                                    k=25,
                                    max_iters=20,
                                    )
            print(round_board.pprint())
            current_player = -current_player
            if result:
                break
if __name__ == "__main__":
    cProfile.run('run()')