from learner import basic_mind
from learner import no_mind
from learner import minimax_mind
from learner import deep_conv_mind
from core import board
from copy import deepcopy
from collections import defaultdict
from learner import conv_mind

from sklearn.metrics import mean_absolute_error
import numpy as np
import keras

minds = []

SIZE = 5

if __name__ == "__main__":
    mind = deep_conv_mind.DeepConvMind(size=SIZE, alpha=0.2, turn_input=True)
    mind.load('../models/pq_net')
    #mind.load('conv_mind_50.pkl')
    #c_mind = conv_mind.ConvMind(size=5, alpha=0.9)
    #c_mind.load('conv_mind_200.pkl')

    for i in range(5000):
        round_board = board.Board(size=SIZE, win_chain_length=4)
        current_player = round_board.player_to_move
        print('Game', i)

        #    versus(c_mind, mind)
        if i % 10 == 0 and i > 0:
            mind.save('../models/pq_net')
        while True:
            result = mind.make_move(round_board,
                                    as_player=current_player,
                                    retrain=True,
                                    epsilon=0.1,
                                    max_depth=10,
                                    k=25,
                                    max_iters=20
                                    )
            print(round_board.pprint())
            current_player = -current_player
            if result:
                break
