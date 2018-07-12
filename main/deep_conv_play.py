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
def versus(mind1, mind2, rounds=100):
    print("Versus!")
    wins = defaultdict(int)
    draws = 0
    for i in range(50):
        print('Playing Versus Game', i)
        tourney_board = board.Board(size=5, win_chain_length=4)
        players = [None, None]
        players[i % 2] = mind1
        players[(i + 1) % 2] = mind2
        while True:
            result = players[0].make_move(tourney_board, as_player=1, retrain=False, verbose=False, epsilon=0.01)
            if result:
                if tourney_board.game_won():
                    wins[players[0]] += 1
                else:
                    draws += 1
                break
            result = players[1].make_move(tourney_board, as_player=-1, retrain=False, verbose=False, epsilon=0.01)
            if result:
                if tourney_board.game_won():
                    wins[players[1]] += 1
                else:
                    draws += 1
                break
        print('Final Result', tourney_board.pprint())

    print('Mind 1 Wins / Mind 2 Wins / Draws', wins[mind1], wins[mind2], draws)

def depth_function(i):
    if i < 200:
        return 2
    elif i < 400:
        return 3
    elif i < 600:
        return 4
    elif i < 800:
        return 5
    return 6

def epsilon_function(i):
    if i < 100:
        return 0.3
    elif i < 200:
        return 0.2
    elif i < 300:
        return 0.1
    return 0.03

if __name__ == "__main__":
    mind = deep_conv_mind.DeepConvMind(size=SIZE, alpha=1)
    #mind.load('conv_mind_50.pkl')
    #c_mind = conv_mind.ConvMind(size=5, alpha=0.9)
    #c_mind.load('conv_mind_200.pkl')

    for i in range(5000):
        round_board = board.Board(size=SIZE, win_chain_length=4)
        current_player = round_board.player_to_move
        print('Game', i)
        if i % 10000 == 0 and i > 0:
            copy_mind = deep_conv_mind.DeepConvMind(size=SIZE, alpha=1)
            copy_mind.est = keras.models.clone_model(mind.est)
            minds.append(copy_mind)
            if len(minds) > 1:
                versus(minds[-1], minds[-2])
                versus(minds[-1], minds[0])
        #if i == 10:
        #    versus(c_mind, mind)
        if i % 50 == 0 and i > 0:
            mind.save('deep_shape1.pkl')
        while True:
            result = mind.make_move(round_board,
                                    as_player=current_player,
                                    retrain=True,
                                    epsilon=0.1,
                                    max_depth=2,
                                    k=25,
                                    max_iters=10
                                    )
            print(round_board.pprint())
            current_player = -current_player
            if result:
                break
