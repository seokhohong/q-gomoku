from learner import basic_mind
from learner import no_mind
from learner import minimax_mind
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


if __name__ == "__main__":
    mind = conv_mind.ConvMind(size=SIZE, alpha=0.9)

    c_mind = conv_mind.ConvMind(size=5, alpha=0.9)
    #c_mind.load('conv_mind_200.pkl')

    for i in range(5000):
        round_board = board.Board(size=SIZE, win_chain_length=4)
        current_player = round_board._player_to_move
        print('Game', i)
        if i % 100 == 0 and i > 0:
            copy_mind = conv_mind.ConvMind(size=SIZE, alpha=0.3)
            copy_mind.est = keras.models.clone_model(mind.est)
            minds.append(copy_mind)
            if len(minds) > 1:
                versus(minds[-1], minds[-2])
                versus(minds[-1], minds[0])
        #if i == 10:
        #    versus(c_mind, mind)
        if i == 50:
            mind.save('conv_mind_50.pkl')
        while True:
            result = mind.make_move(round_board, as_player=current_player, retrain=True, epsilon=0.3)
            print(round_board.pprint())
            current_player = -current_player
            if result:
                break
