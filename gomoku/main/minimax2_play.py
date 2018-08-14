from learner import basic_mind
from learner import no_mind
from learner import minimax_mind
from core import board
from copy import deepcopy
from collections import defaultdict

from sklearn.metrics import mean_absolute_error
import numpy as np

minds = []
SIZE = 5

def versus(mind1, mind2, rounds=100):
    print("Versus!")
    wins = defaultdict(int)
    draws = 0
    for i in range(50):
        tourney_board = board.Board(size=SIZE, win_chain_length=3)
        players = [None, None]
        players[i % 2] = mind1
        players[(i + 1) % 2] = mind2
        while True:
            result = players[0].make_move(tourney_board, as_player=1, retrain=False, verbose=False)
            if result:
                if tourney_board.game_won():
                    wins[players[0]] += 1
                else:
                    draws += 1
                break
            result = players[1].make_move(tourney_board, as_player=-1, retrain=False, verbose=False)
            if result:
                if tourney_board.game_won():
                    wins[players[1]] += 1
                else:
                    draws += 1
                break

    print('Mind 1 Wins / Mind 2 Wins / Draws', wins[mind1], wins[mind2], draws)


def dump_training(mind):
    print('Dump')
    temp_board = board.Board(size=SIZE)
    for vector, label in zip(mind.train_vectors[:100], mind.train_labels[:100]):
        temp_board._matrix = np.array(vector[:-1]).reshape(SIZE, SIZE)
        print(temp_board.pprint(), label, 'TURN: ', vector[-1])


if __name__ == "__main__":
    mind = minimax_mind.MinimaxMind(size=SIZE, epsilon=.1, alpha=0.5)

    gbt_mind = minimax_mind.MinimaxMind(size=5, epsilon=0.1, alpha=0.3)
    gbt_mind.load('mini_gbt_50.pkl')

    nnet_mind = minimax_mind.MinimaxMind(size=5, epsilon=0.1, alpha=0.3)
    nnet_mind.load('mini_partialnnet_100.pkl')

    #versus(gbt_mind, nnet_mind)

    for i in range(5000):
        round_board = board.Board(size=SIZE, win_chain_length=4)
        current_player = round_board._player_to_move
        print('Game', i)
        #if i % 10 == 0:
        #    dump_training(mind)
        if i == 1000:
            mind.save('mini_gbt_1000.pkl')
        if i % 10000 == 0 and i > 0:
            copy_mind = minimax_mind.MinimaxMind(size=SIZE, epsilon=.1, alpha=0.3)
            copy_mind.est = deepcopy(mind.est)
            minds.append(copy_mind)
            if len(minds) > 1:
                versus(minds[-1], minds[-2])
                versus(minds[-1], minds[0])
        while True:
            result = mind.make_move(round_board, as_player=current_player)
            print(round_board.pprint())
            current_player = -current_player
            if result:
                break
