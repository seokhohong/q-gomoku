from learner import deep_conv_mind
from learner import conv_mind
from core import board
from copy import deepcopy
from collections import defaultdict
import random

def versus(mind1, mind2, rounds=100):
    print("Versus!")
    wins = defaultdict(int)
    draws = 0

    for i in range(rounds):
        print('Playing Versus Game', i)
        tourney_board = board.Board(size=5, win_chain_length=4)
        # make a random move for first player
        for i in range(random.randint(0, 3)):
            tourney_board.make_random_move()
        players = {}
        players[1] = mind1
        players[-1] = mind2
        while True:
            result = players[tourney_board._player_to_move].make_move(tourney_board, as_player=tourney_board._player_to_move,
                                                                      retrain=False, verbose=False, epsilon=0.01, max_depth=15,
                                                                      max_iters=20)
            print(tourney_board.pprint())
            if result:
                if tourney_board.game_won():
                    wins[-tourney_board._player_to_move] += 1
                else:
                    draws += 1
                break

    return wins, draws


if __name__ == "__main__":
    SIZE=5
    mind1 = deep_conv_mind.DeepConvMind(size=SIZE, alpha=0.1)
    #mind1 = conv_mind.ConvMind(size=SIZE, alpha=0.1)
    #mind1.load('conv_mind_50.pkl')
    mind1.load('../models/pq_net_32_32')

    mind2 = deep_conv_mind.DeepConvMind(size=SIZE, alpha=0.1)
    mind2.load('../models/pq_net_32')

    wins = defaultdict(int)
    draws = 0

    round_wins, draw = versus(mind1, mind2, rounds=25)
    wins[1] += round_wins[1]
    wins[-1] += round_wins[-1]
    draws += draw

    # switch players
    round_wins, draw = versus(mind2, mind1, rounds=25)
    wins[-1] += round_wins[1]
    wins[1] += round_wins[-1]
    draws += draw

    print('Mind 1 Wins / Mind 2 Wins / Draws', wins[1], wins[-1], draws)