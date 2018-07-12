from learner import deep_conv_mind
from core import board
from copy import deepcopy
from collections import defaultdict

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
            print(tourney_board.pprint())
            result = players[1].make_move(tourney_board, as_player=-1, retrain=False, verbose=False, epsilon=0.01)
            if result:
                if tourney_board.game_won():
                    wins[players[1]] += 1
                else:
                    draws += 1
                break
            print(tourney_board.pprint())


    print('Mind 1 Wins / Mind 2 Wins / Draws', wins[mind1], wins[mind2], draws)

if __name__ == "__main__":
    SIZE=5
    mind1 = deep_conv_mind.DeepConvMind(size=SIZE, alpha=0.1)
    mind1.load('conv_mind_200.pkl')

    mind2 = deep_conv_mind.DeepConvMind(size=SIZE, alpha=0.1, turn_input=False)
    mind2.load('no_turn.pkl')

    versus(mind1, mind2)