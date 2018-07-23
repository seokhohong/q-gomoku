from learner import basic_mind
from learner import no_mind
from core import board

import pickle

from sklearn.metrics import mean_absolute_error
import numpy as np

if __name__ == "__main__":
    mind = no_mind.NoMind(size=3, epsilon=.1, alpha=0.3)
    for i in range(50000):
        round_board = board.Board(size=3, win_chain_length=3)
        current_player = round_board.player_to_move
        print('Game', i)
        while True:
            result = mind.make_move(round_board, as_player=current_player)
            print(round_board.pprint())
            current_player = -current_player
            if result:
                break
    with open('q_memory.pkl', 'wb') as f:
        pickle.dump(mind.q_memory, f)
