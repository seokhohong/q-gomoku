from learner import basic_mind
from learner import no_mind
from core import board

from sklearn.metrics import mean_absolute_error
import numpy as np

if __name__ == "__main__":
    mind = basic_mind.BasicMind(size=3, epsilon=.2, alpha=0.9)
    #no_mind = no_mind.NoMind(size=3, epsilon=.1, alpha=0.2)
    for i in range(1000):
        round_board = board.Board(size=3, win_chain_length=3)
        current_player = round_board.player_to_move
        print('Game', i)
        if i % 100 == 1:
            vectors = []
            labels = []
            for vector in no_mind.q_memory.keys():
                vectors.append(vector)
                labels.append(no_mind.q_memory[vector])
                if len(vectors) > 1:
                    break
            print('Mind Error', mean_absolute_error(mind.est.predict(np.vstack(vectors)), labels))
        while(True):
            if current_player == 1:
                result = mind.make_move(round_board, as_player=current_player)
            else:
                current_player = -1
            print(round_board.pprint())
            current_player = -current_player
            if result:
                break
