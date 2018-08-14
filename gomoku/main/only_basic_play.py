from learner import basic_mind
from learner import no_mind
from core import board

from sklearn.metrics import mean_absolute_error
import numpy as np

if __name__ == "__main__":
    mind = basic_mind.BasicMind(size=3, epsilon=.1, alpha=1)
    for i in range(5000):
        round_board = board.Board(size=3, win_chain_length=3)
        current_player = round_board._player_to_move
        print('Game', i)
        if i % 10 == 1:
            train_vectors = []
            train_labels = []
            for vector in mind.train_vectors.keys():
                train_vectors.append(vector)
                train_labels.append(mind.train_vectors[vector])
            mind.update_model()
            mind.est.fit(train_vectors, train_labels)
            print('Num States Seen', len(mind.train_vectors))
            print('MAE', mean_absolute_error(mind.est.predict(np.vstack(mind.train_vectors)), np.vstack(train_labels)))
        while(True):
            result = mind.make_move(round_board, as_player=current_player)
            print(round_board.pprint())
            current_player = -current_player
            if result:
                break
