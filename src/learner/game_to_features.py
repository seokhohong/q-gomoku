
import numpy as np

from src.core.board import Board

class FeatureSet_v11:
    ALPHA = 0.2
    def __init__(self, record):
        self.q_features = []
        self.q_labels = []
        self.p_features = []
        self.p_labels = []
        self.iterate_on(record)

    def make_feature_tensors(self, board, last_move, next_move, curr_q, next_q):
        # board's last move should be last_move, next_move not performed yet
        channels = 4
        tensor = np.zeros((board.get_size(), board.get_size(), channels))
        for i in range(board.get_size()):
            for j in range(board.get_size()):
                if board.get_spot(i, j) == Board.FIRST_PLAYER:
                    tensor[i, j, 0] = 1
                elif board.get_spot(i, j) == Board.SECOND_PLAYER:
                    tensor[i, j, 1] = 1
                if (i, j) == last_move:
                    tensor[i, j, 2] = 1
        if board.get_player_to_move() == Board.FIRST_PLAYER:
            tensor[:, :, 3].fill(1)
        else:
            tensor[:, :, 3].fill(-1)

        self.q_features.append(tensor)
        # here is the q learning update
        self.q_labels.append(curr_q * (1 - FeatureSet_v11.ALPHA) + next_q * FeatureSet_v11.ALPHA)
        self.p_features.append(tensor)
        self.p_labels.append(next_move[0] * board.get_size() + next_move[1])

    def iterate_on(self, record):
        initial_board = Board.parse_string(record.get_initial_state())
        q_assessments = record.get_q_assessments()
        last_move = None
        for i, move in enumerate(record.get_moves()):
            self.make_feature_tensors(initial_board, last_move, move, q_assessments[i][0], q_assessments[i][1])



