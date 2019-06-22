
import numpy as np
import json

from src.core.board import Board, BoardTransform
from src.core.game_record import GameRecord

class FeatureSet_v1_1:
    ALPHA = 0.2
    CHANNELS = 4
    def __init__(self, record_string):
        self.q_features = []
        self.q_labels = []
        self.p_features = []
        self.p_labels = []
        self.iterate_on(GameRecord.parse(record_string))
        self.channels = 4

    @staticmethod
    def make_p_features(board, last_move):
        return FeatureSet_v1_1.make_features(board, last_move)

    @staticmethod
    def make_q_features(board, last_move):
        return FeatureSet_v1_1.make_features(board, last_move)

    @staticmethod
    def make_features(board, last_move):
        tensor = np.zeros((board.get_size(), board.get_size(), FeatureSet_v1_1.CHANNELS))
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
        return tensor

    def make_feature_tensors(self, board, last_move, next_move, curr_q, next_q):
        # board's last move should be last_move, next_move not performed yet
        feature_tensor = self.make_features(board, last_move)

        rot = BoardTransform(size=board.get_size())
        tensor_rotations = rot.get_rotated_matrices(feature_tensor)
        to_move_rotations = rot.get_rotated_points(rot.coordinate_to_index(*next_move))

        for i in range(len(tensor_rotations)):
            self.q_features.append(tensor_rotations[i])
            # here is the q learning update
            self.q_labels.append(curr_q * (1 - FeatureSet_v1_1.ALPHA) + next_q * FeatureSet_v1_1.ALPHA)
            self.p_features.append(feature_tensor)
            self.p_labels.append(to_move_rotations[i])

    def get_q(self):
        return self.q_features, self.q_labels

    def get_p(self):
        return self.p_features, self.p_labels

    def iterate_on(self, record):
        initial_board = Board.parse_string(record.get_initial_state())
        q_assessments = record.get_q_assessments()
        last_move = None
        for i, move in enumerate(record.get_moves()):
            self.make_feature_tensors(initial_board, last_move, move, q_assessments[i][0], q_assessments[i][1])
            print(initial_board.pprint())
            initial_board.move(*move)



