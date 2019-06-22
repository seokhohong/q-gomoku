
from src.core.board import Board, BoardTransform
from src.core.game_record import GameRecord
from src.learner.game_to_features import FeatureSet_v1_1, FeatureBoard_v1_1
import unittest
import numpy as np

class TestStringMethods(unittest.TestCase):
    def test_basic_parsing(self):
        sample_gamestring = '{"initial_state": {"size": "9", "win_chain_length": "5", "boardstring": "120000000120000000120000000120000000000000000000000000000000000000000000000000000", "player_to_move": "1"}, "moves": [[4, 0]], "winning_player": 1, "q_assessments": [[-4.045114994049072, 1.0]]}'

        feature_set = FeatureSet_v1_1(sample_gamestring)
        self.assertEqual(len(feature_set.get_q()[0]), 8)
        self.assertEqual(len(feature_set.get_p()[0]), 8)

    def verify_sync(self, board, fboard):
        for j in range(board.get_size()):
            for k in range(board.get_size()):
                if board.get_spot(j, k) == Board.FIRST_PLAYER:
                    self.assertEqual(fboard.get_q_features()[j, k, 0], 1)
                    self.assertEqual(fboard.get_q_features()[j, k, 1], 0)
                elif board.get_spot(j, k) == Board.SECOND_PLAYER:
                    self.assertEqual(fboard.get_q_features()[j, k, 1], 1)
                else:
                    self.assertEqual(fboard.get_q_features()[j, k, 0], 0)
                    self.assertEqual(fboard.get_q_features()[j, k, 1], 0)

    def test_feature_board(self):
        size = 9
        board = Board(size=size, win_chain_length=5)
        fboard = FeatureBoard_v1_1(board)
        self.assertEqual(fboard.get_q_features()[0][0][3], Board.FIRST_PLAYER)
        self.assertEqual(np.sum(fboard.get_p_features()[:, :, 2]), 0)
        self.verify_sync(board, fboard)
        for i in range(20):
            board.make_random_move()
            last_move = board.get_last_move()
            fboard.move(last_move)
            self.assertEqual(np.sum(fboard.get_p_features()[:, :, 0]) + np.sum(fboard.get_p_features()[:, :, 1]), i + 1)
            self.verify_sync(board, fboard)

        for i in range(10):
            board.unmove()
            fboard.unmove()
            self.verify_sync(board, fboard)

        for i in range(1000):
            if np.random.rand() < 0.5:
                board.make_random_move()
                if board.game_over():
                    break
                last_move = board.get_last_move()
                fboard.move(last_move)
            elif board.get_last_move():
                board.unmove()
                fboard.unmove()
            self.verify_sync(board, fboard)


    def validate_rotation(self):

        sample_gamestring = '{"initial_state": {"size": "9", "win_chain_length": "5", "boardstring": "120000000120000000120000000120000000000000000000000000000000000000000000000000000", "player_to_move": "1"}, "moves": [[4, 0]], "winning_player": 1, "q_assessments": [[-4.045114994049072, 1.0]]}'

        feature_set = FeatureSet_v1_1(sample_gamestring)

        p_features, p_labels = feature_set.get_p()

        trans = BoardTransform(size=9)

        record = GameRecord.parse(sample_gamestring)
        for i in range(len(record.moves)):
            index = trans.coordinate_to_index(*record.moves[i])
            self.assertEqual(p_labels[i * 8: (i + 1) * 8], trans.get_rotated_points(index))

        for mat, label in zip(p_features, p_labels):
            x, y = trans.index_to_coordinate(label)
            self.assertEqual(mat[x, y, 4], 1)

        for i in range(len(p_labels)):
            x, y = trans.index_to_coordinate(p_labels[i])
            print(p_labels[i])
            # checking rotations are working by validating rotation
            if p_features[i][x, y, 0] + p_features[i][x, y, 1] != Board.STONE_ABSENT:
                print(p_features[i][x, y])
                print(p_features[i][:, :, 0] + p_features[i][:, :, 1], x, y, i, p_labels[i])
            self.assertEqual(p_features[i][x, y, 0] + p_features[i][x, y, 1], Board.STONE_ABSENT)

if __name__ == '__main__':
    unittest.main()
    #TestStringMethods().validate_rotation()


