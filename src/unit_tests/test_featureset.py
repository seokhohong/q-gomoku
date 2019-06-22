
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

    def validate_rotation(self, gamestring):
        feature_set = FeatureSet_v1_1(gamestring)

        p_features, p_labels = feature_set.get_p()

        record = GameRecord.parse(gamestring)
        self.assertLessEqual(len(p_labels) / BoardTransform.NUM_ROTATIONS, (len(record.moves) / 2) + 1)

        trans = BoardTransform(size=9)

        for i in range(int(len(p_labels) / 8)):
            index = p_labels[i * 8]
            self.assertEqual(p_labels[i * 8: (i + 1) * 8], trans.get_rotated_points(index))

        for i in range(len(p_labels)):
            x, y = trans.index_to_coordinate(p_labels[i])
            # checking rotations are working by validating next_move loctaions
            self.assertEqual(p_features[i][x, y, 0] + p_features[i][x, y, 1], Board.STONE_ABSENT)

    def test_rotation(self):
        gamestring_1 = '{"initial_state": {"size": "9", "win_chain_length": "5", "boardstring": "120000000120000000120000000120000000000000000000000000000000000000000000000000000", "player_to_move": "1"}, "moves": [[4, 0]], "winning_player": 1, "q_assessments": [[-4.045114994049072, 1.0]]}'
        gamestring_2 = '{"initial_state": {"size": "9", "win_chain_length": "5", "boardstring": "002000000000010000000000000000000000000000000000000000000000000002010000010200000", "player_to_move": "1"}, "moves": [[5, 8], [5, 1], [8, 4], [7, 6], [7, 8], [5, 7], [0, 1], [0, 5], [1, 6], [7, 7], [5, 5], [4, 2], [1, 2], [5, 6], [2, 2], [0, 3], [2, 4], [8, 6], [3, 2], [7, 3], [6, 0], [0, 6], [0, 4], [6, 6], [4, 6], [6, 3], [3, 5], [0, 0], [3, 7], [6, 5], [1, 3], [1, 5], [2, 8], [6, 4], [6, 8], [6, 2]], "winning_player": 2, "q_assessments": [[-2.96258544921875, -0.9998999834060669], [1.284677505493164, 0.9998999834060669], [-2.7005796432495117, -0.9998999834060669], [1.2349812984466553, 0.9998999834060669], [-2.6320621967315674, -0.9998999834060669], [1.0544953346252441, 0.9998999834060669], [-3.017280101776123, -0.9998999834060669], [0.7588550448417664, 0.7588546872138977], [-3.176118850708008, -0.9998999834060669], [0.8611494302749634, 0.8611494898796082], [-3.90118670463562, -0.9998999834060669], [0.7197551131248474, 0.719754695892334], [-3.523841381072998, -0.9998999834060669], [0.8500248789787292, 0.8500248789787292], [-4.362323760986328, -0.9998999834060669], [0.4997439384460449, 0.499744176864624], [-5.566584587097168, -0.9998999834060669], [-0.26667264103889465, -0.26667261123657227], [-5.716538906097412, -0.9998999834060669], [0.04753449559211731, 0.04753461480140686], [-5.201706886291504, -0.9998999834060669], [0.06054648756980896, 0.060546159744262695], [-5.130354881286621, -0.9998999834060669], [-0.3093544840812683, -0.30935442447662354], [-4.59406042098999, -0.9998999834060669], [0.2482568472623825, 0.24825695157051086], [-4.58466911315918, -0.9998999834060669], [-0.16860592365264893, -0.16860602796077728], [-4.071069240570068, -0.9998999834060669], [-0.2602347135543823, -0.2602351903915405], [-4.271035671234131, -0.9998999834060669], [-0.9805136322975159, -0.9805132746696472], [-4.734828948974609, -0.9998999834060669], [-1.7593207359313965, -0.9998999834060669], [-4.622495651245117, -1.0], [-1.420495629310608, -1.0]]}'
        self.validate_rotation(gamestring_1)
        self.validate_rotation(gamestring_2)



if __name__ == '__main__':
    unittest.main()
    #TestStringMethods().validate_rotation()


