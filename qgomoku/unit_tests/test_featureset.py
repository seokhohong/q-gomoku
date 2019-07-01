
from qgomoku.core.board import Board, BoardTransform, Player, BitBoardCache, BitBoard
from qgomoku.core.game_record import GameRecord
from qgomoku.learner.game_to_features import FeatureSet_v1_1, FeatureBoard_v1_1
from qgomoku.learner.thoughtboard import ThoughtBoard
from qgomoku.learner.pexp_node_v3 import PExpNodeV3
import unittest
import numpy as np

class TestStringMethods(unittest.TestCase):
    def test_basic_parsing(self):
        pass
        #sample_gamestring = '{"initial_state": {"size": "9", "win_chain_length": "5", "boardstring": "120000000120000000120000000120000000000000000000000000000000000000000000000000000", "player_to_move": "1"}, "moves": [[4, 0]], "winning_player": 1, "q_assessments": [[-4.045114994049072, 1.0]]}'

        #feature_set = FeatureSet_v1_1(sample_gamestring)
        #self.assertEqual(len(feature_set.get_q()[0]), 8)
        #self.assertEqual(len(feature_set.get_p()[0]), 8)

    def verify_sync(self, board, fboard):
        for j in range(board.get_size()):
            for k in range(board.get_size()):
                if board.get_spot_coord(j, k) == Player.FIRST:
                    self.assertEqual(fboard.get_q_features()[j, k, 0], 1)
                    self.assertEqual(fboard.get_q_features()[j, k, 1], 0)
                elif board.get_spot_coord(j, k) == Player.SECOND:
                    self.assertEqual(fboard.get_q_features()[j, k, 1], 1)
                else:
                    self.assertEqual(fboard.get_q_features()[j, k, 0], 0)
                    self.assertEqual(fboard.get_q_features()[j, k, 1], 0)

    def test_board_equivalence(self):
        size = 9
        cache = BitBoardCache("../cache/9-magics", size=9, win_chain_length=5, force_build_win_checks=False)
        bitboard = BitBoard(cache, size=9, win_chain_length=5)

        normal_board = Board(size=9, win_chain_length=5)

        fboard_1 = FeatureBoard_v1_1(bitboard)
        fboard_2 = FeatureBoard_v1_1(normal_board)

        for i in range(1000):
            if np.random.rand() < 0.5:
                normal_board.make_random_move()
                bitboard.move(normal_board.get_last_move())
                if normal_board.game_over():
                    break
                fboard_1.move(normal_board.get_last_move())
                fboard_2.move(normal_board.get_last_move())
            elif normal_board.get_last_move():
                normal_board.unmove()
                bitboard.unmove()
                fboard_1.unmove()
                fboard_2.unmove()
            self.verify_sync(normal_board, fboard_2)
            self.verify_sync(bitboard, fboard_1)

    def test_feature_board(self):
        size = 9
        cache = BitBoardCache("../cache/9-magics", size=9, win_chain_length=5, force_build_win_checks=False)
        board = BitBoard(cache, size=9, win_chain_length=5)

        fboard = FeatureBoard_v1_1(board)
        # the last player to move was the other player so it should be -1
        self.assertEqual(fboard.get_q_features()[0][0][3], -1)
        self.assertEqual(np.sum(fboard.get_p_features()[:, :, 2]), 0)
        board.make_random_move()
        last_move = board.get_last_move()
        fboard.move(last_move)
        self.assertEqual(fboard.get_q_features()[0][0][3], 1)
        self.assertEqual(np.sum(fboard.get_p_features()[:, :, 2]), 1)
        self.verify_sync(board, fboard)
        for i in range(20):
            board.make_random_move()
            last_move = board.get_last_move()
            fboard.move(last_move)
            self.assertEqual(np.sum(fboard.get_p_features()[:, :, 0]) + np.sum(fboard.get_p_features()[:, :, 1]), i + 2)
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
            self.assertEqual(p_features[i][x, y, 0] + p_features[i][x, y, 1], 0)

    def test_features_after_move_v1_1(self):
        size = 9
        cache = BitBoardCache("../cache/9-magics", size=9, win_chain_length=5, force_build_win_checks=False)
        board = BitBoard(cache, size=9, win_chain_length=5)

        moves = [76, 65, 58, 27, 40]
        for move in moves:
            board.move(move)

        tboard = ThoughtBoard(board, FeatureBoard_v1_1)
        q_features = tboard.get_q_features()
        x, y = board._transformer.index_to_coordinate(moves[-1])
        self.assertEqual(q_features[x, y, 2], 1)

        #p_features = tboard.get_p_features_after(PExpNodeV3(parent=None, move=None, is_maximizing=board.get_player_to_move() == Player.FIRST))
        p_features = tboard.get_p_features_after([])

        x, y = board._transformer.index_to_coordinate(moves[-1])
        # last move check
        self.assertEqual(p_features[x, y, 2], 1)
        self.assertEqual(np.sum(p_features[:, :, 3]), size ** 2)

    def test_rotation(self):
        gamestring_1 = '{"initial_state": {"size": "9", "win_chain_length": "5", "boardstring": "000000000020100000000000000000000000000002000010002000000000001000000000000000000", "player_to_move": "1"}, "moves": [59, 30, 40, 32, 23, 31, 33, 51, 21, 29, 28, 61, 71, 20, 48, 58, 34, 38, 47, 2, 11, 49, 0, 53, 52, 39, 37, 19, 9, 13, 18, 57, 64, 55, 56, 27, 72], "winning_player": 1, "q_assessments": [[-0.42047926783561707, -0.27915751934051514], [-0.16502320766448975, -0.602515459060669], [-0.45055916905403137, -0.3326343595981598], [-0.012478411197662354, -0.5165740251541138], [-0.5012242794036865, -0.23471033573150635], [-0.09695667028427124, -0.5550012588500977], [-0.6695745587348938, -0.5237785577774048], [-0.20509940385818481, -0.6204209327697754], [-0.6644445657730103, -0.6786763668060303], [-0.31529515981674194, -0.6252139806747437], [-0.6600759029388428, -0.5297435522079468], [-0.3306339979171753, -0.5300264358520508], [-0.6044123768806458, -0.3451654613018036], [-0.3083556890487671, -0.46383896470069885], [-0.6169118285179138, -0.4346025884151459], [-0.24419206380844116, -0.4125952422618866], [-0.5049172639846802, -0.33616673946380615], [-0.26773810386657715, -0.4608137309551239], [-0.6182757616043091, -0.42278173565864563], [-0.2608034610748291, -0.40805116295814514], [-0.4958721399307251, -0.26431113481521606], [-0.21211618185043335, -0.38622453808784485], [-0.49097567796707153, -0.38689103722572327], [-0.29656893014907837, -0.3982450067996979], [-0.5061854124069214, -0.29522737860679626], [-0.19089674949645996, -0.312770277261734], [-0.3866199851036072, -0.20948490500450134], [-0.12915736436843872, -0.27371785044670105], [-0.30733999609947205, -0.186032235622406], [-0.15766698122024536, -0.164814293384552], [-0.33021798729896545, -0.07221251726150513], [-0.16198641061782837, -0.13520735502243042], [-0.3360101282596588, 1.0], [-0.18188196420669556, -0.07597750425338745], [-0.3248758614063263, -0.0915951132774353], [-0.10910540819168091, -0.047596871852874756], [-0.19368615746498108, 1.0]]}'
        self.validate_rotation(gamestring_1)


    def test_full_parsing(self):
        gamestring_1 = '{"initial_state": {"size": "9", "win_chain_length": "5", "boardstring": "000000000020100000000000000000000000000002000010002000000000001000000000000000000", "player_to_move": "1"}, "moves": [59, 30, 40, 32, 23, 31, 33, 51, 21, 29, 28, 61, 71, 20, 48, 58, 34, 38, 47, 2, 11, 49, 0, 53, 52, 39, 37, 19, 9, 13, 18, 57, 64, 55, 56, 27, 72], "winning_player": 1, "q_assessments": [[-0.42047926783561707, -0.27915751934051514], [-0.16502320766448975, -0.602515459060669], [-0.45055916905403137, -0.3326343595981598], [-0.012478411197662354, -0.5165740251541138], [-0.5012242794036865, -0.23471033573150635], [-0.09695667028427124, -0.5550012588500977], [-0.6695745587348938, -0.5237785577774048], [-0.20509940385818481, -0.6204209327697754], [-0.6644445657730103, -0.6786763668060303], [-0.31529515981674194, -0.6252139806747437], [-0.6600759029388428, -0.5297435522079468], [-0.3306339979171753, -0.5300264358520508], [-0.6044123768806458, -0.3451654613018036], [-0.3083556890487671, -0.46383896470069885], [-0.6169118285179138, -0.4346025884151459], [-0.24419206380844116, -0.4125952422618866], [-0.5049172639846802, -0.33616673946380615], [-0.26773810386657715, -0.4608137309551239], [-0.6182757616043091, -0.42278173565864563], [-0.2608034610748291, -0.40805116295814514], [-0.4958721399307251, -0.26431113481521606], [-0.21211618185043335, -0.38622453808784485], [-0.49097567796707153, -0.38689103722572327], [-0.29656893014907837, -0.3982450067996979], [-0.5061854124069214, -0.29522737860679626], [-0.19089674949645996, -0.312770277261734], [-0.3866199851036072, -0.20948490500450134], [-0.12915736436843872, -0.27371785044670105], [-0.30733999609947205, -0.186032235622406], [-0.15766698122024536, -0.164814293384552], [-0.33021798729896545, -0.07221251726150513], [-0.16198641061782837, -0.13520735502243042], [-0.3360101282596588, 1.0], [-0.18188196420669556, -0.07597750425338745], [-0.3248758614063263, -0.0915951132774353], [-0.10910540819168091, -0.047596871852874756], [-0.19368615746498108, 1.0]]}'
        feature_set = FeatureSet_v1_1(gamestring_1)

if __name__ == '__main__':
    unittest.main()
    #TestStringMethods().validate_rotation()


