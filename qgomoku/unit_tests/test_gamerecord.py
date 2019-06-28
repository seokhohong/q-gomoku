

import unittest
from qgomoku.core.board import Board
from qgomoku.core.game_record import GameRecord

class TestStringMethods(unittest.TestCase):
    def test_parse(self):
        board = Board(size=9)
        board.move_coord(5, 5)
        record = GameRecord.create(board)
        record.add_move(35, 0, 0)
        record.set_winner(1)
        with self.assertRaises(ValueError):
            record.add_move(45, 0, 0)
        board = Board(size=9)
        record2 = GameRecord.create(board)
        record2.add_move(45, 0, 0)
        record2.add_move(35, 0, 0)
        record2.set_winner(1)

    def test_complexparse(self):
        sample_gamestring = """{"initial_state": "9.001001000000000000000000000000000001000000000002000000000000000000002200000000000.1", 
                "moves": [[4, 5], [7, 4], [7, 3], [6, 3], [8, 5], [4, 3], [0, 4], [0, 3], [5, 4], [3, 3], [5, 3], [4, 1], [3, 0], [2, 3], [1, 3], 
                [3, 4], [2, 5], [3, 5], [3, 2], [6, 1], [7, 0], [6, 4], [6, 2], [7, 7], [7, 8], [4, 4], [3, 6], [5, 1], [1, 8], [2, 7], [3, 1], 
                [2, 2], [5, 5], [8, 1], [7, 1], [4, 2], [4, 0], [2, 4], [1, 5], [6, 0]], "winning_player": 2, "q_assessments": 
                [[0.0019790641963481903, -0.24267181754112244], [-0.20177625119686127, -0.5521407723426819], [-0.20921818912029266, 
                -0.6166988611221313], [-0.31880027055740356, -0.5630054473876953], [-0.31782427430152893, -0.5711942315101624], 
                [-0.4583733081817627, -0.6889514923095703], [-0.4173961579799652, -0.5759556293487549], [-0.5566257834434509, -0.6832323670387268], 
                [-0.4021470844745636, -0.7034830451011658], [-0.4920775294303894, -0.7756547331809998], [-0.38935402035713196, -0.740565299987793], 
                [-0.48536548018455505, -0.7954807281494141], [-0.5474658608436584, -0.840348482131958], [-0.5121245384216309, -0.8764957189559937], 
                [-0.5996554493904114, -0.9171169400215149], [-0.5762278437614441, -0.6989874839782715], [-0.7694527506828308, -0.701777994632721], 
                [-0.7339904308319092, -0.626954197883606], [-0.7092840671539307, -0.6621876955032349], [-0.686124324798584, -0.699820876121521], 
                [-0.6693198680877686, -0.6919920444488525], [-0.6735756993293762, -0.7122229337692261], [-0.5495494604110718, -0.6875355243682861], 
                [-0.6090625524520874, -0.5493432283401489], [-0.5003388524055481, -0.6300103664398193], [-0.6499016880989075, -0.5552431344985962], 
                [-0.6546697020530701, -0.5464909672737122], [-0.5511935353279114, -1.0], [-0.5547133088111877, -1.0], 
                [-0.3656817376613617, -1.0], [-0.5807197690010071, -1.0], [-0.5815728902816772, -1.0], [-0.6264187693595886, -1.0], 
                [-0.5814750790596008, -1.0], [-0.4301098883152008, -1.0], [-0.45098164677619934, -1.0], [-0.5229567885398865, -1.0], 
                [-0.4639306664466858, -1.0], [-0.5296437740325928, -1.0], [-0.5666452050209045, -1.0]]}"""
        record = GameRecord.parse(sample_gamestring)
        self.assertEqual(record.moves[0], [4, 5])

if __name__ == '__main__':
    unittest.main()

