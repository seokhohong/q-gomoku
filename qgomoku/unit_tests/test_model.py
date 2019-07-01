
import unittest
from qgomoku.learner.pexp_mind_v3 import PExpMind_v3, PEvenSearch, ThoughtBoard
from qgomoku.learner.pexp_node_v3 import PExpNodeV3
from qgomoku.core.board import Board, BoardTransform, GameState, BitBoard, BitBoardCache
from qgomoku.learner.game_to_features import FeatureBoard_v1_1
from qgomoku.core.minimax import MoveList

import numpy as np


class TestMind(unittest.TestCase):

    def test_pnet(self):

        mind = PExpMind_v3(size=9, init=False, search_params=None)
        mind.load_net('../../models/v3_' + str(0))

        cache = BitBoardCache("../cache/9-magics", size=9, win_chain_length=5, force_build_win_checks=False)
        board = BitBoard(cache, size=9, win_chain_length=5)

        fboard = FeatureBoard_v1_1(board)

        for move in [76, 65, 58, 27, 40]:
            board.move(move)
            fboard.move(move)

        print(board)

        mind.make_move(board)
        predictions = mind.policy_est.predict([np.array([fboard.get_q_features()])])[0]
        self.assertEqual(np.argmax(predictions), 49)
        print(sorted(enumerate(predictions), key=lambda x : x[1], reverse=True))


if __name__ == '__main__':
    unittest.main()
    # TestMind().test_longer_loss()


