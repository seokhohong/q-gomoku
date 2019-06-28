from qgomoku.core.game_record import GameRecord
from qgomoku.core.board import BitBoard, BitBoardCache
from qgomoku.learner.pexp_mind_v3 import PExpMind_v3, PEvenSearch

import dill
import pickle

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def debug_replay():
    gamestring = '{"initial_state": {"size": "9", "win_chain_length": "5", "boardstring": "000000000000000000000000000000000000000000000000000000000000000000000000000000000", "player_to_move": "1"}, "moves": [[5, 5], [4, 2], [6, 5], [3, 5], [3, 4], [3, 2], [2, 2], [2, 3], [7, 5], [8, 5], [5, 6], [5, 3], [4, 4], [3, 3], [4, 3], [4, 6], [5, 4], [7, 2], [6, 2], [3, 1], [2, 0], [1, 4], [7, 8], [6, 7], [7, 6], [8, 7], [7, 4], [7, 7], [5, 8], [5, 7], [4, 7], [5, 1], [8, 3]], "winning_player": 2, "q_assessments": [[0.06138022989034653, -0.1255047768354416], [-0.05615045502781868, 0.019648950546979904], [-0.10922026634216309, -0.1784544438123703], [-0.16054783761501312, -0.01633911207318306], [-0.115797258913517, -0.17180128395557404], [-0.023201871663331985, -0.06263655424118042], [-0.002079140394926071, -0.011096451431512833], [0.09861882030963898, -0.04954760894179344], [-0.04238772392272949, -0.04221750795841217], [-0.14244608581066132, -0.08665527403354645], [-0.05948565900325775, -0.11776845157146454], [-0.037942755967378616, -0.00023085251450538635], [-0.2299414873123169, -0.10206032544374466], [-0.10555435717105865, -0.07033582031726837], [-0.06934680044651031, -0.1318686604499817], [-0.08796344697475433, -0.19723573327064514], [-0.2305147498846054, -0.3860225975513458], [-0.28055188059806824, -0.4155530631542206], [-0.3582364320755005, -0.3649487793445587], [-0.3654245436191559, -0.44607529044151306], [-0.3764168918132782, -0.2709214687347412], [-0.36448243260383606, -0.43351683020591736], [-0.4696137309074402, -1.0], [-0.13120876252651215, -1.0], [-0.2660411596298218, -1.0], [-0.3870473802089691, -1.0], [-0.39597445726394653, -1.0], [-0.5461440682411194, -1.0], [-0.6547334790229797, -1.0], [-0.5473154187202454, -1.0], [-0.3565058410167694, -1.0], [-0.5941867232322693, -1.0], [-0.5827884078025818, -1.0]]}'
    GameRecord.parse(gamestring).replay()

def debug_game():
    mind = PExpMind_v3(size=9, init=False, search_params=None)
    mind.load_net('../../models/voldmaster_' + str(0))

    cache = BitBoardCache("../cache/9-magics", size=9, win_chain_length=5, force_build_win_checks=False)
    board = BitBoard(cache, size=9, win_chain_length=5)

    moves = [
        (7, 1), (0, 1),
        (7, 3), (1, 7),
        (7, 4), (3, 1),
        (3, 2), (4, 1),
        (3, 4), (6, 1),
        (6, 3), (6, 2),
        (5, 2), (8, 7),
        (7, 7), (6, 8),
        (5, 8), (1, 6),
        (2, 7)
    ]
    for move in moves:
        board.move_coord(*move)

    print(board)
    searcher = PEvenSearch(board, mind.policy_est, mind.value_est,
                           search_params={
                               'max_iterations': 10,
                               'min_child_p': -7,
                               'p_batch_size': 1 << 10,
                               'q_fraction': 1
                           })

    searcher.run(3)
    print(searcher.get_pv().calculate_pv_order(), [19, 46, 10])

if __name__ == "__main__":
    import cProfile, pstats
    from io import StringIO

    pr = cProfile.Profile()
    pr.enable()
    debug_game()
    pr.disable()
    s = StringIO()
    sortby = 'tottime'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())
