import numpy as np

from qgomoku.core.board import Player, BitBoard, BitBoardCache
from qgomoku.core.game_record import GameRecord


# a match between two ai's
class Match:
    def __init__(self, mind1, mind2, size=9, random_seed=42, opening_moves=10, draw_point=50,
                 trivialize=False, verbose=False):
        cache = BitBoardCache("../cache/9-magics", size=size, win_chain_length=5, force_build_win_checks=False)
        self.board = BitBoard(cache, size=size, win_chain_length=5)

        random_state = np.random.RandomState(random_seed)

        if not trivialize:
            # randomize the board a bit
            for j in range(random_state.randint(0, opening_moves)):
                self.board.make_random_move()
        else:
            self.board.set_to_one_move_from_win()

        # close to draw state
        # for move in [49, 40, 50, 29, 51, 52, 47, 48, 41, 31, 32, 23, 39, 68, 42, 58, 38, 33, 65, 56, 57, 73, 67, 37, 64, 66, 43, 59, 21, 22, 13, 25, 24, 44, 30, 16, 3, 12, 2, 1, 5, 6, 14, 17, 35, 7, 71, 28, 19, 34, 60, 78, 77, 62, 46, 54, 55, 36, 20, 45, 63, 18, 27, 53, 15, 9, 0, 69, 10, 75, 74, 8, 79, 72, 80]:
        #    self.board.move(move)

        self.players = {Player.FIRST: mind1, Player.SECOND: mind2}
        self.game_record = GameRecord.create(self.board)
        self._verbose = verbose

    def play(self):
        print(self.board)
        current_player = self.board.get_player_to_move()

        while True:
            move, current_q, best_q = self.players[current_player].make_move(self.board)

            self.game_record.add_move(move, current_q, best_q)
            self.board.move(move)
            print(self.board.pprint(), move, current_q, best_q)
            if self._verbose:
                print(self.board._ops)

            if self.board.game_over():
                self.game_record.set_winner(self.board.get_winning_player())
                print(self.game_record.get_winning_player())
                break

            current_player = current_player.other()

        return self.game_record.export()
