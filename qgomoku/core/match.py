
import numpy as np
from qgomoku.core.board import Board, Player, BitBoard, BitBoardCache
from qgomoku.core.game_record import GameRecord

# a match between two ai's
class Match:
    def __init__(self, mind1, mind2, size=9, random_seed=42, opening_moves=10, draw_point=50, trivialize=False):
        cache = BitBoardCache("../cache/9-magics", size=9, win_chain_length=5, force_build_win_checks=False)
        self.board = BitBoard(cache, size=9, win_chain_length=5)

        random_state = np.random.RandomState(random_seed)

        if not trivialize:
            # randomize the board a bit
            for j in range(random_state.randint(0, opening_moves)):
                self.board.make_random_move()
        else:
            self.board.set_to_one_move_from_win()

        self.players = {Player.FIRST: mind1, Player.SECOND: mind2}
        self.game_record = GameRecord.create(self.board)

    def play(self):
        print(self.board)
        current_player = self.board.get_player_to_move()

        while True:
            move, current_q, best_q = self.players[current_player].make_move(self.board)

            self.game_record.add_move(move, current_q, best_q)
            self.board.move(move)
            print(self.board.pprint(), move, current_q, best_q)

            if self.board.game_over():
                self.game_record.set_winner(self.board.get_winning_player())
                break

            current_player = current_player.other()

        return self.game_record.export()
