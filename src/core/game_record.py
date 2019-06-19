import numpy as np
import json
import pickle

class GameRecord:

    # winning_player values
    DRAW = 0

    def __init__(self, initial_state_string, moves, winning_player, q_assessments):
        self.initial_state = initial_state_string
        self.moves = moves
        self._winning_player = winning_player
        self.q_assessments = q_assessments

    # initial_board_state (Board): initial position of the game
    @classmethod
    def create(cls, initial_board_state):
        return GameRecord(initial_board_state.export_string(), [], GameRecord.DRAW, [])

    @classmethod
    def parse(cls, string):
        obj_dict = json.loads(string)
        return GameRecord(obj_dict['initial_state'],
                          obj_dict['moves'],
                          obj_dict['winning_player'],
                          obj_dict['curr_q_assessments'])

    class PrettyFloat(float):
        def __repr__(self):
            return '%.15g' % self

    def add_move(self, move, curr_q, best_q):
        # assert we have not concluded the game yet
        if self._winning_player != GameRecord.DRAW:
            raise ValueError
        self.moves.append(move)
        self.q_assessments.append((GameRecord.PrettyFloat(curr_q), GameRecord.PrettyFloat(best_q)))

    def set_winner(self, player):
        self._winning_player = player

    def export(self):
        obj_dict = {
            'initial_state': self.initial_state,
            'moves': self.moves,
            'winning_player': self._winning_player,
            'q_assessments': self.q_assessments
        }
        return json.dumps(obj_dict)

