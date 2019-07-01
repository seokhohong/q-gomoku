
from qgomoku.learner.game_to_features import FeatureSet_v1_1, FeatureBoard_v1_1
from qgomoku.core.board import Player
import copy

# ThoughtBoard manages rapid move/unmove sequences and computes available moves and win conditions
# feature_system_cls is the class that gets wrapped into making the features
class ThoughtBoard:
    def __init__(self, board, feature_system_cls):
        self.board = board
        self.feature_system = feature_system_cls(board)
        # root is used for queries that have a movelist (class functions stateless-ly)
        self.root_board = copy.deepcopy(board)
        self.root_feature_system = feature_system_cls(board)
        self.delta = 0

    def make_move(self, move):
        self.board.move(move)
        self.feature_system.move(move)
        self.delta += 1

    def get_hard_q(self):
        winning_player = self.board.get_winning_player()
        if winning_player == Player.FIRST:
            return 1
        elif winning_player == Player.SECOND:
            return -1
        return 0

    # this call assumes that the game is in a concluding state after move_list
    def get_hard_q_after(self, move_list):
        return self._call_function_with_board_state(self.root_board.get_winning_player, move_list)

    def get_game_status(self):
        return self.board.game_status()

    def game_over(self):
        return self.board.game_over()

    # checks whether the game is over after move_list
    def game_over_after(self, move_list):
        return self._call_function_with_board_state(self.root_board.game_over, move_list)

    def get_available_move_vector_after(self, move_list):
        move_vector = self.root_feature_system.get_init_available_move_vector()
        for move in move_list:
            move_vector[move] = 0
        return move_vector

    # puts the internal featureboard into a state as defined by move_list, then operates this function
    def _call_function_with_feature_state(self, func, move_list):
        for move in move_list:
            self.root_feature_system.move(move)

        value = func()

        for i in range(len(move_list)):
            self.root_feature_system.unmove()

        return value

    # puts the internal featureboard into a state as defined by move_list, computes win/loss conditions
    # then runs function
    def _call_function_with_board_state(self, func, move_list):
        for move in move_list:
            self.root_board.blind_move(move)

        self.root_board.compute_game_state()
        value = func()

        for i in range(len(move_list)):
            self.root_board.unmove()

        return value

    def unmove(self):
        self.board.unmove()
        self.feature_system.unmove()
        self.delta -= 1
        assert self.delta >= 0

    def get_q_features(self):
        return self.feature_system.get_q_features()

    # does not check if move_list is a valid set of moves
    def get_q_features_after(self, move_list):
        return self._call_function_with_feature_state(self.root_feature_system.get_q_features, move_list)

    def get_p_features_after(self, move_list):
        return self._call_function_with_feature_state(self.root_feature_system.get_p_features, move_list)

    # moves the board forward by move_list moves
    def make_moves(self, move_list):
        for move in move_list:
            self.make_move(move)

    # returns a defensive copy
    def available_moves(self):
        return set(self.board.get_available_moves())

    # not sure if faster to undo or copy
    def reset(self):
        for i in range(self.delta):
            self.unmove()