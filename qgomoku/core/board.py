from collections import defaultdict

import numpy as np
import random
from enum import Enum

import os
import pickle

from qgomoku.util import utils, bitops


class GameState(Enum):
    WON = 1
    DRAW = 2
    NOT_OVER = 3

class BoardTransform:
    NUM_ROTATIONS = 8
    def __init__(self, size):
        self._size = size
        self._cached_point_rotations = defaultdict(list)
        self._cache_rotations()

    # Returns rotations and mirrors of the board state
    # This is important for teaching the convolution layers about rotational invariance
    def get_rotated_matrices(self, matrix):
        transposition_axes = (1, 0, 2)
        assert matrix.shape[0] == self._size
        return [
            matrix,
            np.transpose(matrix, axes=transposition_axes),
            np.rot90(matrix),
            np.transpose(np.rot90(matrix), axes=transposition_axes),
            np.rot90(matrix, 2),
            np.transpose(np.rot90(matrix, 2), axes=transposition_axes),
            np.rot90(matrix, 3),
            np.transpose(np.rot90(matrix, 3), axes=transposition_axes)
        ]

    def get_rotated_points(self, point):
        return self._cached_point_rotations[point]

    def coordinate_to_index(self, x, y):
        return x * self._size + y

    def index_to_coordinate(self, index):
        return int(index / self._size), int(index % self._size)

    # Precomputes some useful rotation values
    def _cache_rotations(self):
        indices = np.array(range(self._size ** 2)).reshape(self._size, self._size, 1)
        for i, matrix in enumerate(self.get_rotated_matrices(indices)):
            # iterate through the matrices
            for x in range(self._size):
                for y in range(self._size):
                    # where ever you see the matrix[x, y] value, indices[x, y] is where it originally started
                    self._cached_point_rotations[matrix[x, y, 0]].append(indices[x, y, 0])

    def in_bounds(self, x, y):
        return 0 <= x < self._size and 0 <= y < self._size

# Board class represents the state of the game

# board perception will always be from the perspective of Player 1
# Q will always be from the perspective of Player 1 (Player 1 Wins = Q = 1, Player -1 Wins, Q = -1)

class Player(Enum):
    NONE = 0
    FIRST = 1
    SECOND = 2
    def other(self):
        if self == Player.SECOND:
            return Player.FIRST
        elif self == Player.FIRST:
            return Player.SECOND
        return Player.NONE
    def flip(self, times):
        if times % 2 == 0:
            return self
        return self.other()
    @staticmethod
    def get(value):
        for player in Player:
            if player.value == value:
                return player
        return Player.NONE


class BitBoardCache:
    DELTA_SETS = [((-1, -1), (1, 1)), ((0, -1), (0, 1)), ((-1, 0), (1, 0)), ((1, -1), (-1, 1))]
    def __init__(self, filename, size=9, win_chain_length=5,
                 verbose=False, force_build_magics=False, force_build_win_checks=False):
        self._size = size
        self._transformer = BoardTransform(size=9)
        self._win_chain_length = win_chain_length
        self._filename = filename
        self._verbose = verbose
        loaded = self._load()

        #if force_build_magics or not loaded:
        #    self._build_magics()
        if force_build_win_checks or not loaded:
            self._build_win_checks()

        self._save()

    def _load(self):
        if os.path.exists(self._filename):
            if self._verbose:
                print('Loading Bitboard cache from', self._filename)
            with open(self._filename, 'rb') as f:
                self._delta_masks, self._win_checks = pickle.load(f)
                return True
        return False

    def _marked_locations(self, index, deltas):
        board = Board(size=self._size, win_chain_length=self._win_chain_length)
        all_locations = []
        for delta in deltas:
            all_locations.extend(board._check_locations[(index, *delta)])
        return sorted(all_locations)

    def _build_magics(self):
        print('Bitboard not found, creating')

        self._magics = []
        for i in range(self._size ** 2):
            self._magics.append([])
            for j, deltaset in enumerate(self.DELTA_SETS):
                all_locations = self._marked_locations(i, deltaset)
                magic, mask = bitops.create_magic(all_locations, self._size ** 2)
                self._magics[i].append((magic, mask))
                print("Made Magic for ", i, j, all_locations, magic, mask)

    def _build_win_checks(self):
        self._delta_masks = []
        self._win_checks = []
        for move in range(self._size ** 2):
            self._delta_masks.append([])
            self._win_checks.append([])
            for deltaset_index, deltaset in enumerate(self.DELTA_SETS):
                self._delta_masks[move].append([])
                self._win_checks[move].append(set())
                marked_locations = self._marked_locations(move, deltaset)
                self._delta_masks[move][deltaset_index] = bitops.bitstring_with(marked_locations)
                # generate every combination
                for i in range(1 << len(marked_locations)):
                    # check for 4 consecutive set bits
                    if bitops.has_consecutive_bits(i, 4):
                        marked_location_set_bits = np.array(marked_locations)[bitops.array_of_set_bits(i)]
                        win_check = bitops.bitstring_with(marked_location_set_bits)
                        self._win_checks[move][deltaset_index].add(win_check)


    def check_win(self, bitstring, played_move):
        masks = self._delta_masks[played_move]
        win_checks = self._win_checks[played_move]
        if bitstring & masks[0] in win_checks[0]:
            return True
        if bitstring & masks[1] in win_checks[1]:
            return True
        if bitstring & masks[2] in win_checks[2]:
            return True
        if bitstring & masks[3] in win_checks[3]:
            return True
        return False


    def _save(self):
        if os.path.exists(os.path.dirname(self._filename)):
            with open(self._filename, 'wb') as f:
                pickle.dump((self._delta_masks, self._win_checks), f)

# High performance board relying on bit manipulation to track game status
class BitBoard:
    def __init__(self, cache, size=9, win_chain_length=5, draw_point=None):
        self._size = size
        self._cache = cache
        self._draw_point = draw_point

        self._win_chain_length = win_chain_length
        # stack of (index, player) tuple
        self._ops = []

        self._transformer = BoardTransform(size=self._size)
        self._game_state = GameState.NOT_OVER
        self._winning_player = Player.NONE

        # board representations
        self.bitstrings = [0, 0]

    def get_spot(self, index):
        if self.bitstrings[0] & (1 << index):
            return Player.FIRST
        elif self.bitstrings[1] & (1 << index):
            return Player.SECOND
        return Player.NONE

    def get_spot_coord(self, x, y):
        index = self._transformer.coordinate_to_index(x, y)
        return self.get_spot(index)

    # returns a Player enum
    def get_player_to_move(self):
        return Player.FIRST if len(self._ops) % 2 == 0 else Player.SECOND

    # does not check whether the game ends with each move
    def blind_move(self, move):
        player_to_move = -1 * len(self._ops) % 2
        self.bitstrings[player_to_move] = self.bitstrings[player_to_move] | (1 << move)
        self._ops.append(move)

    def unmove(self):
        move = self._ops.pop()
        player_to_move = -1 * len(self._ops) % 2
        self.bitstrings[player_to_move] = self.bitstrings[player_to_move] & ~(1 << move)
        self._game_state = GameState.NOT_OVER

    def is_winning_move(self, move):
        player_to_move = -1 * len(self._ops) % 2
        return self._cache.check_win(self.bitstrings[player_to_move], move)

    def move(self, move):
        player_to_move = -1 * len(self._ops) % 2
        self.blind_move(move)
        if self._cache.check_win(self.bitstrings[player_to_move], move):
            self._game_state = GameState.WON
            self._winning_player = self.get_player_to_move().other()

    def game_status(self):
        return self._game_state

    def move_coord(self, x, y):
        self.move(self._transformer.coordinate_to_index(x, y))

    def get_winning_player(self):
        return self._winning_player

    def game_over(self):
        return self._game_state != GameState.NOT_OVER or len(self._ops) == self._size ** 2

    def game_approx_drawn(self):
        return len(self._ops) >= self._draw_point

    def get_size(self):
        return self._size

    # prefer get_available_moves_bitstring
    def get_available_moves(self):
        moves = set()
        for i in range(self._size ** 2):
            if self.is_move_available(i):
                moves.add(i)
        return moves

    def get_last_move(self):
        return utils.peek_stack(self._ops)

    # for testing purposes
    def set_to_one_move_from_win(self):
        self.move_coord(0, 0)
        self.move_coord(0, 1)
        self.move_coord(1, 0)
        self.move_coord(1, 1)
        self.move_coord(2, 0)
        self.move_coord(2, 1)
        self.move_coord(3, 0)
        self.move_coord(3, 1)

    def make_random_move(self):
        self.move(random.choice(list(self.get_available_moves())))

    def is_move_available(self, index):
        return ~(self.bitstrings[0] | self.bitstrings[1]) & (1 << index) > 0

    # simplest matrix representation of the board state
    def export(self):
        flatmatrix = np.zeros((self._size, self._size))
        for x in range(self._size):
            for y in range(self._size):
                index = self._transformer.coordinate_to_index(x, y)
                flatmatrix[x, y] = self.get_spot(index).value

        return {'size': str(self._size),
                   'win_chain_length': str(self._win_chain_length),
                    'boardstring': ''.join([str(elem) for elem in flatmatrix.astype(np.int32).reshape(-1)]),
                    'player_to_move': str(self.get_player_to_move().value)}

    # takes a board string and recreates a game state from it (quite slow)
    @classmethod
    def create_board_from_specs(Board, size, win_chain_length, boardstring):
        board = Board(size=size, win_chain_length=win_chain_length)
        player_1_moves = []
        player_2_moves = []
        reshaped_board = np.array([int(elem) for elem in boardstring]).reshape((size, size))
        for i in range(size):
            for j in range(size):
                index = board._transformer.coordinate_to_index(i, j)
                if reshaped_board[i][j] == Player.FIRST.value:
                    player_1_moves.append(index)
                elif reshaped_board[i][j] == Player.SECOND.value:
                    player_2_moves.append(index)

        for i in range(len(player_1_moves)):
            board.blind_move(player_1_moves[i])
            if i < len(player_2_moves):
                board.blind_move(player_2_moves[i])

        return board

    def display_char(self, index, lastmove_highlight):
        move = utils.peek_stack(self._ops)
        if move is not None:
            was_last_move = (index == move)
            if bitops.get_bit(self.bitstrings[0], index):
                if was_last_move and lastmove_highlight:
                    return 'X'
                return 'x'
            elif bitops.get_bit(self.bitstrings[1], index):
                if was_last_move and lastmove_highlight:
                    return 'O'
                return 'o'
        return ' '

    def pprint(self, lastmove_highlight=True):
        board_string = ""
        for i in range(0, self._size):
            board_string += "\n"
            for j in range(self._size):
                board_string += "|" + self.display_char(self._transformer.coordinate_to_index(j, i), lastmove_highlight)
            board_string += "|"
        return board_string

    def __str__(self):
        return self.pprint()

class Board:
    # channels
    NO_PLAYER_INDEX = 0
    FIRST_PLAYER_INDEX = 1
    SECOND_PLAYER_INDEX = 2
    TURN_INFO_INDEX = 3

    DELTAS = [(-1, 0), (1, 0), (-1, 1), (1, -1), (1, 1), (-1, -1), (0, 1), (0, -1)]

    STONE_PRESENT_VALUE = 1
    STONE_ABSENT_VALUE = 0

    def __init__(self, size=9, win_chain_length=5, draw_point=None):
        self._size = size

        self._matrix = np.zeros((self._size ** 2), dtype=np.int)

        self._win_chain_length = win_chain_length
        # stack of (index, player) tuple
        self._ops = []
        self._player_to_move = Player.FIRST
        self._game_state = GameState.NOT_OVER
        self._available_moves = set()
        for i in range(self._size ** 2):
            self._available_moves.add(i)

        self._num_moves = 0

        # number of plays at which it's a draw
        self.draw_point = draw_point if draw_point else self._size ** 2

        self._transformer = BoardTransform(size=self._size)

        self._precompute()

    # for testing purposes
    def set_to_one_move_from_win(self):
        self.move_coord(0, 0)
        self.move_coord(0, 1)
        self.move_coord(1, 0)
        self.move_coord(1, 1)
        self.move_coord(2, 0)
        self.move_coord(2, 1)
        self.move_coord(3, 0)
        self.move_coord(3, 1)

    def unmove(self):
        previous_move = self._ops.pop()

        self._matrix[previous_move] = 0

        self._available_moves.add(previous_move)
        self._flip_player_to_move()
        self._game_state = GameState.NOT_OVER

        self._num_moves -= 1

    def get_spot(self, index):
        if self._matrix[index] == 1:
            return Player.FIRST
        elif self._matrix[index] == 2:
            return Player.SECOND
        return Player.NONE

    def get_spot_coord(self, x, y):
        index = self._transformer.coordinate_to_index(x, y)
        return Player.get(self._matrix[index])

    def get_player_to_move(self):
        return self._player_to_move

    def get_player_last_move(self):
        return self.get_player_to_move().other()

    def get_last_move(self):
        return utils.peek_stack(self._ops)

    def get_size(self):
        return self._size

    # Places a stone at x, y for the next player's turn
    # Does not compute whether the game has completed or not (performance optimization)
    def blind_move(self, move):
        assert self._game_state is GameState.NOT_OVER

        self._ops.append(move)

        self._matrix[move] = self._player_to_move.value

        self._available_moves.remove(move)
        self._flip_player_to_move()

        self._num_moves += 1

    # Places a stone at x, y for the next player's turn
    # Computes whether game has concluded and if so, who the winner is
    def move_coord(self, x, y):
        self.blind_move(self._transformer.coordinate_to_index(x, y))
        self.compute_game_state()
        assert np.sum(self._matrix) >= len(self._ops)

    def move(self, index):
        self.blind_move(index)
        self.compute_game_state()

    # Executes a random valid move
    def make_random_move(self):
        self.move(random.choice(list(self._available_moves)))

    def _flip_player_to_move(self):
        self._player_to_move = self._player_to_move.other()

    # returns None if game has not concluded, True if the last move won the game, False if draw
    # frequently called function, needs to be optimized
    def compute_game_state(self):
        last_move = utils.peek_stack(self._ops)
        if last_move is not None:
            if self.chain_length(last_move, -1, 0) + self.chain_length(last_move, 1, 0) >= self._win_chain_length + 1\
                    or self.chain_length(last_move, -1, 1) + self.chain_length(last_move, 1, -1) >= self._win_chain_length + 1\
                    or self.chain_length(last_move, 1, 1) + self.chain_length(last_move, -1, -1) >= self._win_chain_length + 1\
                    or self.chain_length(last_move, 0, 1) + self.chain_length(last_move, 0, -1) >= self._win_chain_length + 1:
                self._game_state = GameState.WON
                return
            if len(self._ops) == self._size ** 2:
                self._game_state = GameState.DRAW
                return
        self._game_state = GameState.NOT_OVER

    # lists all the positions needed to check for chain length
    def _precompute(self):
        self._check_locations = {}
        for delta_pair in Board.DELTAS:
            for i in range(self.get_size() ** 2):
                coords = []
                x, y = self._transformer.index_to_coordinate(i)
                for j in range(1, self._win_chain_length):
                    step_x = delta_pair[0] * j
                    step_y = delta_pair[1] * j
                    if self._transformer.in_bounds(x + step_x, y + step_y):
                        coords.append(self._transformer.coordinate_to_index(x + step_x, y + step_y))
                self._check_locations[(i, delta_pair[1], delta_pair[0])] = tuple(coords)

    def chain_length(self, center, delta_x, delta_y):
        center_stone = self._matrix[center]
        i = 1
        for location in self._check_locations[(center, delta_x, delta_y)]:
            if self._matrix[location] == center_stone:
                i += 1
            else:
                break
        return i

    def is_move_available(self, index):
        return index in self._available_moves

    # does not make a defensive copy
    def get_available_moves(self):
        return self._available_moves

    def game_won(self):
        return self._game_state == GameState.WON

    def game_status(self):
        return self._game_state

    # probably drawn, cheap check
    def game_assume_drawn(self):
        return len(self._ops) >= self.draw_point

    def game_over(self):
        return self._game_state != GameState.NOT_OVER

    def get_winning_player(self):
        if self.game_over():
            return self._player_to_move.other()
        return Player.NONE

    # simplest matrix representation of the board state
    def export(self):
        flatmatrix = np.zeros((self._size, self._size))
        for x in range(self._size):
            for y in range(self._size):
                index = self._transformer.coordinate_to_index(x, y)
                flatmatrix[x, y] = self._matrix[index]

        return {'size': str(self._size),
                   'win_chain_length': str(self._win_chain_length),
                    'boardstring': ''.join([str(elem) for elem in flatmatrix.astype(np.int32).reshape(-1)]),
                    'player_to_move': str(self._player_to_move.value)}

    # takes a board string and recreates a game state from it (quite slow)
    @classmethod
    def create_board_from_specs(Board, size, win_chain_length, boardstring):
        board = Board(size=size, win_chain_length=win_chain_length)
        player_1_moves = []
        player_2_moves = []
        reshaped_board = np.array([int(elem) for elem in boardstring]).reshape((size, size))
        for i in range(size):
            for j in range(size):
                index = board._transformer.coordinate_to_index(i, j)
                if reshaped_board[i][j] == Player.FIRST.value:
                    player_1_moves.append(index)
                elif reshaped_board[i][j] == Player.SECOND.value:
                    player_2_moves.append(index)

        for i in range(len(player_1_moves)):
            board.blind_move(player_1_moves[i])
            if i < len(player_2_moves):
                board.blind_move(player_2_moves[i])

        return board

    @classmethod
    def load(Board, item_dict):
        return Board.create_board_from_specs(int(item_dict['size']),
                                             int(item_dict['win_chain_length']),
                                             item_dict['boardstring'])

    def display_char(self, index, lastmove_highlight):
        move = utils.peek_stack(self._ops)
        if move is not None:
            was_last_move = (index == move)
            if self._matrix[index] == Player.FIRST:
                if was_last_move and lastmove_highlight:
                    return 'X'
                return 'x'
            elif self._matrix[index] == Player.SECOND:
                if was_last_move and lastmove_highlight:
                    return 'O'
                return 'o'
        return ' '

    def pprint(self, lastmove_highlight=True):
        board_string = ""
        for i in range(0, self._size):
            board_string += "\n"
            for j in range(self._size):
                board_string += "|" + self.display_char(self._transformer.coordinate_to_index(i, j), lastmove_highlight)
            board_string += "|"
        return board_string

    def guide_print(self, lastmove_highlight=True):
        board_string = " "

        for i in range(0, self._size):
            board_string += " " + str(i)
        for i in range(0, self._size):
            board_string += "\n" + str(i)
            for j in range(self._size):
                board_string += "|" + self.display_char(self._transformer.coordinate_to_index(j, i), lastmove_highlight)
            board_string += "|"
        return board_string

    def __str__(self):
        return self.pprint()


class TranspositionTable:
    def __init__(self):
        self._transposition_dict = {}
        self._transposition_hits = 0

    def get(self, key):
        if key in self._transposition_dict:
            self._transposition_hits += 1
            return self._transposition_dict[key]
        return None

    def put(self, key, node):
        self._transposition_dict[key] = node

    def get_num_hits(self):
        return self._transposition_hits