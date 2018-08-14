from collections import defaultdict

import numpy as np
import random

from util import utils

class Move:
    def __init__(self, player, x, y):
        self.player = player
        self.x = x
        self.y = y


class GameState:
    WON = 1
    DRAW = 2
    NOT_OVER = 3

# Board class represents the state of the game

# board perception will always be from the perspective of Player 1
# Q will always be from the perspective of Player 1 (Player 1 Wins = Q = 1, Player -1 Wins, Q = -1)

class Board:
    _player_to_move: int
    NO_PLAYER = 0
    FIRST_PLAYER = 1
    SECOND_PLAYER = 2
    TURN_INFO_INDEX = 3

    def __init__(self, size=5, win_chain_length=3):
        self._size = size

        # three for No Player, Player 1, Player 2, one for turn index
        self._matrix = np.zeros((self._size, self._size, 4), dtype=np.int)
        self._matrix[:, :, Board.NO_PLAYER].fill(1)

        # tracks which player played which spot (optimization
        self._which_stone = np.zeros((self._size, self._size), dtype=np.int)

        self._win_chain_length = win_chain_length
        # stack of (x, y, player) tuple
        self._ops = []
        self._player_to_move = 1
        self._game_state = GameState.NOT_OVER
        self._available_moves = set()
        for i in range(self._size):
            for j in range(self._size):
                self._available_moves.add((i, j))

        self.cached_point_rotations = defaultdict(list)
        self._cache_rotations()

        self._num_moves = 0

    def unmove(self):
        previous_move = self._ops.pop()

        self._matrix[previous_move.x, previous_move.y, previous_move.player] = 0
        self._matrix[previous_move.x, previous_move.y, Board.NO_PLAYER] = 1
        self._which_stone[previous_move.x, previous_move.y] = Board.NO_PLAYER

        self._available_moves.add((previous_move.x, previous_move.y))
        self._flip_player_to_move()
        self._game_state = GameState.NOT_OVER

        self._num_moves -= 1

    def get_matrix(self):
        if self._player_to_move == Board.FIRST_PLAYER:
            self._matrix[:, :, Board.TURN_INFO_INDEX].fill(1)
        else:
            self._matrix[:, :, Board.TURN_INFO_INDEX].fill(-1)
        return np.copy(self._matrix)

    def get_player_to_move(self):
        return self._player_to_move

    def get_size(self):
        return self._size

    # Returns rotations and mirrors of the board state
    # This is important for teaching the convolution layers about rotational invariance
    def get_rotated_matrices(self):
        transposition_axes = (1, 0, 2)
        matrix = self.get_matrix()
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

    # Precomputes some useful rotation values
    def _cache_rotations(self):
        indices = np.array(range(self._size ** 2)).reshape(self._size, self._size)
        for matrix in [
                indices,
                indices.transpose(),
                np.rot90(indices),
                np.rot90(indices).transpose(),
                np.rot90(indices, 2),
                np.rot90(indices, 2).transpose(),
                np.rot90(indices, 3),
                np.rot90(indices, 3).transpose()
            ]:
            for x in range(self._size):
                for y in range(self._size):
                    self.cached_point_rotations[matrix[x, y]].append(indices[x, y])

    def coordinate_to_index(self, x, y):
        return x * self._size + y

    def get_rotated_point(self, index):
        return self.cached_point_rotations[index]

    # Places a stone at x, y for the next player's turn
    # Does not compute whether the game has completed or not (performance optimization)
    def blind_move(self, x, y):
        assert(self._game_state is GameState.NOT_OVER)
        self._ops.append(Move(self._player_to_move, x, y))

        self._matrix[x, y, self._player_to_move] = 1
        self._which_stone[x, y] = self._player_to_move

        self._matrix[x, y, Board.NO_PLAYER] = 0

        self._available_moves.remove((x, y))
        self._flip_player_to_move()

        self._num_moves += 1

    # Places a stone at x, y for the next player's turn
    # Computes whether game has concluded and if so, who the winner is
    def move(self, x, y):
        self.blind_move(x, y)
        self.compute_game_state()

    # Executes a random valid move
    def make_random_move(self):
        move_x, move_y = random.choice(list(self._available_moves))
        self.move(move_x, move_y)

    def _flip_player_to_move(self):
        if self._player_to_move == Board.FIRST_PLAYER:
            self._player_to_move = Board.SECOND_PLAYER
        else:
            self._player_to_move = Board.FIRST_PLAYER

    # returns None if game has not concluded, True if the last move won the game, False if draw
    # frequently called function, needs to be optimized
    def compute_game_state(self):
        last_move = utils.peek_stack(self._ops)
        if last_move:
            last_x, last_y = last_move.x, last_move.y
            if self.chain_length(last_x, last_y, -1, 0) + self.chain_length(last_x, last_y, 1, 0) >= self._win_chain_length + 1\
                    or self.chain_length(last_x, last_y, -1, 1) + self.chain_length(last_x, last_y, 1, -1) >= self._win_chain_length + 1\
                    or self.chain_length(last_x, last_y, 1, 1) + self.chain_length(last_x, last_y, -1, -1) >= self._win_chain_length + 1\
                    or self.chain_length(last_x, last_y, 0, 1) + self.chain_length(last_x, last_y, 0, -1) >= self._win_chain_length + 1:
                self._game_state = GameState.WON
                return
            if len(self._ops) == self._size ** 2:
                self._game_state = GameState.DRAW
                return
        self._game_state = GameState.NOT_OVER

    def _in_bounds(self, x, y):
        return 0 <= x < self._size and 0 <= y < self._size

    def chain_length(self, center_x, center_y, delta_x, delta_y):
        center_stone = self._which_stone[center_x, center_y]
        for chain_length in range(1, self._win_chain_length + 1):
            step_x = delta_x * chain_length
            step_y = delta_y * chain_length
            if not(0 <= center_x + step_x < self._size and 0 <= center_y + step_y < self._size and
                   self._matrix[center_x + step_x, center_y + step_y, center_stone] == 1):
                break
        return chain_length

    def is_move_available(self, x, y):
        return (x, y) in self._available_moves

    # runs a full compute
    def game_won(self):
        return self._game_state == GameState.WON

    # probably drawn, cheap check
    def game_drawn(self):
        return len(self._ops) == self._size ** 2

    def game_over(self):
        return self._game_state != GameState.NOT_OVER

    def pprint(self):
        def display_char(x, y):
            move = utils.peek_stack(self._ops)
            if move:
                was_last_move = (x == move.x and y == move.y)
                if self._matrix[x, y, Board.FIRST_PLAYER] == 1:
                    if was_last_move:
                        return 'X'
                    return 'x'
                elif self._matrix[x, y, Board.SECOND_PLAYER] == 1:
                    if was_last_move:
                        return 'O'
                    return 'o'
            return ' '
        board_string = ""
        for i in range(0, self._size):
            board_string += "\n"
            for j in range(self._size):
                board_string += "|" + display_char(j, i)
            board_string += "|"
        return board_string

    def guide_print(self):
        def display_char(x, y):
            move = utils.peek_stack(self._ops)
            if move:
                was_last_move = (x == move.x and y == move.y)
                if self._matrix[x, y, Board.FIRST_PLAYER] == 1:
                    if was_last_move:
                        return 'X'
                    return 'x'
                elif self._matrix[x, y, Board.SECOND_PLAYER] == 1:
                    if was_last_move:
                        return 'O'
                    return 'o'
            return ' '
        board_string = " "

        for i in range(0, self._size):
            board_string += " " + str(i)
        for i in range(0, self._size):
            board_string += "\n" + str(i)
            for j in range(self._size):
                board_string += "|" + display_char(j, i)
            board_string += "|"
        return board_string

    def __str__(self):
        return self.pprint()

