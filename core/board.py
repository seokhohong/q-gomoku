from enum import Enum
from collections import defaultdict

import numpy as np
import random

from util import utils

class Move:
    def __init__(self, player, x, y):
        self.player = player
        self.x = x
        self.y = y


class GameState(Enum):
    WON = 1
    DRAW = 2
    NOT_OVER = 3

class Board:
    player_to_move: int
    NO_PLAYER = 0
    FIRST_PLAYER = 1
    SECOND_PLAYER = -1

    def __init__(self, size=5, win_chain_length=3):
        self.size = size
        self.matrix = np.zeros((self.size, self.size))
        self.matrix.fill(Board.NO_PLAYER)
        self.win_chain_length = win_chain_length
        # store (x, y, player) tuple
        self.ops = []
        self.player_to_move = 1
        self.game_state = GameState.NOT_OVER
        self.available_moves = set()
        for i in range(self.size):
            for j in range(self.size):
                self.available_moves.add((i, j))

        self.cached_point_rotations = defaultdict(list)
        self.cache_rotations()

        # whether current game_state is accurate
        self.state_computed = False

    def _mark_not_computed(self):
        self.state_computed = False

    # lightweight version of move
    def hypothetical_move(self, x, y):
        assert (self.game_state is GameState.NOT_OVER)
        self.ops.append(Move(self.player_to_move, x, y))
        self.matrix[x, y] = self.player_to_move
        self.available_moves.remove((x, y))
        self.flip_player_to_move()
        self._mark_not_computed()

    def unmove(self):
        previous_move = self.ops.pop()
        self.matrix[previous_move.x, previous_move.y] = Board.NO_PLAYER
        self.available_moves.add((previous_move.x, previous_move.y))
        self.flip_player_to_move()
        self.game_state = GameState.NOT_OVER

    # +1 for self, -1 for other
    def get_matrix(self, as_player):
        if as_player == Board.FIRST_PLAYER:
            return np.copy(self.matrix)
        return -np.copy(self.matrix)

    def get_rotated_matrices(self, as_player):
        matrix = self.get_matrix(as_player)
        return [
            matrix,
            matrix.transpose(),
            np.rot90(matrix),
            np.rot90(matrix).transpose(),
            np.rot90(matrix, 2),
            np.rot90(matrix, 2).transpose(),
            np.rot90(matrix, 3),
            np.rot90(matrix, 3).transpose()
        ]

    def cache_rotations(self):
        indices = np.array(range(self.size ** 2)).reshape(self.size, self.size)
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
            for x in range(self.size):
                for y in range(self.size):
                    self.cached_point_rotations[matrix[x, y]].append(indices[x, y])


    def coordinate_to_index(self, x, y):
        return x * self.size + y

    def get_rotated_point(self, index):
        return self.cached_point_rotations[index]

    # deprecate
    def make_move(self, x, y):
        assert(self.game_state is GameState.NOT_OVER)
        self.ops.append(Move(self.player_to_move, x, y))
        self.matrix[x, y] = self.player_to_move
        self.available_moves.remove((x, y))
        self.flip_player_to_move()
        self.compute_game_state()

    def make_random_move(self):
        move_x, move_y = random.choice(list(self.available_moves))
        self.hypothetical_move(move_x, move_y)

    def flip_player_to_move(self):
        if self.player_to_move == Board.FIRST_PLAYER:
            self.player_to_move = Board.SECOND_PLAYER
        else:
            self.player_to_move = Board.FIRST_PLAYER

    # returns None if game has not concluded, True if the last move won the game, False if draw
    def compute_game_state(self):
        last_move = utils.peek_stack(self.ops)
        if last_move:
            last_x, last_y = last_move.x, last_move.y
            max_chain = max(
                self.chain_length(last_x, last_y, -1, 0) + self.chain_length(last_x, last_y, 1, 0),
                self.chain_length(last_x, last_y, -1, 1) + self.chain_length(last_x, last_y, 1, -1),
                self.chain_length(last_x, last_y, 1, 1) + self.chain_length(last_x, last_y, -1, -1),
                self.chain_length(last_x, last_y, 0, 1) + self.chain_length(last_x, last_y, 0, -1),
            )
            if max_chain >= self.win_chain_length + 1:
                self.game_state = GameState.WON
                return
            if self.game_drawn():
                self.game_state = GameState.DRAW
                return
        self.game_state = GameState.NOT_OVER
        self.state_computed = True

    # runs a full compute
    def game_won(self):
        if not self.state_computed:
            self.compute_game_state()
        return self.game_state == GameState.WON

    # probably drawn, cheap check
    def game_drawn(self):
        return len(self.ops) == self.size ** 2

    def game_over(self):
        if not self.state_computed:
            self.compute_game_state()
        return self.game_state != GameState.NOT_OVER

    def in_bounds(self, x, y):
        return x >= 0 and y >= 0 and x < self.size and y < self.size

    def chain_length(self, center_x, center_y, delta_x, delta_y):
        center_stone = self.matrix[center_x, center_y]
        if center_stone == Board.NO_PLAYER:
            return 0
        chain_length = 0
        for step in range(self.size):
            step_x = delta_x * step
            step_y = delta_y * step
            if self.in_bounds(center_x + step_x, center_y + step_y) and \
                    self.matrix[center_x + step_x, center_y + step_y] == center_stone:
                chain_length += 1
            else:
                break
        return chain_length

    def pprint(self):
        def display_char(x, y):
            move = utils.peek_stack(self.ops)
            if move:
                was_last_move = (x == move.x and y == move.y)
                if self.matrix[x, y] == Board.FIRST_PLAYER:
                    if was_last_move:
                        return 'X'
                    return 'x'
                elif self.matrix[x, y] == Board.SECOND_PLAYER:
                    if was_last_move:
                        return 'O'
                    return 'o'
            return ' '
        board_string = ""
        for i in range(0, self.size):
            board_string += "\n"
            for j in range(self.size):
                board_string += "|" + display_char(j, i)
            board_string += "|"
        return board_string

