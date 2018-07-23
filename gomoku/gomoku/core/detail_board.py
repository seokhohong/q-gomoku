from enum import Enum
from collections import defaultdict

import numpy as np
import random

def peek_stack(list):
    if len(list) == 0:
        return None
    else:
        return list[-1]

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
        # three channels
        self.matrix = np.zeros((self.size, self.size, 3))
        self.matrix[:, :, Board.NO_PLAYER].fill(1)

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

        # memory of how each move changed the chain lengths
        self.chain_length_memory = []
        self.chain_matrix = np.zeros((self.size, self.size))
        self.chain_matrix.fill(Board.NO_PLAYER)

        # whether current game_state is accurate
        self.state_computed = False

    def _mark_not_computed(self):
        self.state_computed = False

    # makes a move and performs feature computation and game end check
    def move(self, x, y):
        assert (self.game_state is GameState.NOT_OVER)
        # record move
        self.ops.append(Move(self.player_to_move, x, y))
        self.matrix[x, y, self.player_to_move] = 1
        self.matrix[x, y, self.NO_PLAYER] = 0
        self.chain_length_memory.append(np.copy(self.chain_matrix))

        self.available_moves.remove((x, y))
        self.compute_game_state()

        self.flip_player_to_move()


    def unmove(self):
        previous_move = self.ops.pop()
        self.matrix[previous_move.x, previous_move.y, previous_move.player] = 0
        self.matrix[previous_move.x, previous_move.y, Board.NO_PLAYER] = 1

        previous_chain = self.chain_length_memory.pop()
        self.chain_matrix = previous_chain

        self.available_moves.add((previous_move.x, previous_move.y))
        self.flip_player_to_move()
        self.game_state = GameState.NOT_OVER

    # +1 for self, -1 for other
    def get_matrix(self, as_player):

        if as_player == Board.FIRST_PLAYER:
            matrix = np.copy(self.matrix)

        matrix = -np.copy(self.matrix)

        return np.concatenate((matrix, np.copy(self.chain_matrix).reshape(self.size, self.size, 1)), axis=2)


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

    def make_random_move(self):
        move_x, move_y = random.choice(list(self.available_moves))
        self.move(move_x, move_y)

    def flip_player_to_move(self):
        if self.player_to_move == Board.FIRST_PLAYER:
            self.player_to_move = Board.SECOND_PLAYER
        else:
            self.player_to_move = Board.FIRST_PLAYER

    # returns None if game has not concluded, True if the last move won the game, False if draw
    # frequently called function, needs to be optimized
    def compute_game_state(self):
        last_move = peek_stack(self.ops)
        if last_move:
            last_x, last_y = last_move.x, last_move.y
            # check win
            if self.chain_length(last_x, last_y, -1, 0, last_move.player) >= self.win_chain_length\
                    or self.chain_length(last_x, last_y, -1, 1, last_move.player) >= self.win_chain_length \
                    or self.chain_length(last_x, last_y, 1, 1, last_move.player) >= self.win_chain_length \
                    or self.chain_length(last_x, last_y, 0, 1, last_move.player) >= self.win_chain_length:
                    self.game_state = GameState.WON
                    return
            if len(self.ops) == self.size ** 2:
                self.game_state = GameState.DRAW
                return
        self.game_state = GameState.NOT_OVER
        self.state_computed = True

    def in_bounds(self, x, y):
        return 0 <= x < self.size and 0 <= y < self.size

    # does both directions
    def chain_length(self, center_x, center_y, delta_x, delta_y, center_stone):
        # other spots other than center that are part of the chain
        chain_positions = [(center_x, center_y)]
        if center_stone == Board.NO_PLAYER:
            return 0
        chain_length = 1
        for step in range(1, self.win_chain_length):
            step_x = delta_x * step
            step_y = delta_y * step
            if 0 <= center_x + step_x < self.size and 0 <= center_y + step_y < self.size and \
                    self.matrix[center_x + step_x, center_y + step_y, center_stone] == 1:
                chain_length += 1
                chain_positions.append([center_x + step_x, center_y + step_y])
            else:
                break
        # other direction
        for step in range(1, self.win_chain_length):
            step_x = -delta_x * step
            step_y = -delta_y * step
            if 0 <= center_x + step_x < self.size and 0 <= center_y + step_y < self.size and \
                    self.matrix[center_x + step_x, center_y + step_y, center_stone] == 1:
                chain_length += 1
                chain_positions.append([center_x + step_x, center_y + step_y])
            else:
                break

        for x, y in chain_positions:
            if chain_length > self.chain_matrix[x, y]:
                self.chain_matrix[x, y] = chain_length

        return chain_length


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

    def pprint(self):
        def display_char(x, y):
            move = peek_stack(self.ops)
            if move:
                was_last_move = (x == move.x and y == move.y)
                if self.matrix[x, y, Board.FIRST_PLAYER] == 1:
                    if was_last_move:
                        return 'X'
                    return 'x'
                elif self.matrix[x, y, Board.SECOND_PLAYER] == 1:
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

    def __str__(self):
        return self.pprint()

