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

    CHAIN_DIRECTIONS = 4
    NUM_PLAYERS = 2

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

        # memory of blocked directions
        self.chain_block_memory = []

        # how many of the same stones are strung together
        # stores an integer
        self.chain_matrix = np.zeros((self.size, self.size, Board.NUM_PLAYERS, Board.CHAIN_DIRECTIONS))
        self.chain_matrix.fill(0)

        # whether the chain in a particular direction is blocked
        # 1 indicates free on both, 0 indicates blocked on one side, -1 indicates blocked on both
        self.chain_block_matrix = np.zeros((self.size, self.size, Board.NUM_PLAYERS, Board.CHAIN_DIRECTIONS))
        self.chain_block_matrix.fill(1)

        class Direction:
            def __init__(self, x, y, index):
                self.x = x
                self.y = y
                self.index = index

        # direction mapping
        self.directions = {
            'HORIZONTAL': Direction(-1, 0, 0),
            'SLASH': Direction(-1, 1, 1),
            'BACKSLASH': Direction(1, 1, 2),
            'VERTICAL': Direction(0, 1, 3)
        }

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
        self.chain_block_memory.append(np.copy(self.chain_block_matrix))

        self.available_moves.remove((x, y))
        self.compute_game_state()
        self.update_chains(x, y)

        self.flip_player_to_move()


    def unmove(self):
        previous_move = self.ops.pop()
        self.matrix[previous_move.x, previous_move.y, previous_move.player] = 0
        self.matrix[previous_move.x, previous_move.y, Board.NO_PLAYER] = 1

        self.chain_matrix = self.chain_length_memory.pop()
        self.chain_block_matrix = self.chain_block_memory.pop()

        self.available_moves.add((previous_move.x, previous_move.y))
        self.flip_player_to_move()
        self.game_state = GameState.NOT_OVER

    # +1 for self, -1 for other
    def get_matrix(self):
        turn_matrix = np.zeros((self.size, self.size, 1))
        turn_matrix.fill(self.player_to_move)
        return np.concatenate((self.matrix,
                               self.chain_matrix.reshape(self.size, self.size, -1),
                               self.chain_block_matrix.reshape(self.size, self.size, -1),
                               turn_matrix), axis=2)


    def get_rotated_matrices(self):
        matrix = self.get_matrix()
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
        last_move = utils.peek_stack(self.ops)
        if last_move:
            last_x, last_y = last_move.x, last_move.y
            # check win
            if self.chain_length(last_x, last_y, self.directions['HORIZONTAL']) >= self.win_chain_length\
                    or self.chain_length(last_x, last_y, self.directions['SLASH']) >= self.win_chain_length \
                    or self.chain_length(last_x, last_y, self.directions['BACKSLASH']) >= self.win_chain_length \
                    or self.chain_length(last_x, last_y, self.directions['VERTICAL']) >= self.win_chain_length:
                    self.game_state = GameState.WON
                    return
            if len(self.ops) == self.size ** 2:
                self.game_state = GameState.DRAW
                return
        self.game_state = GameState.NOT_OVER
        self.state_computed = True

    def in_bounds(self, x, y):
        return 0 <= x < self.size and 0 <= y < self.size

    def center_stone(self, x, y):
        if abs(self.matrix[x, y, 1] - 1) < 1E-6:
            return Board.FIRST_PLAYER
        if abs(self.matrix[x, y, 2] - 1) < 1E-6:
            return Board.SECOND_PLAYER
        return Board.NO_PLAYER

    # search in all directions from move made and update chain lengths there
    def update_chains(self, center_x, center_y):
        # for center only
        for key, chain_dir in self.directions.items():
            self.chain_length(center_x, center_y, chain_dir)
        # walking direction
        for direction in self.directions.values():
            for step in range(1, self.size):
                step_x = direction.x * step
                step_y = direction.y * step
                # stepped x, y
                this_x = center_x + step_x
                this_y = center_y + step_y
                # search in all directions and update
                if 0 <= this_x < self.size and 0 <= this_y < self.size:
                    # chain check direction
                    self.chain_length(this_x, this_y, direction)
                else:
                    break

        for direction in self.directions.values():
            for step in range(1, self.size):
                step_x = direction.x * step * -1
                step_y = direction.y * step * -1
                # stepped x, y
                this_x = center_x + step_x
                this_y = center_y + step_y
                # search in all directions and update
                if 0 <= this_x < self.size and 0 <= this_y < self.size:
                    # chain check direction
                    self.chain_length(this_x, this_y, direction)
                else:
                    break

    def directional_chain_length(self, center_x, center_y, delta_x, delta_y, center_stone, chain_positions):
        chain_length = 0

        blocked = 1
        for step in range(1, self.win_chain_length):
            step_x = delta_x * step
            step_y = delta_y * step
            # stepped x, y
            this_x = center_x + step_x
            this_y = center_y + step_y
            if 0 <= this_x < self.size and 0 <= this_y < self.size:
                if self.matrix[this_x, this_y, center_stone] == 1:
                    chain_length += 1
                    chain_positions.append([this_x, this_y])
                elif self.matrix[this_x, this_y, Board.NO_PLAYER] == 1:
                    blocked = 0
                    break
                else:
                    break
            else:
                break

        return chain_length, blocked

    # does both directions
    def chain_length(self, center_x, center_y, direction):
        # other spots other than center that are part of the chain
        delta_x, delta_y = direction.x, direction.y
        chain_positions = [(center_x, center_y)]

        center_stone = self.center_stone(center_x, center_y)
        chain_player = 0 if center_stone == 1 else 1

        if center_stone == Board.NO_PLAYER:
            return 0

        chain_length_1, blocked_1 = self.directional_chain_length(center_x, center_y, delta_x, delta_y, center_stone, chain_positions)
        chain_length_2, blocked_2 = self.directional_chain_length(center_x, center_y, -delta_x, -delta_y, center_stone, chain_positions)
        chain_length = chain_length_1 + chain_length_2 + 1

        # blocked math is funky, but works out to produce an indicator
        self.chain_block_matrix[center_x, center_y, chain_player, direction.index] = -(blocked_1 + blocked_2) + 1

        for x, y in chain_positions:
            if chain_length > self.chain_matrix[x, y, chain_player, direction.index]:
                self.chain_matrix[x, y, chain_player, direction.index] = chain_length

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
            move = utils.peek_stack(self.ops)
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

    def guide_print(self):
        def display_char(x, y):
            move = utils.peek_stack(self.ops)
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
        board_string = " "
        for i in range(0, self.size):
            board_string += " " + str(i)
        for i in range(0, self.size):
            board_string += "\n" + str(i)
            for j in range(self.size):
                board_string += "|" + display_char(j, i)
            board_string += "|"
        return board_string

    def __str__(self):
        return self.pprint()

