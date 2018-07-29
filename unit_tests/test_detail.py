from core.detail_board import Board
import numpy as np
import random

board = Board(size=5, win_chain_length=4)
board.move(2, 2)
board.move(2, 3)

assert board.chain_block_matrix[2, 2, 0, board.directions['BACKSLASH'].index] == 1

assert board.chain_block_matrix[2, 2, 0, board.directions['HORIZONTAL'].index] == 1

assert board.chain_block_matrix[2, 3, 1, board.directions['VERTICAL'].index] == 0
assert board.chain_block_matrix[2, 2, 0, board.directions['VERTICAL'].index] == 0

board.move(1, 4)
board.move(4, 4)
board.move(4, 3)
board.move(1, 3)

print(board)

assert(board.chain_block_matrix[1, 3, 1, board.directions['HORIZONTAL'].index] == 1)
assert(board.chain_length(1, 3, board.directions['VERTICAL']) == 1)
assert(board.chain_length(2, 3, board.directions['HORIZONTAL']) == 2)
assert(board.chain_block_matrix[4, 4, 1, board.directions['HORIZONTAL'].index] == 0)
assert(board.chain_block_matrix[4, 4, 1, board.directions['VERTICAL'].index] == -1)

board.move(3, 3)

print(board)

assert(board.chain_block_matrix[3, 3, 0, board.directions['HORIZONTAL'].index] == -1)
assert(board.chain_block_matrix[4, 3, 0, board.directions['HORIZONTAL'].index] == -1)
assert(board.chain_block_matrix[3, 3, 0, board.directions['VERTICAL'].index] == 1)
assert(board.chain_block_matrix[4, 3, 0, board.directions['VERTICAL'].index] == 0)

assert(board.chain_block_matrix[3, 3, 0, board.directions['BACKSLASH'].index] == 0)

board.move(1, 2)

print(board)

assert(board.chain_block_matrix[1, 2, 0, board.directions['SLASH'].index] == 1)
assert(board.chain_length(1, 2, board.directions['SLASH']) == 1)
assert(board.chain_length(1, 2, board.directions['BACKSLASH']) == 2)
assert(board.chain_block_matrix[1, 2, 0, board.directions['BACKSLASH'].index] == 1)

board.move(1, 1)
board.move(3, 4)

print(board)
assert(board.chain_length(2, 2, board.directions['BACKSLASH']) == 3)
assert(board.chain_block_matrix[2, 2, 0, board.directions['SLASH'].index] == 0)

board.unmove()
board.unmove()
board.unmove()

assert(board.chain_block_matrix[3, 3, 0, board.directions['HORIZONTAL'].index] == -1)
assert(board.chain_block_matrix[4, 3, 0, board.directions['HORIZONTAL'].index] == -1)
assert(board.chain_block_matrix[3, 3, 0, board.directions['VERTICAL'].index] == 1)
assert(board.chain_block_matrix[4, 3, 0, board.directions['VERTICAL'].index] == 0)

assert(board.chain_block_matrix[3, 3, 0, board.directions['BACKSLASH'].index] == 0)

print(board.get_matrix().shape)

for i in range(10):
    board = Board(size=7, win_chain_length=5)
    while True:
        num_steps = random.randint(0, 10)
        board.make_random_move()
        steps_taken = 0
        for j in range(num_steps):
            if not board.game_over():
                board.make_random_move()
                steps_taken += 1
        for j in range(steps_taken):
            board.unmove()
        if board.game_won():
            assert(np.max(board.chain_matrix) >= 5)
            #assert(np.max(board.get_matrix()[:, :, 7: 11]) >= 5)
            #assert(np.max(board.chain_matrix[:, :, 0, :]) >= 5)
            #assert(np.max(board.chain_matrix[:, :, 0, :]) < 5)
            break
        elif board.game_drawn():
            print('draw')
            break

# check proper transposition
rotated_matrices = board.get_rotated_matrices()

for rot in rotated_matrices:
    for i in range(3):
        assert(np.sum(rot[:, :, i]) == np.sum(board.get_matrix()[:, :, i]))