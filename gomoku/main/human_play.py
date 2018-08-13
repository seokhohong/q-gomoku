
from learner import pqmind
from learner import pexp_mind
from core.board import Board
from numpy.random import RandomState
from core.optimized_minimax import PExpNode
import numpy as np

import random

SIZE = 9
CHANNELS = 4

if __name__ == "__main__":
    mind = pexp_mind.PExpMind(size=SIZE, alpha=0.2, init=False, channels=CHANNELS)
    mind.load_net('../models/9_4_4')


    def expanding_p(depth, p):
        return np.logical_or(np.logical_or(
            np.logical_and(depth < 4, p > -5),
            np.logical_and(depth < 6, p > -4),
            np.logical_and(depth < 8, p > -4)),
            np.logical_and(depth < np.inf, p > -3)
        )


    def permissive_expansion(depth):
        if depth < 2:
            return np.inf
        if depth < 8:
            return 5
        return 3

    mind.define_policies(expanding_p, permissive_expansion, convergence_count=5)

    board = Board(size=SIZE, win_chain_length=5)

    # randomize the board a bit
    for j in range(random.randint(0, int(SIZE * 3))):
        board.make_random_move()

    print(board.guide_print())

    while True:
        if board.player_to_move == Board.FIRST_PLAYER:
            inp = input("Input your move (i.e. \"3 5\"): ")
            if len(inp.split(' ')) != 2:
                print('Incorrect number of coordinates, please try again!')
                continue
            x, y = inp.split(' ')
            try:
                x = int(x)
                y = int(y)
            except:
                print('Please input Numbers!')
                continue
            if x < 0 or x >= SIZE or y < 0 or y >= SIZE:
                print('Out of bounds!')
                continue
            if (x, y) not in board.available_moves:
                print('Invalid Move!')
                continue
            result = board.move(x, y)
            print(board.guide_print())
        else:
            print('Computer is thinking...')

            possible_moves, root_node = mind.pvs_best_moves(board,
                                                required_depth=6,
                                                max_iters=20,
                                                k=SIZE ** 3)

            mind.save_root(board, root_node)
            picked_move, picked_node = possible_moves[0]
            # add training example assuming best move
            move, best_node = possible_moves[0]
            best_q = best_node.q
            print(" ")
            print(move, 'Q:', best_q)

            if best_q > PExpNode.MAX_MODEL_Q:
                print('Computer Resigns!')
                break

            board.move(move[0], move[1])
            print(board.guide_print())

        if board.game_drawn():
            print("DRAW!")
            break

        if board.game_won():
            if board.player_to_move == 1:
                print('COMPUTER WINS!')
            else:
                print('YOU WIN!')
            break
