
from qgomoku.learner.pexp_mind_v3 import PExpMind_v3
from qgomoku.core.board import Board, BitBoard, BitBoardCache, Player
from qgomoku.core.minimax import PExpNode
import numpy as np
import keras

import random

import os

# bug with python on macos
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def play():

    SIZE = 9



    mind = PExpMind_v3(size=9, init=False, search_params={
        'max_iterations': 10,
        'min_child_p': -7,
        'p_batch_size': 1 << 10,
        'num_pv_expand': 25,
        'q_fraction': 1
    })
    mind.load_net('../../models/v3_' + str(0))

    cache = BitBoardCache("../cache/9-magics", size=9, win_chain_length=5, force_build_win_checks=False)
    board = BitBoard(cache, size=9, win_chain_length=5)

    # randomize the board a bit
    for j in range(random.randint(0, int(SIZE * 2))):
        board.make_random_move()
    #board.move_coord(4, 4)

    print(board.pprint())

    while True:
        if board.get_player_to_move() == Player.FIRST:
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
            index = board._transformer.coordinate_to_index(x, y)
            if not board.is_move_available(index):
                print('Invalid Move!')
                continue
            result = board.move_coord(x, y)
            print(board)
        else:
            print('Computer is thinking...')
            print(board._ops)

            move, current_q, best_q = mind.make_move(board)
            print(" ")
            print(move, 'Q:', best_q)

            #if best_q > PExpNode.MAX_MODEL_Q :
            #    print('Computer Resigns!')
            #    break

            board.move(move)
            print(board)

        if board.game_over():
            if board.get_winning_player() == Player.FIRST:
                print('COMPUTER WINS!')
            elif board.get_winning_player() == Player.SECOND:
                print('YOU WIN!')
            print("DRAW!")
            break


if __name__ == "__main__":
    play()