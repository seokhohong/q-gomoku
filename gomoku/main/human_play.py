
from learner import pqmind
from learner import pexp_mind
from core import board
from numpy.random import RandomState
from core.optimized_minimax import PExpNode

import random

SIZE = 9
CHANNELS = 4

if __name__ == "__main__":
    mind = pexp_mind.PExpMind(size=SIZE, alpha=0.2, init=False, channels=CHANNELS)
    mind.load_net('../models/9_4_4')
    round_board = board.Board(size=SIZE, win_chain_length=5)

    # randomize the board a bit
    for j in range(random.randint(0, int(SIZE * 3))):
        round_board.make_random_move()

    print(round_board.guide_print())

    while True:
        if round_board.player_to_move == board.Board.FIRST_PLAYER:
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
            if (x, y) not in round_board.available_moves:
                print('Invalid Move!')
                continue
            result = round_board.move(x, y)
            print(round_board.guide_print())
        else:
            print('Computer is thinking...')
            possible_moves, root_node = mind.pvs_best_moves(round_board,
                                                required_depth=5,
                                                max_iters=50,
                                                k=SIZE ** 2,
                                                p_threshold=-4)
            mind.save_root(round_board, root_node)
            picked_move, picked_node = possible_moves[0]
            # add training example assuming best move
            move, best_node = possible_moves[0]
            best_q = best_node.q
            print(" ")
            print(move, 'Q:', best_q)

            if best_q > PExpNode.MAX_MODEL_Q:
                print('Computer Resigns!')
                break

            round_board.move(move[0], move[1])
            print(round_board.guide_print())

        if round_board.game_drawn():
            print("DRAW!")
            break

        if round_board.game_won():
            if round_board.player_to_move == 1:
                print('COMPUTER WINS!')
            else:
                print('YOU WIN!')
            break
