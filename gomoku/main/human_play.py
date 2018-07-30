
from learner import pqmind
from core import detail_board

import random

SIZE = 7
CHANNELS = 20

if __name__ == "__main__":
    mind = pqmind.PQMind(size=SIZE, alpha=0.2, init=False, channels=20)
    mind.load_net('../models/7_channel20')
    round_board = detail_board.Board(size=SIZE, win_chain_length=5)

    # randomize the board a bit
    for j in range(random.randint(0, 10)):
        round_board.make_random_move()

    print(round_board.guide_print())

    while True:
        if round_board.player_to_move == 1:
            inp = input("Input your move (i.e. \"3,5\"): ")
            if len(inp.split(',')) != 2:
                print('Incorrect number of coordinates, please try again!')
                continue
            x, y = inp.split(',')
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
            move, best_q = mind.pvs(round_board,
                                    epsilon=0,
                                    verbose=True,
                                    max_depth=20,
                                    max_iters=15,
                                    k=SIZE ** 2)
            print(" ")
            print(move, 'Q:', best_q)

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
