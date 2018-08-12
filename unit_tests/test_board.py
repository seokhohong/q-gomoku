from core.board import Board

board = Board(size=5, win_chain_length=4)
board.move(0, 0)
assert(board.get_matrix()[0, 0, 0] == 0)
assert(board.get_matrix()[0, 1, 0] == 1)
assert(board.get_rotated_matrices()[2][4][0][1] == 1)
board.move(0, 1)
board.move(1, 1)
assert(board.chain_length(1, 1, -1, 0) == 1)
assert(board.chain_length(1, 1, -1, -1) == 2)
assert(board.chain_length(0, 0, 1, 1) == 2)
board.move(1, 0)
board.move(2, 2)
board.move(2, 3)
board.move(3, 3)
board.unmove()
board.move(3, 3)
assert(board.chain_length(3, 3, 1, 1) == 1)
print(board.pprint())
assert(board.get_matrix()[2, 2, Board.TURN_INFO_INDEX] == -1)
assert(board.game_won())


