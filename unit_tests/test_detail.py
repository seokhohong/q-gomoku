from core.detail_board import Board

board = Board(size=5, win_chain_length=4)
board.move(2, 2)
board.move(2, 3)
board.move(3, 2)

print(board.get_matrix(1))

board.unmove()

print(board.get_matrix(1))
board.pprint()