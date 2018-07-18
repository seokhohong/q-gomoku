import cProfile


from core.board import Board
from core import minimax
from learner.pqmind import PQMind


def run():
    mind = PQMind(size=13, alpha=0.5, turn_input=True)

    board = Board(size=13, win_chain_length=5)

    mind.pvs(board, max_iters=25, k=100, max_depth=3)

if __name__ == "__main__":
    import cProfile, pstats
    from io import StringIO

    pr = cProfile.Profile()
    pr.enable()
    run()
    pr.disable()
    s = StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())