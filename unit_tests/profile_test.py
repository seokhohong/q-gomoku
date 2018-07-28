import cProfile


from core.detail_board import Board
from learner.pqmind import PQMind


SIZE = 7

def run():
    mind = PQMind(size=SIZE, alpha=0.5, init=True, channels=19)

    board = Board(size=SIZE, win_chain_length=5)

    mind.pvs(board, max_iters=10, k=SIZE ** 2, max_depth=25)

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