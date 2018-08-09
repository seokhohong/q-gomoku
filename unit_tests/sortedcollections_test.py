from sortedcontainers import SortedSet
import random
import cProfile
from queue import PriorityQueue

def mass_init():
    for i in range(100000):
        d = PriorityQueue()
        #SortedSet(key=lambda x: x.move_goodness)


if __name__ == "__main__":
    #run()

    import cProfile, pstats
    from io import StringIO

    pr = cProfile.Profile()
    pr.enable()
    mass_init()
    pr.disable()
    s = StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())