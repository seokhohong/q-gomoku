
from src.core.match import Match
from src.learner import pexp_mind
from src.learner import pexp_mind_v2
import numpy as np

import os
# bug with python on macos
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def init_pexp_mind(size):

    mind = pexp_mind.PExpMind(size=size, init=False, channels=4)
    #mind.load_net('../../trained_models/9_4_4')

    def expanding_p(depth, p):
        return np.logical_or.reduce([
            np.logical_and(depth < 2, p > -6),
            np.logical_and(depth < 4, p > -4),
            np.logical_and(depth < 6, p > -4),
            np.logical_and(depth < np.inf, p > -3)
        ])

    def permissive_expansion(depth):
        if depth < 2:
            return np.inf
        if depth < 8:
            return 5
        return 3

    mind.define_policies(expanding_p, permissive_expansion, convergence_count=5,
                         alpha=0.2, q_exp_batch_size=size ** 2,
                         p_exp_batch_size=size ** 3, required_depth=6, max_iters=20)

    return mind

def init_new_mind(size):

    mind = pexp_mind_v2.PExpMind_v2(size=size, init=True)

    def expanding_p(depth, p):
        return np.logical_or.reduce([
            np.logical_and(depth < 2, p > -6),
            np.logical_and(depth < 4, p > -4),
            np.logical_and(depth < 6, p > -4),
            np.logical_and(depth < np.inf, p > -3)
        ])

    def permissive_expansion(depth):
        if depth < 5:
            return np.inf
        if depth < 8:
            return 5
        return 3

    def full_p(depth, p):
        return np.logical_and(depth < np.inf, p > -np.inf)

    def full_expansion(depth):
        return np.inf

    mind.define_policies(full_p, full_expansion, convergence_count=5,
                         alpha=0.2, q_exp_batch_size=size ** 2,
                         p_exp_batch_size=size ** 3, required_depth=2, max_iters=3)

    return mind

def play():
    size = 9
    player = init_new_mind(size)
    match = Match(player, player, trivialize=True)
    result = match.play()
    print(result)

if __name__ == "__main__":
    play()