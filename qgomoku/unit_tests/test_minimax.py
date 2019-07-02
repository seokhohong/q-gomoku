import unittest

from qgomoku.core.minimax import MoveList


class TestStringMethods(unittest.TestCase):
    def test_movelist_hash(self):
        ml_1 = MoveList((), [])
        ml_1.append((5, 5))
        ml_1.append((3, 5))
        ml_2 = MoveList((), [])
        ml_2.append((3, 5))
        ml_2.append((5, 5))
        assert ml_1.transposition_hash() == ml_2.transposition_hash()


if __name__ == '__main__' and __package__ is None:
    from os import sys, path

    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
