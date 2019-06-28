from time import time
import random
import numpy as np
import math
from typing import List, Dict, Tuple, Sequence

def set_bit(value, bit):
    return value | (1 << bit)

def clear_bit(value, bit):
    return value & ~(1 << bit)

def get_bit(value, bit):
    return (value & (1 << bit)) > 0

def array_of_set_bits(value):
    set_bits = []
    for i in range(msb(value)):
        if get_bit(value, i):
            set_bits.append(i)
    return set_bits

def has_consecutive_bits(value, num_consecutive):
    mask = (1 << num_consecutive) - 1
    while value > 0:
        if value & mask == mask:
            return True
        value = int(value / 2)
    return False


def num_1s(value):
    num_set = 0
    for i in range(msb(value)):
        if get_bit(value, i):
            num_set += 1
    return num_set

def msb(n):
    if n == 0:
        return 0
    msb = 0
    while n > 0:
        n = int(n / 2)
        msb += 1
    return msb

def bitstring_with(bits_to_set: List[int]):
    bitstring = 0
    for location in bits_to_set:
        bitstring = set_bit(bitstring, int(location))
    return bitstring

def random_bits(length, num_approx_set):
    bitstring = 0
    for i in random.sample(range(length), num_approx_set):
        bitstring = set_bit(bitstring, i)
    return bitstring

def use_magic(bitstring, magic, shift, force_overflow = (1 << 81) - 1):
    return ((bitstring * magic) & force_overflow) >> shift

def validate_magic(magic, shift, positions, length=81):
    unique = set()
    for i in range(1 << len(positions)):
        bitstring = 0
        for j in range(len(positions)):
            if get_bit(i, j):
                bitstring = set_bit(bitstring, positions[j])
        magic_result = use_magic(bitstring, magic, shift, force_overflow=(1 << length) - 1)
        if magic_result in unique:
            return False
        unique.add(magic_result)
    return len(unique) == 1 << len(positions)

def bitstrings_that_produce(magic, shift, value, positions, length=81):
    results = []
    for i in range(1 << len(positions)):
        bitstring = 0
        for j in range(len(positions)):
            if get_bit(i, j):
                bitstring = set_bit(bitstring, positions[j])
        magic_result = use_magic(bitstring, magic, shift, force_overflow=(1 << length) - 1)
        if magic_result == value:
            results.append(bitstring)
    return results

# @deprecated
def create_magic(bit_positions, length):
    mask = bitstring_with(bit_positions)
    num_set = num_1s(mask, length)
    print('Attempting magic of', num_set)
    iters = 1
    while True:
        random_set = random.randint(5, int(5 + np.log(iters) / np.log(2)))
        magic = random_bits(length, random_set)
        # arbitrarily shift back since python has no overflow
        iters += 1
        force_overflow = (1 << length) - 1
        shift = (length - num_set)
        if ((mask * magic) & force_overflow) >> shift != (1 << num_set) - 1:
            continue
        #if passed % 1000 == 0:
        #    print(passed)
        # check for collisions
        used = np.zeros((1 << num_set))
        for i in range(1 << num_set):
            # assemble pretend board with the bit combination represented by i, but spread out to their proper indices
            pretend_board = 0
            for j in range(num_set):
                if get_bit(i, j):
                    pretend_board = set_bit(pretend_board, bit_positions[j])
            magicked = ((pretend_board * magic) & force_overflow) >> shift
            if used[magicked] == 1:
                break
            used[magicked] = 1
            if i == (1 << num_set) - 1:
                assert validate_magic(magic, shift, bit_positions)
                return magic, shift

def run_python_bitting():
    start = time()

    bitstring = 1 << 81

    for i in range(1000000):
        set_bit(bitstring, 5)

    print(time() - start)

def check_overflow():
    a = random.randint(1 << 50, 1 << 55)
    b = random.randint(1 << 50, 1 << 55)
    numpy_int = np.array([a]) * np.array([b])

if __name__ == "__main__":
    assert has_consecutive_bits(15, 4)
