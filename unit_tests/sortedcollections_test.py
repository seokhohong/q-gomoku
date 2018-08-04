from sortedcontainers import sortedset
import random

class Foo:
    def __init__(self):
        self.value = random.random()


leaves = sortedset.SortedSet([Foo()], key=lambda x: x.value)

for i in range(10):
    leaves.add(Foo())

newFoo = Foo()
leaves.add(newFoo)

leaves.remove(newFoo)