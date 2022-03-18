
"""
Fibo() class computes elements in a fibonacci sequence.

Methods:
    next(): Returns the next element.
    get(): Returns the current element.
"""
class Fibo(object):
    def __init__(self, first=1, second=1):
        self.v = -1
        self.first = first
        self.second = second

    def next(self):
        self.v, self.first, self.second = self.first, self.second, (self.first + self.second)

        # Must use get() to return the next value
        return self.get()

    def get(self):
        return self.v

class FiboMax(Fibo):
    def __init__(self, limit):
        super().__init__()
        self.limit = limit

    def get(self):
        v = super().get()
        return self.limit if v > self.limit else v

class FiboSplit(Fibo):
    def __init__(self, size):
        super().__init__()
        self.pie = size

    def next(self):
        v = super().next()
        o = None if self.pie < 0 else self.get()
        self.pie -= v
        return o

    def get(self):
        v = super().get()
        return self.pie if v > self.pie else v

def fibonacci(count):
    fibo = Fibo()
    return list(map(lambda _: fibo.next(), range(count)))

def xyfibonacci(x, y, count):
    fibo = Fibo(x, y)
    return list(map(lambda _: fibo.next(), range(count)))

def nfibonacci(limit):
    fibo = FiboMax(limit)
    items = []

    while fibo.get() != limit:
        items.append(fibo.next())

    return items

def sfibonacci(size):
    fibo = FiboSplit(size)
    items = []

    while v := fibo.next():
        items.append(v)

    return items
