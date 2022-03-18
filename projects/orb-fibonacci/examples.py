from fibonacci import fibonacci, nfibonacci, sfibonacci, xyfibonacci

def sequence():
    s = fibonacci(9)
    print([1, 1, 2, 3, 5, 8, 13, 21, 34], '-> Expected')
    print(s, '-> Got')

def xysequence():
    s = xyfibonacci(6, 7, 9)
    print([6, 7, 13, 20, 33, 53, 86, 139, 225], '-> Expected')
    print(s, '-> Got')

def limited_sequence():
    s = nfibonacci(22)
    print([1, 1, 2, 3, 5, 8, 13, 21, 22], '-> Expected')
    print(s, '-> Got')

def fibonacci_uneven_split():
    s = sfibonacci(35)
    print([1, 1, 2, 3, 5, 8, 13, 2], '-> Expected')
    print(s, '-> Got')

def fibonacci_even_split():
    s = sfibonacci(33)
    print([1, 1, 2, 3, 5, 8, 13], '-> Expected')
    print(s, '-> Got')

if __name__ == '__main__':
    sequence()
    xysequence()
    limited_sequence()
    fibonacci_uneven_split()
    fibonacci_even_split()
