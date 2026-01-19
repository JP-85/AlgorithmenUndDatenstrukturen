# (1) Berechnen Sie die ersten 35 Fibonacci-Zahlen mir rekursiver und iterativer Methode. Speichern Sie die für jede Zahl benötigte Rechenzeit und geben Sie das Ergebnis im Vergleich der beiden Algorithmen als Grafik aus.

import numpy as np
import matplotlib.pyplot as plt

def fib(n):
    a, b = 0, 1
    for i in range(n):
        a, b = b, a+b
    return a

def fib_rek(n):

    if n > 2:
        f = fib(n-1) + fib(n-2)
    elif n == 0:
        return 0
    else:
        return 1
    return f

if __name__ == '__main__':
    for i in range(10):
        print(fib(i))

    print()

    for i in range(10):
        print(fib_rek(i))