import time
import matplotlib.pyplot as plt


def fib_rekursiv(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fib_rekursiv(n - 1) + fib_rekursiv(n - 2)


def fib_iterativ(n):
    if n == 0:
        return 0
    if n == 1:
        return 1

    a = 0
    b = 1

    for i in range(2, n + 1):
        res = a + b
        a = b
        b = res
    return b


n_werte = []
zeiten_rekursiv = []
zeiten_iterativ = []

print("Berechnung läuft... in der Ruhe liegt die Kraft...")

for n in range(36):
    n_werte.append(n)

    # Messung Rekursiv
    start_zeit = time.time()
    fib_rekursiv(n)
    end_zeit = time.time()
    zeiten_rekursiv.append(end_zeit - start_zeit)

    # Messung Iterativ
    start_zeit = time.time()
    fib_iterativ(n)
    end_zeit = time.time()
    zeiten_iterativ.append(end_zeit - start_zeit)


plt.plot(n_werte, zeiten_rekursiv, label="Rekursiv", color="red")
plt.plot(n_werte, zeiten_iterativ, label="Iterativ", color="blue")

plt.xlabel("Fibonacci-Zahl n")
plt.ylabel("Zeit in Sekunden")
plt.title("Vergleich: Rekursiv vs. Iterativ")
plt.legend()
plt.grid(True)

plt.show()
