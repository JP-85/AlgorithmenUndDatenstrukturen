import time
import matplotlib.pyplot as plt

# ----------------------------
# Fibonacci-Funktionen
# ----------------------------
# Fibonacci-Zahlen: F(1) = 1 | F(2) = 1 | F(n) = F(n-1) + F(n-2)


# Rekursive Implementierung -> Zeitkomplexität T(n) ≈ O(2n)
def fib_recursive(n):
    if n <= 2:
        return 1
    return fib_recursive(n - 1) + fib_recursive(n - 2)


# Iterative Implementierung -> Zeitkomplexität T(n) ≈ O(n)
def fib_iterative(n):
    if n <= 2:
        return 1
    a, b = 1, 1
    for _ in range(3, n + 1):
        a, b = b, a + b
    return b


# ----------------------------
# Laufzeitmessung
# ----------------------------

N_MAX = 35  # Festlegung der Größe von n

times_recursive = []
times_iterative = []
fib_values = []

for n in range(1, N_MAX + 1):
    # Rekursiv
    start = time.perf_counter()
    fib_r = fib_recursive(n)
    t_rec = time.perf_counter() - start

    # Iterativ
    start = time.perf_counter()
    fib_i = fib_iterative(n)
    t_it = time.perf_counter() - start

    # Überprüfung: gleiche Ergebnisse, sonst Abbruch
    assert fib_r == fib_i

    fib_values.append(fib_i)
    times_recursive.append(t_rec)
    times_iterative.append(t_it)

# ----------------------------
# Grafik
# ----------------------------

plt.figure("Fibonacci")
plt.semilogy(range(1, N_MAX + 1), times_recursive, "o-", label="rekursiv")
plt.semilogy(range(1, N_MAX + 1), times_iterative, "o-", label="iterativ")
plt.xscale("linear")
plt.yscale("linear")
plt.xlabel("n")
plt.ylabel("Rechenzeit [s]")
plt.title("Fibonacci: rekursiv vs. iterativ")
plt.legend()
plt.grid(True, which="both")
plt.show()
