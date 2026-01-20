import random
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ----------------------------
# Bubblesort
# ----------------------------


def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]


# ----------------------------
# Parameter
# ----------------------------

sizes = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000]
times = []

# ----------------------------
# Laufzeitmessung
# ----------------------------

for n in sizes:
    array = random.sample(range(1, 10 * n), n)

    start = time.perf_counter()
    bubble_sort(array)
    t = time.perf_counter() - start

    times.append(t)
    print(f"n={n}, Zeit={t:.4f} s")

# Vermutung: T(n) = a * n^2


def quadratic(n, a):
    return a * n**2


params, _ = curve_fit(quadratic, np.array(sizes), np.array(times))
a_fit = params[0]

# Fit-Kurve
n_fit = np.linspace(min(sizes), max(sizes), 200)
t_fit = quadratic(n_fit, a_fit)

# ----------------------------
# Grafik
# ----------------------------

plt.figure("Bubblesort")
plt.plot(sizes, times, "o", label="Messdaten")
plt.plot(n_fit, t_fit, "-", label=f"Kurve: a·n², a={a_fit:.2e}")

# Textausgabe der Messwerte im Plot
for n, t in zip(sizes, times):
    plt.annotate(
        f"n={n}\n{t:.4f} s",
        (n, t),
        textcoords="offset points",
        xytext=(5, 5),
        fontsize=8,
    )

plt.xlabel("n")
plt.ylabel("Rechenzeit [s]")
plt.title("Bubblesort - Laufzeitverhalten")
plt.legend()
plt.grid(True)
plt.show()
