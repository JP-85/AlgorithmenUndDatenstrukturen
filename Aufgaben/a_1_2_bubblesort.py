import time
import random
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

def bubblesort(arr):
    n = len(arr)
    i = 0

    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]

def quadratische_funktion(x, a):
    return a * (x**2)

laengen = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
zeiten = []

for n in laengen:
    zufallszahlen = random.sample(range(100000), n)

    start_zeit = time.time()
    bubblesort(zufallszahlen)
    end_zeit = time.time()
    zeiten.append(end_zeit - start_zeit)

    print(f"n={n} fertig")

popt, _ = curve_fit(quadratische_funktion, laengen, zeiten)
faktor_a = popt[0]

print(f"Gefundener Faktor a: {faktor_a}")

x_fit = np.linspace(min(laengen), max(laengen), 100)
y_fit = quadratische_funktion(x_fit, faktor_a)

plt.figure(figsize=(10, 6))
plt.scatter(laengen, zeiten, color="blue", label="Messdaten")
plt.plot(x_fit, y_fit, color="red", linestyle="--", label=f"Fit: T(n) = {faktor_a:.2e} * n²")

plt.xlabel("Anzahl der Elemente n")
plt.ylabel("Zeit in Sekunden")
plt.title("Bubblesort Zeitkomplexität")
plt.legend()
plt.grid(True)

plt.show()





