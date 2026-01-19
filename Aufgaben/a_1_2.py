# (2) Machen Sie sich mit dem Bubblesort Algorithmus bekannt. Erzeugen Sie unsortierte (also zufällige) Arrays mit beliebigen natürlichen Zahlen der Längen (n=500, 1000, 1500, 2000, 2500, 3000, 3500, 4000).
# (Tipp: wählen Sie den Bereich der natürlichen Schlüsselwerte groß, damit es beim zufälligen Erzeugen zu möglichst wenigen Mehrfach auswahlen kommt)
# Sortieren Sie diese jeweils mit Bubblesort und speichern Sie die dazu benötigen Rechenzeiten. Geben Sie diese als Grafik aus. Welche Vermutung der Zeitabhängigkeit T(n) könnte man anstellen?
# Prüfen Sie dies mit einem Fit der Daten an die vermutete Abhängigkeit (mit einer Python Fit routine). Geben Sie Fit und Daten in einer Grafik wider.
from typing import Callable, Any, Tuple, Sequence
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

import time
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

def measure_runtime(func: Callable, *args, **kwargs) -> Tuple[Any, float]:
    """
    Führt eine Funktion aus und misst ihre Laufzeit.

    :param func: Die auszuführende Funktion
    :param args: Positionsargumente für die Funktion
    :param kwargs: Keyword-Argumente für die Funktion
    :return: Tuple aus (Rückgabewert der Funktion, Laufzeit in Sekunden)
    """
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    end_time = time.perf_counter()

    runtime = end_time - start_time
    return result, runtime


def bubble_sort(sort_array: npt.NDArray) -> npt.NDArray:
    """
    Sortiert ein NumPy-Array mithilfe des Bubble-Sort-Algorithmus in aufsteigender Reihenfolge.

    Bubble Sort vergleicht wiederholt benachbarte Elemente und vertauscht sie,
    falls sie in der falschen Reihenfolge stehen. Der Algorithmus hat eine
    Laufzeit von O(n²) im Worst- und Average-Case.

    :param sort_array: NumPy-Array mit vergleichbaren Elementen
    :return: das sortierte NumPy-Array (in-place sortiert)
    """
    sort_len = len(sort_array)
    for i in range(sort_len):
        swapped = False
        for j in range(sort_len - i - 1):
            if sort_array[j] > sort_array[j + 1]:
                sort_array[j], sort_array[j + 1] = sort_array[j + 1], sort_array[j]
                swapped = True
        if not swapped:
            break
    return sort_array


def plot_runtime_comparison(
    times_list: Sequence[float],
    length: Sequence[int],
    coef: np.ndarray,
    intercept: float,
    degree: int
) -> None:
    """
    Plottet gemessene Laufzeiten und die zugehörige Regressionsfunktion.

    :param times_list: Gemessene Laufzeiten
    :param length: Eingabelängen n
    :param coef: Modellkoeffizienten (model.coef_)
    :param intercept: Modellbias (model.intercept_)
    :param degree: Grad des Polynoms
    """
    x = np.array(length, dtype=float)
    y = np.array(times_list, dtype=float)

    x_fit = np.linspace(x.min(), x.max(), 300)
    y_fit = evaluate_polynomial(x_fit, coef, intercept)

    plt.figure(figsize=(8, 5))

    # Messpunkte
    plt.scatter(x, y, label="Messdaten")

    # Fit-Kurve
    plt.plot(x_fit, y_fit, label=f"Polynom-Fit (Grad {degree})")

    plt.title("Laufzeitvergleich Bubblesort")
    plt.xlabel("Eingabegröße n")
    plt.ylabel("Zeit in Sekunden")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()



def evaluate_polynomial(x: np.ndarray, coef: np.ndarray, intercept: float) -> np.ndarray:
    """
    Wertet ein Polynom der Form
    a_n x^n + ... + a_1 x + b
    aus.

    :param x: x-Werte
    :param coef: Koeffizienten [a1, a2, ..., an]
    :param intercept: Bias b
    :return: y-Werte
    """
    y = np.zeros_like(x, dtype=float)
    for power, a in enumerate(coef, start=1):
        y += a * x**power
    return y + intercept


def fit_runtime_polynomial(
        x: list,
        y: list,
        degree: int
) -> Tuple[np.ndarray, float]:
    """
    Approximiert Laufzeitdaten mit einem Polynom beliebigen Grades.

    Modell:
        Grad 1: ax + b
        Grad 2: ax² + bx + c
        Grad n: a_n x^n + ... + a_1 x + b

    :param x: Array mit Eingabelängen (z. B. n)
    :param y: Array mit gemessenen Laufzeiten
    :param degree: Grad des Polynoms
    :return: (Koeffizienten, Bias)
             Koeffizienten in aufsteigender Ordnung:
             [a_1, a_2, ..., a_n], Bias = b
    """
    x = np.array(x)
    x = x.reshape(-1, 1)

    y = np.array(y)

    poly = PolynomialFeatures(degree=degree, include_bias=False)
    x_poly = poly.fit_transform(x)

    model = LinearRegression()
    model.fit(x_poly, y)

    return model.coef_, model.intercept_


if __name__ == '__main__':
    n = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000]
    ts = []
    for size in n:
        arr = np.random.randint(1, size * 10, size=size)
        sorted_arr, t = measure_runtime(bubble_sort, arr)
        ts.append(t)

    coef, intercept = fit_runtime_polynomial(n, ts, degree=2)
    print(coef, intercept)

    plot_runtime_comparison(
        times_list=ts,
        length=n,
        coef=coef,
        intercept=intercept,
        degree=2
    )

