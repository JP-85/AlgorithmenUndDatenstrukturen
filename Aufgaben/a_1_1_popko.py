# (1) Berechnen Sie die ersten 35 Fibonacci-Zahlen mir rekursiver und iterativer Methode. Speichern Sie die für jede Zahl benötigte Rechenzeit und geben Sie das Ergebnis im Vergleich der beiden Algorithmen als Grafik aus.
from matplotlib import pyplot as plt
import time
from typing import Callable, Any, Tuple


def fib(n):
    """
    Berechnet die n-te Fibonacci-Zahl mit einer iterativen (seriellen) Methode.

    :param n: Index der zu berechnenden Fibonacci-Zahl
    :return: n-te Fibonacci-Zahl
    """
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def fib_rek(n):
    """
    Berechnet die n-te Fibonacci-Zahl mit einer rekursiven Methode.

    :param n: Index der zu berechnenden Fibonacci-Zahl
    :return: n-te Fibonacci-Zahl
    """
    if n >= 2:
        f = fib_rek(n - 1) + fib_rek(n - 2)
    elif n == 0:
        return 0
    else:
        return 1
    return f


def plot_fibonacci_comparison(
    fib_serial: list, fib_recursive: list, title: str, y_label: str
) -> None:
    """
    Stellt zwei Fibonacci-Folgen nebeneinander dar:
    - links: seriell berechnet
    - rechts: rekursiv berechnet

    :param y_label: Zahl oder Zeit
    :param title: Übertitel des Plots
    :param fib_serial: Liste mit seriell berechneten Fibonacci-Zahlen
    :param fib_recursive: Liste mit rekursiv berechneten Fibonacci-Zahlen
    """
    x_serial = range(len(fib_serial))
    x_recursive = range(len(fib_recursive))

    plt.figure(figsize=(12, 5))
    plt.suptitle(title)

    # Seriell
    plt.subplot(1, 2, 1)
    plt.plot(x_serial, fib_serial, marker="o")
    plt.title("Fibonacci – Seriell")
    plt.xlabel("Index")
    plt.ylabel(y_label)
    plt.grid(True)

    # Rekursiv
    plt.subplot(1, 2, 2)
    plt.plot(x_recursive, fib_recursive, marker="o")
    plt.title("Fibonacci – Rekursiv")
    plt.xlabel("Index")
    plt.ylabel(y_label)
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_runtime_comparison(times_serial: list, times_recursive: list) -> None:
    """
    Vergleicht die Laufzeiten zweier Algorithmen in einem Diagramm.

    :param times_serial: Liste mit Laufzeiten (z. B. serieller Algorithmus)
    :param times_recursive: Liste mit Laufzeiten (z. B. rekursiver Algorithmus)
    """
    x_serial = range(len(times_serial))
    x_recursive = range(len(times_recursive))

    plt.figure(figsize=(8, 5))

    plt.plot(x_serial, times_serial, marker="o", label="Seriell")
    plt.plot(x_recursive, times_recursive, marker="o", label="Rekursiv")

    plt.title("Laufzeitvergleich Fibonacci")
    plt.xlabel("Messung / n")
    plt.ylabel("Zeit in Sekunden")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


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


if __name__ == "__main__":
    n_max = 35

    print(fib(n_max))
    print(fib_rek(n_max))
    print()

    fib_nums = []
    fib_nums_runtime = []
    for i in range(n_max):
        fib_num, fib_time = measure_runtime(fib, i)
        fib_nums.append(fib_num)
        fib_nums_runtime.append(fib_time)

    fib_rek_nums = []
    fib_rek_nums_runtime = []
    for i in range(n_max):
        fib_num_rek, fib_time_rek = measure_runtime(fib_rek, i)
        fib_rek_nums.append(fib_num_rek)
        fib_rek_nums_runtime.append(fib_time_rek)

    plot_fibonacci_comparison(
        fib_nums, fib_rek_nums, "Fibonacci-Zahlen", "Fibonacci-Zahl"
    )
    plot_fibonacci_comparison(
        fib_nums_runtime, fib_rek_nums_runtime, "Fibonacci-Zeiten", "Zeit"
    )
    plot_runtime_comparison(fib_nums_runtime, fib_rek_nums_runtime)
