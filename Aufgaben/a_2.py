# Aufgabe Heapsort (RL):
# Die folgende Aufgabe soll mit einem Python Programm gelöst werden.
# Erzeugen Sie einen Array mit n=20 zufällig ausgewählten natürlichen Schlüsseln. Bringen Sie ihn
# auf Heapstruktur. Davon ausgehend bilden Sie einen sortierten Array – egal von klein nach groß
# oder von groß nach klein.

import heapq

import numpy as np
import numpy.typing as npt


def heap_sort(arr: npt.NDArray, reverse: bool = False) -> npt.NDArray:
    """
    Sortiert das Numpy Array arr mittels Heapsort Algorithmus.
    :param arr: Das zu sortierende Numpy Array.
    :param reverse: Wenn True, wird absteigend sortiert, sonst aufsteigend.
    :return: Das sortierte Numpy Array.
    """
    n = arr.size
    if reverse:
        # Max-Heap bauen
        for i in range(n // 2 - 1, -1, -1):
            versickere(arr, n, i, reverse)
        # Sortieren
        for i in range(n - 1, 0, -1):
            arr[0], arr[i] = arr[i], arr[0]
            versickere(arr, i, 0, reverse)
    else:
        # Min-Heap bauen
        for i in range(n // 2 - 1, -1, -1):
            versickere(arr, n, i, reverse)
        # Sortieren
        for i in range(n - 1, 0, -1):
            arr[0], arr[i] = arr[i], arr[0]
            versickere(arr, i, 0, reverse)
    return arr

def versickere(arr: npt.NDArray, n: int, i: int, reverse: bool) -> None:
    """
    Hilfsfunktion, die die Heap-Eigenschaft wiederherstellt.
    :param arr: Das Numpy Array, das den Heap repräsentiert.
    :param n: Die Größe des Heaps.
    :param i: Der Index des Knotens, der versickert werden soll.
    :param reverse: Wenn True, wird ein Max-Heap verwendet, sonst ein Min-Heap.
    """
    largest_or_smallest = i
    left = 2 * i + 1
    right = 2 * i + 2

    if reverse:
        if left < n and arr[left] > arr[largest_or_smallest]:
            largest_or_smallest = left
        if right < n and arr[right] > arr[largest_or_smallest]:
            largest_or_smallest = right
    else:
        if left < n and arr[left] < arr[largest_or_smallest]:
            largest_or_smallest = left
        if right < n and arr[right] < arr[largest_or_smallest]:
            largest_or_smallest = right

    if largest_or_smallest != i:
        arr[i], arr[largest_or_smallest] = arr[largest_or_smallest], arr[i]
        versickere(arr, n, largest_or_smallest, reverse)


def main():
    # Zufälliges Array (und zwei gleiche Listen) mit 20 natürlichen Schlüsseln erzeugen:
    np.random.seed(42)
    keys = np.arange(20) + 1
    np.random.shuffle(keys)
    keys_2 = keys.copy()
    keys_list = keys.tolist()
    keys_list_2 = keys.tolist()

    print("Unsortiertes Array:", keys)

    print("Builtins von Python:")
    heapq.heapify_max(keys_list)
    print("Heapstruktur (Max-Heap):", keys_list)
    heapq.heapify(keys_list_2)
    print("Heapstruktur (Min-Heap):", keys_list_2)
    sorted_max_arr = heap_sort(keys, True)
    print("keys", keys)
    print("Sortiertes Array (absteigend):", sorted_max_arr)
    sorted_min_arr = heap_sort(keys_2)
    print("Sortiertes Array (aufsteigend):", sorted_min_arr)
    print()


if __name__ == '__main__':
    main()
