# Aufgabe Heapsort (RL)
# Folgende Aufgabe soll mit einem Python Programm gelöst werden:
# Erzeugen Sie einen Array mit n=20 zufällig ausgewählten natürlichen Schlüsseln. Bringen Sie ihn
# auf Heapstruktur. Davon ausgehend bilden Sie einen sortierten Array – egal von klein nach groß
# oder von groß nach klein.

import numpy as np
import numpy.typing as npt


def main() -> None:
    n_keys = 20
    np.random.seed(42)
    keys = np.arange(n_keys) + 1
    np.random.shuffle(keys)
    print(keys)
    print()
    print_tree(keys)
    print()
    max_heap = heap_sort(keys)
    print(max_heap)
    print()
    print_tree(max_heap)
    print()
    print()
    min_heap = heap_sort(keys, reverse=True)
    print(min_heap)
    print()
    print_tree(min_heap)


def print_tree(arr: npt.NDArray[np.int_]) -> None:
    """Druckt das Array als Baum.
    :param arr: das zu druckende Array"""
    n = arr.size
    depth = int(np.floor(np.log2(n)) + 1)
    leaves = 2**depth
    max_char = (leaves * 4) + 4

    tree = (" " * ((max_char // 2) + 2)) + f"{arr[0]:02}\n"
    for i in range(1, depth):
        nodes = 2 ** (i + 1) - 2**i
        for j in range(2**i - 1, 2 ** (i + 1) - 1):
            empty_char = " " * ((max_char // (nodes + 1)) - 2)
            if j == n:
                break
            # if j % 2:
            tree += empty_char + f"{arr[j]:02}"
        tree += "\n"
    print(tree)


def build_max_heap(arr: npt.NDArray[np.int_]) -> None:
    """Erstellt einen Maxheap.
    :param arr: das zu sortierende Array"""
    n = arr.size
    for i in range(n // 2 - 1, -1, -1):
        versickere(arr, i, n)


def versickere(arr: npt.NDArray[np.int_], parent: int, heap_size: int) -> None:
    """Vertauscht Kinder mit Elternknoten, wenn sie größer sind.
    :param arr: das Array, welches zum heap sortiert wird
    :param parent: ab welchem Elternknoten wird versickert?
    :param heap_size: Anzahl der Knoten/Blätter im (verkleinerten) Baum
    """
    while True:
        left_child = 2 * parent + 1
        right_child = 2 * parent + 2
        if left_child >= heap_size:
            break
        if right_child < heap_size and arr[right_child] > arr[left_child]:
            max_child_index = right_child
        else:
            max_child_index = left_child
        if arr[parent] < arr[max_child_index]:
            arr[parent], arr[max_child_index] = arr[max_child_index], arr[parent]
            parent = max_child_index
        else:
            break


def heap_sort(arr: npt.NDArray[np.int_], reverse: bool = False) -> npt.NDArray[np.int_]:
    """Macht einen heap-sort auf dem Array
    :param arr: das zu sortierende Array
    :param reverse: aufstegend, wenn False, sonst absteigend sortiert
    :return: sortiertes Array"""
    arr = arr.copy()
    n = arr.size

    build_max_heap(arr)

    for end in range(n - 1, 0, -1):
        arr[0], arr[end] = arr[end], arr[0]
        versickere(arr, 0, end)
    if reverse:
        return arr[::-1]
    else:
        return arr


if __name__ == "__main__":
    main()
