# Aufgabe Heapsort (RL)
# Folgende Aufgabe soll mit einem Python Programm gelöst werden:
# Erzeugen Sie einen Array mit n=20 zufällig ausgewählten natürlichen Schlüsseln. Bringen Sie ihn
# auf Heapstruktur. Davon ausgehend bilden Sie einen sortierten Array – egal von klein nach groß
# oder von groß nach klein.

import numpy as np
import numpy.typing as npt
import math


def heap_sort(arr: npt.NDArray, reverse: bool = False) -> npt.NDArray:
    """
    Sorts the Numpy array arr using heapsort.
    :param arr: array to sort (in-place).
    :param reverse: If True, sort descending; otherwise ascending.
    :return: The sorted Numpy array (same object).
    """
    n = arr.size
    # build heap: use not reverse so that reverse=True => descending result
    for i in range(n // 2 - 1, -1, -1):
        versickere(arr, n, i, not reverse)
    # sort
    for i in range(n - 1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]
        versickere(arr, i, 0, not reverse)
    return arr


def versickere(arr: npt.NDArray, n: int, i: int, reverse: bool) -> None:
    """
    Restore heap property for node i on heap size n.
    :param arr: Numpy array representing heap.
    :param n: heap size.
    :param i: index to sift down.
    :param reverse: If True, use max-heap comparisons; else min-heap.
    """
    selected = i
    left = 2 * i + 1
    right = 2 * i + 2

    if reverse:
        if left < n and arr[left] > arr[selected]:
            selected = left
        if right < n and arr[right] > arr[selected]:
            selected = right
    else:
        if left < n and arr[left] < arr[selected]:
            selected = left
        if right < n and arr[right] < arr[selected]:
            selected = right

    if selected != i:
        arr[i], arr[selected] = arr[selected], arr[i]
        versickere(arr, n, selected, reverse)


def print_tree(arr: npt.NDArray) -> None:
    n = int(arr.size)
    if n == 0:
        print("<empty>")
        return

    values = [str(x) for x in arr.tolist()]
    height = int(math.floor(math.log2(n))) + 1
    node_width = max(len(s) for s in values) + 2

    total_slots = 2**height
    total_width = total_slots * node_width

    level_info = []
    for level in range(height):
        slots = 2**level
        spacing = total_width // slots
        offset = spacing // 2 - node_width // 2
        level_info.append((spacing, offset))

    def node_center(level_center, idx_in_level):
        spacing_center, offset_center = level_info[level_center]
        return offset_center + idx_in_level * spacing_center + node_width // 2

    for level in range(height):
        start = 2**level - 1
        end = min(2 ** (level + 1) - 1, n)
        count = max(0, end - start)
        if count == 0:
            continue

        line_chars = [" "] * total_width
        for i in range(count):
            arr_idx = start + i
            s = values[arr_idx]
            center = node_center(level, i)
            left = center - (len(s) // 2)
            for k, ch in enumerate(s):
                pos = left + k
                if 0 <= pos < total_width:
                    line_chars[pos] = ch
        print("".join(line_chars).rstrip())

        if level < height - 1:
            conn = [" "] * total_width
            for i in range(count):
                parent_idx = start + i
                parent_center = node_center(level, i)
                if 0 <= parent_center < total_width:
                    conn[parent_center] = "|"
                left_child = 2 * parent_idx + 1
                right_child = 2 * parent_idx + 2
                child_i_base = 2 ** (level + 1) - 1
                for child in (left_child, right_child):
                    if child < n:
                        child_i = child - child_i_base
                        child_center = node_center(level + 1, child_i)
                        if 0 <= child_center < total_width:
                            conn[child_center] = "|"
                        a, b = sorted((parent_center, child_center))
                        for pos in range(a + 1, b):
                            if 0 <= pos < total_width and conn[pos] == " ":
                                conn[pos] = "-"
            print("".join(conn).rstrip())


def main():
    np.random.seed(42)
    keys = np.arange(20) + 1
    np.random.shuffle(keys)

    print("Unsortiertes Array:", keys)

    # heapsort: pass copies to avoid modifying original if needed
    sorted_desc = heap_sort(keys.copy(), True)
    print("Sortiertes Array (absteigend):", sorted_desc)
    print()
    print_tree(sorted_desc)

    print()
    print()

    sorted_asc = heap_sort(keys.copy(), False)
    print("Sortiertes Array (aufsteigend):", sorted_asc)

    print()
    print_tree(sorted_asc)


if __name__ == "__main__":
    main()
