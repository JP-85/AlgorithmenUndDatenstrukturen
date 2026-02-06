from __future__ import annotations

import math
import heapq
import tkinter as tk
from dataclasses import dataclass
from itertools import count
from typing import Dict, List, Optional, Tuple


# -----------------------------
# Domain: Node / Edge / Graph
# -----------------------------


@dataclass(frozen=True)
class Node:
    name: str
    x: float
    y: float

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Edge:
    src: Node
    dst: Node
    weight: float

    def __post_init__(self) -> None:
        if self.weight < 0:
            raise ValueError(
                "Dijkstra funktioniert nicht mit negativen Kantengewichten."
            )


class Graph:
    """
    Gerichteter Graph mit Dijkstra:
      - add_node(name, x, y)
      - add_edge(src, dst, weight)
      - walk(start, end) -> (path_nodes, distance)
      - draw(canvas, highlight_path)
    """

    def __init__(self) -> None:
        self._nodes: Dict[str, Node] = {}
        self._edges: List[Edge] = []
        self._adj: Dict[Node, List[Edge]] = {}

    def add_node(self, name: str, x: float, y: float) -> Node:
        if name in self._nodes:
            raise ValueError(f"Node '{name}' existiert bereits.")
        n = Node(name=name, x=x, y=y)
        self._nodes[name] = n
        self._adj[n] = []
        return n

    def node(self, name: str) -> Node:
        if name not in self._nodes:
            raise ValueError(f"Unbekannter Node: '{name}'")
        return self._nodes[name]

    def add_edge(self, src: str, dst: str, weight: float) -> None:
        s = self.node(src)
        d = self.node(dst)
        e = Edge(src=s, dst=d, weight=weight)
        self._edges.append(e)
        self._adj[s].append(e)

    def walk(self, start: str, end: str) -> Tuple[List[Node], float]:
        if start not in self._nodes or end not in self._nodes:
            raise ValueError("Start oder Ziel existiert nicht im Graphen.")

        s = self._nodes[start]
        t = self._nodes[end]

        dist: Dict[Node, float] = {n: math.inf for n in self._adj}
        prev: Dict[Node, Optional[Node]] = {n: None for n in self._adj}

        dist[s] = 0.0
        pq: List[Tuple[float, int, Node]] = [(0.0, next(count()), s)]
        # ^ Achtung: count() oben wäre jedes Mal neu. Daher:
        # Wir machen den Counter korrekt:
        c = count()
        pq = [(0.0, next(c), s)]

        visited: set[Node] = set()

        while pq:
            d_u, _, u = heapq.heappop(pq)
            if u in visited:
                continue
            visited.add(u)

            if u == t:
                break
            if d_u != dist[u]:
                continue

            for e in self._adj[u]:
                v = e.dst
                alt = d_u + e.weight
                if alt < dist[v]:
                    dist[v] = alt
                    prev[v] = u
                    heapq.heappush(pq, (alt, next(c), v))

        if dist[t] == math.inf:
            raise ValueError("Kein Pfad gefunden (Ziel unerreichbar).")

        # Pfad rekonstruieren
        path: List[Node] = []
        cur: Optional[Node] = t
        while cur is not None:
            path.append(cur)
            cur = prev[cur]
        path.reverse()

        return path, dist[t]

    # -------- drawing helpers --------

    @staticmethod
    def _shorten_line(
        ax: float, ay: float, bx: float, by: float, r: float
    ) -> Tuple[float, float, float, float]:
        """
        Kürzt eine Linie A->B so, dass sie nicht in die Node-Kreise reinläuft:
          Start = A + r*unit
          End   = B - r*unit
        """
        dx = bx - ax
        dy = by - ay
        length = math.hypot(dx, dy)
        if length == 0:
            return ax, ay, bx, by
        ux = dx / length
        uy = dy / length
        return ax + ux * r, ay + uy * r, bx - ux * r, by - uy * r

    def draw(
        self,
        canvas: tk.Canvas,
        highlight_path: Optional[List[Node]] = None,
        *,
        node_r: int = 18,
    ) -> None:
        canvas.delete("all")

        # Highlight-Kanten (gerichtet!)
        hl_edges = set()
        hl_nodes = set()
        if highlight_path:
            hl_nodes = set(highlight_path)
            for i in range(len(highlight_path) - 1):
                hl_edges.add((highlight_path[i].name, highlight_path[i + 1].name))

        # Kanten zuerst
        for e in self._edges:
            is_hl = (e.src.name, e.dst.name) in hl_edges
            width = 5 if is_hl else 2
            color = "blue" if is_hl else "gray30"

            x1, y1, x2, y2 = self._shorten_line(
                e.src.x, e.src.y, e.dst.x, e.dst.y, node_r
            )

            canvas.create_line(
                x1,
                y1,
                x2,
                y2,
                arrow="last",  # <= wichtig: Pfeil
                arrowshape=(18, 22, 8),  # <= sichtbare Spitze
                fill=color,
                width=width,
            )

            # Gewicht mittig (auf gekürzter Linie)
            mx = (x1 + x2) / 2
            my = (y1 + y2) / 2
            canvas.create_rectangle(
                mx - 14, my - 11, mx + 14, my + 11, outline="", fill="white"
            )
            canvas.create_text(mx, my, text=str(e.weight), font=("Arial", 10))

        # Nodes danach (liegen oben)
        for n in self._adj.keys():
            is_hl = n in hl_nodes
            fill = "lightblue" if is_hl else "white"
            outline = "black"
            w = 3 if is_hl else 2

            canvas.create_oval(
                n.x - node_r,
                n.y - node_r,
                n.x + node_r,
                n.y + node_r,
                fill=fill,
                outline=outline,
                width=w,
            )
            canvas.create_text(n.x, n.y, text=n.name, font=("Arial", 12, "bold"))


# -----------------------------
# UI: Tkinter App
# -----------------------------


class GraphApp(tk.Tk):
    def __init__(self, graph: Graph) -> None:
        super().__init__()
        self.title("Dijkstra – gerichteter Graph mit Pfeilen + Highlight")
        self.geometry("940x560")
        self.graph = graph

        self.canvas = tk.Canvas(self, bg="white")
        self.canvas.pack(side="left", fill="both", expand=True, padx=10, pady=10)

        right = tk.Frame(self)
        right.pack(side="right", fill="y", padx=10, pady=10)

        row = tk.Frame(right)
        row.pack(fill="x")

        tk.Label(row, text="Start:").pack(side="left")
        self.start_entry = tk.Entry(row, width=8)
        self.start_entry.pack(side="left", padx=(5, 15))

        tk.Label(row, text="Ziel:").pack(side="left")
        self.end_entry = tk.Entry(row, width=8)
        self.end_entry.pack(side="left", padx=(5, 15))

        tk.Button(row, text="Berechnen", command=self.on_run).pack(side="left")
        tk.Button(row, text="Reset", command=self.on_reset).pack(
            side="left", padx=(10, 0)
        )

        self.output = tk.Text(right, height=22, width=34, wrap="word")
        self.output.pack(fill="both", expand=True, pady=(10, 0))

        self.start_entry.insert(0, "A")
        self.end_entry.insert(0, "E")

        self.graph.draw(self.canvas)

    def on_run(self) -> None:
        start = self.start_entry.get().strip()
        end = self.end_entry.get().strip()
        self.output.delete("1.0", "end")

        try:
            path, dist = self.graph.walk(start, end)
            self.graph.draw(self.canvas, highlight_path=path)

            self.output.insert("end", f"Start: {start}\nZiel:  {end}\n\n")
            self.output.insert("end", f"Kürzeste Distanz: {dist}\n")
            self.output.insert(
                "end", "Pfad: " + " -> ".join(n.name for n in path) + "\n"
            )
        except Exception as e:
            self.graph.draw(self.canvas)
            self.output.insert("end", f"Fehler: {e}\n")

    def on_reset(self) -> None:
        self.graph.draw(self.canvas)
        self.output.delete("1.0", "end")
        self.output.insert("end", "Highlight zurückgesetzt.\n")


# -----------------------------
# Example Graph (gerichtet)
# -----------------------------


def build_example_graph() -> Graph:
    g = Graph()

    # Nodes mit festen Positionen
    g.add_node("A", 120, 140)
    g.add_node("B", 320, 110)
    g.add_node("C", 230, 280)
    g.add_node("D", 470, 290)
    g.add_node("E", 480, 140)

    # Gerichtete Kanten (Pfeile!)
    g.add_edge("A", "B", 4)
    g.add_edge("A", "C", 2)
    g.add_edge("C", "B", 1)
    g.add_edge("B", "D", 5)
    g.add_edge("C", "D", 8)
    g.add_edge("C", "E", 10)
    g.add_edge("D", "E", 2)

    return g


if __name__ == "__main__":
    app = GraphApp(build_example_graph())
    app.mainloop()
