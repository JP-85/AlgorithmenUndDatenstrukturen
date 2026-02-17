import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

from dataclasses import dataclass

# Aufgabe 14
# Gegeben sind die Worte
# Elefant, Giraffe, L¨owe, Panther, Hund, Ratte, Maus, Walhai,
# Thunfisch, Forelle, Clownfisch, Hering, Seepferdchen
# dazu die Merkmale S¨augetier, Gr¨oße
# • Finden Sie eine geeignete numerische Abbildung der Merkmale in die
# reellen Zahlen. Diese Abbildung kann sich von Anwendung zu Anwendung durchaus ¨andern. Man kann die Gr¨oße in den Vordergrund stellen
# oder die Artenzugeh¨origkeit.
# • Finden Sie die Analogie:
# die Maus verh¨alt sich zum Elefanten wie der Clownfisch zum
# ....
# • Bilden Sie darauf aufbauend die Einheitsvektoren mit gemeinsamem
# Mittelpunkt (zero mean) und stellen Sie diese grafisch mit einem Einheitskreis dar.
# • Berechnen Sie die Skalarprodukte und geben Sie die Ahnlichkeiten an


@dataclass
class Animal:
    name: str
    mammal: float
    size: float

    def __str__(self):
        return f"Animal: {self.name}, mammal: {self.mammal}, size: {self.size}"

    def __repr__(self):
        return f"({self.__str__()})"


def get_feature_vectors(animals: npt.NDArray) -> np.ndarray:
    """
    Extract feature vectors (mammal, size) from animals.

    Parameters
    ----------
    animals : npt.NDArray
        NumPy array of Animal objects.

    Returns
    -------
    np.ndarray
        NumPy array of shape (n, 2) containing [mammal, size] for each animal.
    """
    vectors = np.array([[a.mammal, a.size] for a in animals])
    return vectors


def normalize_vectors(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize vectors to unit vectors with zero mean.

    Parameters
    ----------
    vectors : np.ndarray
        NumPy array of shape (n, 2) containing feature vectors.

    Returns
    -------
    normalized_vectors : np.ndarray
        Unit vectors with zero mean (shape: n x 2).
    centered_vectors : np.ndarray
        Vectors after zero-mean centering (shape: n x 2).
    means : np.ndarray
        Mean values of original vectors (shape: 2).
    """
    # Mittelwert berechnen
    means = np.mean(vectors, axis=0)
    # Mittelwert + Zentrieren
    centered = vectors - means

    # Standardisieren: Alle Vektoren auf Länge 1 normieren
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normalized = centered / norms

    return normalized, centered, means


def find_analogy(
    vectors: np.ndarray, names: np.ndarray, word1: str, word2: str, word3: str
) -> str:
    """
    Find analogy: word1 is to word2 as word3 is to ?

    Using vector arithmetic: vec(word2) - vec(word1) + vec(word3) ≈ vec(?).

    Parameters
    ----------
    vectors : np.ndarray
        NumPy array of feature vectors for all animals (shape: n x 2).
    names : np.ndarray
        NumPy array of animal names corresponding to vectors.
    word1 : str
        First word in the analogy (e.g., "Maus").
    word2 : str
        Second word in the analogy (e.g., "Elefant").
    word3 : str
        Third word in the analogy (e.g., "Clownfisch").

    Returns
    -------
    str
        Name of the animal that best completes the analogy.
    """
    # finde die Indizes der Wörter
    idx1 = np.where(names == word1)[0][0]
    idx2 = np.where(names == word2)[0][0]
    idx3 = np.where(names == word3)[0][0]

    # Vektor berechnen: (word2 - word1) + word3
    target = vectors[idx2] - vectors[idx1] + vectors[idx3]

    # Ähnlichstes Wort finden (ohne word3)
    similarities = []
    for i, vec in enumerate(vectors):
        if i == idx3:
            similarities.append(-np.inf)
        else:
            sim = np.dot(target, vec) / (np.linalg.norm(target) * np.linalg.norm(vec))
            similarities.append(sim)

    best_idx = np.argmax(similarities)
    return names[best_idx]


def plot_unit_circle(normalized_vectors: np.ndarray, names: np.ndarray, save_file: str):
    """
    Plot normalized vectors on unit circle.

    Parameters
    ----------
    normalized_vectors : np.ndarray
        NumPy array of vectors (shape: n x 2).
    names : np.ndarray
        NumPy array of animal names corresponding to vectors.

    Returns
    -------
    None
        Displays plot and saves to 'animal_vectors.png'.
    """
    plt.figure(figsize=(12, 12))

    max_length = np.max(np.linalg.norm(normalized_vectors, axis=1))

    for radius in np.arange(1, max_length + 1, 1):
        theta = np.linspace(0, 2 * np.pi, 100)
        plt.plot(
            radius * np.cos(theta),
            radius * np.sin(theta),
            "k--",
            alpha=0.15,
            linewidth=1,
        )

    cmap = plt.colormaps["tab20c"]
    colors = cmap(np.linspace(0.1, 0.9, len(names)))

    legend_handles = []
    for i, (vec, name, color) in enumerate(zip(normalized_vectors, names, colors)):
        plt.arrow(
            0,
            0,
            vec[0],
            vec[1],
            head_width=0.05,
            head_length=0.05,
            fc=color,
            ec=color,
            alpha=0.8,
            length_includes_head=True,
            linewidth=2.5,
            label=name,
        )

        from matplotlib.patches import Patch

        legend_handles.append(Patch(facecolor=color, edgecolor=color, label=name))

    lim = max_length * 1.3
    plt.xlim(-lim, lim)
    plt.ylim(-lim, lim)
    plt.axhline(y=0, color="k", linestyle="-", alpha=0.3, linewidth=1)
    plt.axvline(x=0, color="k", linestyle="-", alpha=0.3, linewidth=1)
    plt.grid(True, alpha=0.2, linestyle="--")
    plt.xlabel("Säugetier", fontsize=13, weight="bold")
    plt.ylabel("Größe", fontsize=13, weight="bold")
    plt.title("Vektoren der Tiere", fontsize=15, weight="bold", pad=20)

    plt.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        fontsize=10,
        framealpha=0.9,
        title="Tiere",
        title_fontsize=11,
    )

    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(f"{save_file}.png", dpi=300, bbox_inches="tight")
    plt.show()


def calculate_similarities(vectors: np.ndarray, names: np.ndarray):
    """
    Calculate and display pairwise similarities.

    Parameters
    ----------
    vectors : np.ndarray
        NumPy array of feature vectors (shape: n x 2).
    names : np.ndarray
        NumPy array of animal names corresponding to vectors.

    Returns
    -------
    None
        Prints similarity matrix and statistics to console.
    """
    n = len(names)

    print("\n" + "=" * 80)
    print("SKALARPRODUKTE UND ÄHNLICHKEITEN (Cosine Similarity)")
    print("=" * 80)

    similarities = np.zeros((n, n))
    dot_products = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            # Matrixprodukt (Punktprodukt)
            dot_products[i, j] = np.dot(vectors[i], vectors[j])

            norm_i = np.linalg.norm(vectors[i])
            norm_j = np.linalg.norm(vectors[j])
            if norm_i > 0 and norm_j > 0:
                similarities[i, j] = dot_products[i, j] / (norm_i * norm_j)
            else:
                similarities[i, j] = 0

    print(f"\n{'Tier':15}", end="")
    for name in names:
        print(f"{name:12}", end="")
    print("\n" + "-" * 80)

    for i, name1 in enumerate(names):
        print(f"{name1:15}", end="")
        for j, name2 in enumerate(names):
            print(f"{similarities[i, j]:12.4f}", end="")
        print()

    print("\n" + "=" * 80)
    print("TOP 5 ÄHNLICHSTE TIERPAARE:")
    print("=" * 80)

    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((similarities[i, j], names[i], names[j]))

    pairs.sort(reverse=True)
    for rank, (sim, name1, name2) in enumerate(pairs[:5], 1):
        print(f"{rank}. {name1:15} ↔ {name2:15}  Ähnlichkeit: {sim:.4f}")

    print("\n" + "=" * 80)
    print("TOP 5 UNÄHNLICHSTE TIERPAARE:")
    print("=" * 80)

    for rank, (sim, name1, name2) in enumerate(reversed(pairs[-5:]), 1):
        print(f"{rank}. {name1:15} ↔ {name2:15}  Ähnlichkeit: {sim:.4f}")


def main() -> None:
    """
    Main function to execute the animal vector analysis task.

    Performs the following steps:
    1. Creates animal data with features (mammal type, size)
    2. Extracts feature vectors
    3. Finds analogies using vector arithmetic
    4. Normalizes vectors to unit vectors with zero mean
    5. Calculates and displays pairwise similarities
    6. Visualizes vectors on unit circle

    Parameters
    ----------

    Returns
    -------
    None
        Prints analysis results to console and generates visualization.
    """
    print("=" * 80)
    print("AUFGABE: WORD EMBEDDINGS UND VEKTORANALYSE FÜR TIERE")
    print("=" * 80)

    names = np.array(
        [
            "Elefant",
            "Giraffe",
            "Löwe",
            "Hund",
            "Ratte",
            "Maus",
            "Walhai",
            "Thunfisch",
            "Forelle",
            "Clownfisch",
            "Hering",
            "Seepferdchen",
        ],
        dtype=str,
    )
    mammals = np.array(
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float
    )
    sizes = np.array(
        [3.3, 5.5, 2.1, 0.9, 0.25, 0.09, 12.0, 2.5, 0.7, 0.11, 0.35, 0.15], dtype=float
    )

    animals = []
    for i in range(names.size):
        animals.append(Animal(name=names[i], mammal=mammals[i], size=sizes[i]))
    animals = np.array(animals, dtype=object)

    print("\nTIERE UND IHRE MERKMALE:")
    print("-" * 80)
    for animal in animals:
        print(f"  {animal}")

    # Get feature vectors
    print("\nFEATURE VEKTOREN (Säugetier, Größe):")
    print("-" * 80)
    vectors = get_feature_vectors(animals)
    for name, vec in zip(names, vectors):
        print(f"  {name:15} → [{vec[0]:.1f}, {vec[1]:.2f}]")

    # Find analogy
    print("\nANALOGIE-SUCHE:")
    print("-" * 80)
    print("  Frage: Maus verhält sich zu Elefant wie Clownfisch zu ...?")

    result = find_analogy(vectors, names, "Maus", "Elefant", "Clownfisch")
    print(f"  Antwort: {result}")
    print(f"  → Maus : Elefant = Clownfisch : {result}")
    print("\n  Erklärung:")
    print("    - Maus ist kleines Säugetier, Elefant ist großes Säugetier")
    print(f"    - Clownfisch ist kleiner Fisch, {result} ist großer Fisch")

    # Normalize vectors (zero mean, unit length)
    print("\nNORMALISIERUNG (Zero Mean + Einheitsvektoren):")
    print("-" * 80)
    normalized, centered, means = normalize_vectors(vectors)
    print(f"  Mittelwerte: Säugetier={means[0]:.3f}, Größe={means[1]:.3f}")
    print("\n  Zentrierte Vektoren (zero mean):")
    for name, vec in zip(names, centered):
        length = np.linalg.norm(vec)
        print(f"  {name:15} → [{vec[0]:7.4f}, {vec[1]:7.4f}]  (Länge: {length:.4f})")

    # Calculate similarities
    print("\nÄHNLICHKEITSANALYSE:")
    calculate_similarities(vectors, names)

    print("\n" + "=" * 80)
    print("ZUSAMMENFASSUNG:")
    print("=" * 80)
    print(f"Analogie gefunden: Maus : Elefant = Clownfisch : {result}")
    print("=" * 80)

    plot_unit_circle(centered, names, "animal_vectors")

    for name, vec in zip(names, normalized):
        length = np.linalg.norm(vec)
        print(f"  {name:15} → [{vec[0]:7.4f}, {vec[1]:7.4f}]  (Länge: {length:.4f})")

    plot_unit_circle(normalized, names, "animal_vectors_normalized")


if __name__ == "__main__":
    main()
