# Algorithmen und Datenstrukturen

Prüfe die Installation von Python und Git:

```powershell
python --version
git --version
````

---

## uv installieren

Öffne **PowerShell** und führe folgenden Befehl aus:

```powershell
irm https://astral.sh/uv/install.ps1 | iex
```

Danach **PowerShell neu starten** und überprüfen:

```powershell
uv --version
```

---

## Projekt von GitHub klonen

Stelle sicher, dass Git installiert ist (siehe Voraussetzungen).

Repository klonen:

```powershell
git clone https://github.com/JP-85/AlgorithmenUndDatenstrukturen.git
```

---

## In das Projektverzeichnis wechseln

```powershell
cd AlgorithmenUndDatenstrukturen
```

---

## Abhängigkeiten installieren

Alle benötigten Bibliotheken sind in der Datei `pyproject.toml` definiert
(z. B. numpy, pandas, matplotlib, scikit-learn).

Installation der Abhängigkeiten und Erstellen der virtuellen Umgebung:

```powershell
uv sync
```

---

## Python-Dateien ausführen

Eine Python-Datei kann wie folgt gestartet werden:

```powershell
uv run python <dateiname>.py
```

Allgemein:

```powershell
uv run python <dateiname>.py
```

---

## Verwendung in PyCharm

* PyCharm nach der Installation von `uv` **neu starten**
* Das integrierte Terminal verwenden
* Falls nötig, funktioniert immer:

```powershell
python -m uv run python main.py
```

---

## Verwendete Bibliotheken

* numpy
* pandas
* matplotlib
* scikit-learn

---
