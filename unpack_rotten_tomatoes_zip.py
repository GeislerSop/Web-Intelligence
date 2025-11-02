import os
import zipfile
from pathlib import Path
import pandas as pd

# Basispfade
BASE = Path(__file__).resolve().parent
RAW_DIR = BASE / "raw"
OUT_DIR = BASE / "outputs"
OUT_DIR.mkdir(exist_ok=True)

# Suche ZIP-Datei (egal ob .zip oder .csv mit ZIP-Inhalt)
CANDIDATES = [
    RAW_DIR / "rotten_tomatoes_movies.csv",
    RAW_DIR / "rotten_tomatoes_movies.zip",
    BASE / "rotten_tomatoes_movies.csv",
]

zip_path = None
for c in CANDIDATES:
    if c.exists():
        zip_path = c
        break

if not zip_path:
    raise FileNotFoundError("Keine ZIP-Datei gefunden – bitte in raw/ ablegen.")

print(f"Gefundene ZIP-Datei: {zip_path}")

# ZIP prüfen und entpacken
with open(zip_path, "rb") as f:
    header = f.read(4)

if not header.startswith(b"PK"):
    print("⚠️  Datei scheint keine ZIP-Datei zu sein (kein PK Header).")
else:
    with zipfile.ZipFile(zip_path, "r") as z:
        names = z.namelist()
        print(f"\n📦 Enthaltene Dateien im ZIP ({len(names)}):")
        for n in names:
            print(" -", n)
        print()

        # Entpacke alles nach outputs/
        z.extractall(OUT_DIR)
        print(f"✅ Alle Dateien entpackt nach: {OUT_DIR}")

# Zeige erste Zeilen jeder CSV im Output-Ordner
csvs = [p for p in OUT_DIR.glob("*.csv")]
if not csvs:
    print("⚠️  Keine CSV-Dateien im outputs/ gefunden.")
else:
    for p in csvs:
        print(f"\n=== {p.name} ===")
        try:
            df = pd.read_csv(p, nrows=5)
            print(df.head().to_string(index=False))
            print("Spalten:", ", ".join(df.columns[:12].tolist()), "...")
        except Exception as e:
            print(f"Fehler beim Lesen von {p}: {e}")
