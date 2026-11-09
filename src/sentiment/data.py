"""
Critiques de films Allociné (jeu de Théophile Blard, 2020) : 200 000 critiques en français,
moitié positives, moitié négatives, déjà réparties en entraînement (160 000), validation (20 000) et test (20 000).

Les critiques ne sont pas recopiées dans ce dépôt : elles sont téléchargées depuis le dépôt d'origine
à la première exécution, puis gardées en cache dans data/ (ignoré par Git).
"""

from __future__ import annotations

import json
import tarfile
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
ARCHIVE = DATA / "data.tar.bz2"
URL = ("https://github.com/TheophileBlard/french-sentiment-analysis-with-bert/raw/master/"
       "allocine_dataset/data.tar.bz2")
SPLITS = ("train", "val", "test")


def ensure_data() -> Path:
    folder = DATA / "data"
    if all((folder / f"{s}.jsonl").exists() for s in SPLITS):
        return folder
    DATA.mkdir(parents=True, exist_ok=True)
    if not ARCHIVE.exists():
        print("Téléchargement des critiques Allociné (≈ 65 Mo, une seule fois)...")
        urllib.request.urlretrieve(URL, ARCHIVE)
    with tarfile.open(ARCHIVE, "r:bz2") as tar:
        members = [m for m in tar.getmembers() if m.name.endswith(".jsonl")]
        tar.extractall(DATA, members=members, filter="data")
    return folder


def load(split: str) -> pd.DataFrame:
    folder = ensure_data()
    rows = [json.loads(line) for line in open(folder / f"{split}.jsonl", encoding="utf-8")]
    df = pd.DataFrame(rows).rename(columns={"review": "texte", "polarity": "positif", "film-url": "film"})
    df["positif"] = df["positif"].astype(int)
    df["texte"] = df["texte"].fillna("").astype(str)
    return df[["film", "texte", "positif"]]
