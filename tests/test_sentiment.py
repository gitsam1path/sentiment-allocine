import json
from pathlib import Path

import pytest

from sentiment.train import EXEMPLES, RuleBased

ROOT = Path(__file__).resolve().parents[1]


def test_regles_et_negation():
    m = RuleBased()
    assert m.predict(["Un film magnifique et touchant."])[0] == 1
    assert m.predict(["Un navet ennuyeux."])[0] == 0
    assert m.predict(["Ce n'est pas mauvais."])[0] == 1     # la négation inverse le mot qui suit
    assert m.predict(["Je n'ai pas aimé."])[0] == 0


def test_exemples_equilibres():
    labels = [e[1] for e in EXEMPLES]
    assert 0 < sum(labels) < len(labels)


def test_donnees_hors_du_depot():
    """Les critiques ne sont jamais versionnées : le dossier data/ est ignoré par Git."""
    assert "data/" in (ROOT / ".gitignore").read_text().splitlines()


def test_rapport():
    path = ROOT / "reports" / "metrics.json"
    if not path.exists():
        pytest.skip("lancer d'abord : cd src && python -m sentiment.train")
    r = json.loads(path.read_text(encoding="utf-8"))
    res = {a["approche"]: a for a in r["resultats"]}
    assert res[r["approche_retenue"]]["test_accuracy"] > 0.9
    assert res["Règles (liste de mots)"]["test_accuracy"] < res[r["approche_retenue"]]["test_accuracy"]
    assert r["donnees"]["test"] == 20000
