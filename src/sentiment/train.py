"""
Analyse de sentiments de critiques de films en français : positive ou négative ?

1. 5 approches comparées, réglées sur la validation (20 000 critiques) et évaluées UNE fois sur le test (20 000) :
   règles (liste de mots), Naive Bayes, régression logistique (mots), SVM linéaire (morceaux de mots), combinaison ;
2. courbe d'apprentissage : combien de critiques faut-il pour bien apprendre ?
3. interprétation : les mots qui font basculer le verdict ;
4. analyse des erreurs : longueur, négations, ironie (sur des phrases d'exemple écrites pour le projet).

Lancer depuis src/ :  python -m sentiment.train
"""

from __future__ import annotations

import json
import re
import time
import warnings

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

from .data import ROOT, load

warnings.filterwarnings("ignore")
REPORTS, FIG, MODELS = ROOT / "reports", ROOT / "reports" / "figures", ROOT / "models"
SEED = 42

plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "font.size": 10})
BLUE, ORANGE, GREEN, RED, GREY, PURPLE = "#1a73e8", "#e8710a", "#188038", "#d93025", "#9aa0a6", "#8e6bbf"

# Référence « à la main » : une petite liste de mots, sans apprentissage
POSITIFS = {"bien", "bon", "bonne", "excellent", "excellente", "superbe", "magnifique", "génial", "géniale", "chef",
            "parfait", "parfaite", "beau", "belle", "adoré", "aime", "aimé", "drôle", "émouvant", "réussi", "réussie",
            "bravo", "incontournable", "sublime", "touchant", "efficace", "agréable", "top", "merveilleux"}
NEGATIFS = {"mauvais", "mauvaise", "nul", "nulle", "ennui", "ennuyeux", "ennuyeuse", "décevant", "décevante", "déçu",
            "déception", "raté", "ratée", "pire", "navet", "horrible", "affligeant", "médiocre", "inutile", "long",
            "lent", "ridicule", "catastrophe", "bof", "dommage", "pénible", "fade", "plat", "zéro"}

# Phrases d'exemple écrites pour ce projet (aucune n'est tirée du jeu de données)
EXEMPLES = [
    ("Un film magnifique, des acteurs justes et une fin qui m'a ému.", 1),
    ("Deux heures d'ennui, un scénario creux et des dialogues ridicules.", 0),
    ("Pas mauvais du tout, je ne m'attendais pas à passer un si bon moment.", 1),
    ("Je ne recommande pas : rien n'est réussi, ni l'histoire ni la mise en scène.", 0),
    ("Bravo aux scénaristes, il fallait oser faire aussi plat pendant deux heures.", 0),
    ("Le début est lent mais la seconde moitié est vraiment excellente.", 1),
    ("Les effets spéciaux sont superbes, dommage que l'histoire ne suive pas.", 0),
    ("Rien d'original, mais ça se regarde avec plaisir en famille.", 1),
]


class RuleBased:
    """+1 par mot positif, −1 par mot négatif ; une négation (« pas », « jamais »…) juste avant inverse le mot."""

    NEG = {"pas", "jamais", "plus", "rien", "aucun", "aucune", "ni"}

    def fit(self, X, y=None):
        return self

    def decision_function(self, texts):
        out = []
        for t in texts:
            words = re.findall(r"\w+", t.lower())
            s = 0
            for i, w in enumerate(words):
                v = (w in POSITIFS) - (w in NEGATIFS)
                if v and any(p in self.NEG for p in words[max(0, i - 3):i]):
                    v = -v
                s += v
            out.append(s)
        return np.array(out, dtype=float)

    def predict(self, texts):
        return (self.decision_function(texts) >= 0).astype(int)


def approaches() -> dict:
    words = dict(ngram_range=(1, 2), min_df=3, max_df=0.9, sublinear_tf=True, max_features=300_000)
    chars = dict(analyzer="char_wb", ngram_range=(2, 5), min_df=5, sublinear_tf=True, max_features=400_000)
    return {
        "Règles (liste de mots)": (RuleBased(), None),
        "Naive Bayes (mots)": (make_pipeline(CountVectorizer(ngram_range=(1, 2), min_df=3, max_features=300_000),
                                             MultinomialNB()), "multinomialnb__alpha", [0.1, 0.3, 1.0]),
        "Régression logistique (mots et paires de mots)": (make_pipeline(TfidfVectorizer(**words), LogisticRegression(max_iter=3000)),
                                                          "logisticregression__C", [2, 5, 10]),
        "SVM linéaire (morceaux de mots)": (make_pipeline(TfidfVectorizer(**chars), LinearSVC()),
                                            "linearsvc__C", [0.1, 0.3, 1.0]),
    }


def scores(model, texts):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(texts)[:, 1] - 0.5
    return model.decision_function(texts)


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def main():
    for d in (FIG, MODELS):
        d.mkdir(parents=True, exist_ok=True)
    train, val, test = load("train"), load("val"), load("test")
    print(f"Critiques : {len(train)} entraînement, {len(val)} validation, {len(test)} test · "
          f"{train['positif'].mean():.0%} de positives")

    rows, fitted, test_scores = [], {}, {}
    for name, spec in approaches().items():
        t0 = time.perf_counter()
        if spec[1] is None:
            model, best_param = spec[0], None
        else:
            # réglage de l'hyperparamètre sur la VALIDATION uniquement
            pipe, param, grid = spec
            best_acc, best_param = -1, None
            for v in grid:
                m = pipe.set_params(**{param: v}).fit(train["texte"], train["positif"])
                acc = accuracy_score(val["positif"], m.predict(val["texte"]))
                if acc > best_acc:
                    best_acc, best_param = acc, v
            model = pipe.set_params(**{param: best_param}).fit(train["texte"], train["positif"])
        s = scores(model, test["texte"])
        pred = (s >= 0).astype(int)
        fitted[name], test_scores[name] = model, s
        rows.append({"approche": name, "parametre": best_param,
                     "val_accuracy": accuracy_score(val["positif"], model.predict(val["texte"])),
                     "test_accuracy": accuracy_score(test["positif"], pred),
                     "test_f1": f1_score(test["positif"], pred), "test_roc_auc": roc_auc_score(test["positif"], s),
                     "temps_s": time.perf_counter() - t0})
        print(f"  {name:48s} test accuracy {rows[-1]['test_accuracy']:.4f} · ROC AUC {rows[-1]['test_roc_auc']:.4f} "
              f"· {rows[-1]['temps_s']:.0f} s")

    # Combinaison : moyenne des scores standardisés de la régression logistique et du SVM
    duo = ["Régression logistique (mots et paires de mots)", "SVM linéaire (morceaux de mots)"]
    vs = {m: scores(fitted[m], val["texte"]) for m in duo}
    std = {m: vs[m].std() for m in duo}
    comb = sum(test_scores[m] / std[m] for m in duo) / 2
    comb_val = sum(vs[m] / std[m] for m in duo) / 2
    name = "Combinaison (logistique + SVM)"
    test_scores[name] = comb
    pred = (comb >= 0).astype(int)
    rows.append({"approche": name, "parametre": None, "val_accuracy": accuracy_score(val["positif"], (comb_val >= 0).astype(int)),
                 "test_accuracy": accuracy_score(test["positif"], pred), "test_f1": f1_score(test["positif"], pred),
                 "test_roc_auc": roc_auc_score(test["positif"], comb), "temps_s": None})
    table = pd.DataFrame(rows).sort_values("val_accuracy", ascending=False).reset_index(drop=True)
    best = table.iloc[0]["approche"]
    print(f"Approche retenue (meilleure validation) : {best}")
    best_pred = (test_scores[best] >= 0).astype(int)
    cm = confusion_matrix(test["positif"], best_pred)

    # ─── Courbe d'apprentissage (régression logistique) ───
    lr_name = duo[0]
    lr_c = table.set_index("approche").loc[lr_name, "parametre"]
    sizes = [500, 2000, 5000, 20000, 50000, 160000]
    curve = []
    for n in sizes:
        sub = train.sample(n, random_state=SEED)
        words = dict(ngram_range=(1, 2), min_df=min(3, max(1, n // 2000)), max_df=0.9, sublinear_tf=True, max_features=300_000)
        m = make_pipeline(TfidfVectorizer(**words), LogisticRegression(C=lr_c, max_iter=3000)).fit(sub["texte"], sub["positif"])
        curve.append({"critiques": n, "test_accuracy": accuracy_score(test["positif"], m.predict(test["texte"]))})
        print(f"  courbe d'apprentissage : {n:>6} critiques -> {curve[-1]['test_accuracy']:.4f}")

    # ─── Mots qui comptent (régression logistique) ───
    lr = fitted[lr_name]
    vocab = lr[0].get_feature_names_out()
    coef = lr[-1].coef_[0]
    order = np.argsort(coef)
    top_neg = [(vocab[i], float(coef[i])) for i in order[:15]]
    top_pos = [(vocab[i], float(coef[i])) for i in order[::-1][:15]]

    # ─── Analyse des erreurs ───
    t = test.assign(pred=best_pred, mots=test["texte"].str.split().str.len())
    t["longueur"] = pd.qcut(t["mots"], 4, labels=["courtes", "moyennes", "longues", "très longues"])
    err_len = (1 - t.assign(ok=t["pred"] == t["positif"]).groupby("longueur", observed=True)["ok"].mean()) * 100
    neg_words = r"\b(pas|jamais|rien|aucun|aucune|ni)\b"
    has_neg = t["texte"].str.lower().str.contains(neg_words)
    err_neg = {"avec négation": float((t.loc[has_neg, "pred"] != t.loc[has_neg, "positif"]).mean() * 100),
               "sans négation": float((t.loc[~has_neg, "pred"] != t.loc[~has_neg, "positif"]).mean() * 100)}
    word_counts = {"mediane": float(t["mots"].median()), "moyenne": float(t["mots"].mean())}

    # ─── Démonstration sur des phrases écrites pour le projet ───
    demo_texts = [e[0] for e in EXEMPLES]
    demo = []
    for name_ in ["Règles (liste de mots)", lr_name, "SVM linéaire (morceaux de mots)"]:
        p = (scores(fitted[name_], demo_texts) >= 0).astype(int)
        demo.append(p)
    demo_rows = [{"phrase": e[0], "attendu": e[1], "regles": int(demo[0][i]), "logistique": int(demo[1][i]), "svm": int(demo[2][i])}
                 for i, e in enumerate(EXEMPLES)]

    # ─── Figures ───
    fig_comparaison(table)
    fig_courbe(curve)
    fig_mots(top_pos, top_neg)
    fig_confusion(cm, best)
    fig_erreurs(err_len, err_neg)

    # le modèle sauvegardé est la régression logistique (rapide, explicable) : on la réentraîne si besoin ailleurs
    joblib.dump({"modele": fitted[lr_name], "nom": lr_name}, MODELS / "model.joblib", compress=3)
    report = {
        "donnees": {"entrainement": len(train), "validation": len(val), "test": len(test),
                    "part_positives": round(float(train["positif"].mean()), 3), "mots_par_critique": word_counts},
        "approche_retenue": best,
        "resultats": json.loads(table.round(4).to_json(orient="records", force_ascii=False)),
        "matrice_confusion_test": {"vrais_negatifs": int(cm[0, 0]), "faux_positifs": int(cm[0, 1]),
                                   "faux_negatifs": int(cm[1, 0]), "vrais_positifs": int(cm[1, 1])},
        "courbe_apprentissage": curve,
        "mots_positifs": [{"mot": w, "poids": round(c, 3)} for w, c in top_pos],
        "mots_negatifs": [{"mot": w, "poids": round(c, 3)} for w, c in top_neg],
        "erreur_selon_longueur_pct": {str(k): round(float(v), 2) for k, v in err_len.items()},
        "erreur_selon_negation_pct": {k: round(v, 2) for k, v in err_neg.items()},
        "exemples": demo_rows,
    }
    (REPORTS / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(table[["approche", "val_accuracy", "test_accuracy", "test_roc_auc"]].round(4).to_string(index=False))
    print("Erreur selon la longueur :", err_len.round(2).to_dict(), "· négation :", err_neg)
    for r in demo_rows:
        print(f"  attendu {r['attendu']} | règles {r['regles']} logistique {r['logistique']} svm {r['svm']} | {r['phrase']}")


# ─────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────
def fig_comparaison(table):
    t = table.sort_values("test_accuracy")
    fig, ax = plt.subplots(figsize=(9, 3.8))
    colors = [GREY if a.startswith("Règles") else RED if a.startswith("Combinaison") else BLUE for a in t["approche"]]
    bars = ax.barh(t["approche"], t["test_accuracy"] * 100, color=colors)
    for b, v in zip(bars, t["test_accuracy"] * 100):
        ax.text(v + 0.2, b.get_y() + b.get_height() / 2, f"{v:.1f} %", va="center", fontsize=9)
    ax.set_xlim(50, 100)
    ax.set_xlabel("Critiques bien classées sur le test (20 000 critiques, 50 % = hasard)")
    ax.set_title("Positive ou négative ? Comparaison des approches")
    ax.grid(axis="y", visible=False)
    _save(fig, "comparaison_approches")


def fig_courbe(curve):
    c = pd.DataFrame(curve)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(c["critiques"], c["test_accuracy"] * 100, marker="o", color=BLUE, lw=2)
    for _, r in c.iterrows():
        ax.text(r["critiques"], r["test_accuracy"] * 100 - 0.9, f"{r['test_accuracy'] * 100:.1f} %", ha="center", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Nombre de critiques d'entraînement (échelle logarithmique)")
    ax.set_ylabel("Critiques bien classées sur le test (%)")
    ax.set_title("Plus de données, meilleur modèle… avec des rendements décroissants")
    _save(fig, "courbe_apprentissage")


def fig_mots(pos, neg):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, words, color, title in [(axes[0], neg, RED, "Les mots qui pèsent le plus vers « négatif »"),
                                    (axes[1], pos, GREEN, "Les mots qui pèsent le plus vers « positif »")]:
        w = words[::-1]
        ax.barh([x[0] for x in w], [abs(x[1]) for x in w], color=color)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("Poids dans la régression logistique")
    _save(fig, "mots_importants")


def fig_confusion(cm, name):
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}".replace(",", " "), ha="center", va="center", fontsize=12,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["prédit négatif", "prédit positif"])
    ax.set_yticks([0, 1], ["vraiment négatif", "vraiment positif"])
    ax.grid(False)
    ax.set_title(f"Matrice de confusion (test)\n{name}", fontsize=9.5)
    _save(fig, "matrice_confusion")


def fig_erreurs(err_len, err_neg):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [2, 1.2]})
    axes[0].bar(err_len.index.astype(str), err_len.values, color=BLUE)
    axes[0].set_title("Taux d'erreur selon la longueur de la critique", fontsize=10)
    axes[0].set_ylabel("Erreurs (%)")
    axes[1].bar(list(err_neg), list(err_neg.values()), color=[ORANGE, GREY])
    axes[1].set_title("…et selon la présence d'une négation", fontsize=10)
    for ax in axes:
        ax.grid(axis="x", visible=False)
    _save(fig, "analyse_erreurs")


if __name__ == "__main__":
    main()
