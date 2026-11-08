"""Clasificador supervisado TF-IDF sobre AG News (DS III · Módulo 3).

Flujo: carga (loader de la cátedra) -> preprocesamiento del Módulo 2 -> TF-IDF -> clasificador -> evaluación.
Uso:  python -m src.train_baseline      (desde la raíz del repo)

Regla anti-leakage: el TfidfVectorizer vive DENTRO de un sklearn.Pipeline. Así, tanto en la validación
cruzada como en el entrenamiento final, el vocabulario y el IDF se aprenden solo con datos de entrenamiento
(fit_transform) y al test solo se le aplica transform.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PROCESSED = DATA / "processed"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
SEED = 42

sys.path.insert(0, str(DATA))
from cargar_dataset import cargar  # noqa: E402  (loader provisto por la cátedra)

CLASES = ["Business", "Sci_Tech", "Sports", "World"]

# Parámetros fijos del vectorizador (los que se exploran van en la grilla)
TFIDF_FIJOS = dict(
    token_pattern=r"\S+",   # el texto ya viene tokenizado/lematizado por spaCy: respetamos esos tokens
    lowercase=False,        # ya está en minúsculas
    min_df=2,               # descarta términos que aparecen en un solo documento (typos, nombres únicos)
    max_df=0.95,            # descarta términos presentes en >95% de los documentos
    sublinear_tf=True,      # 1 + log(tf): amortigua palabras repetidas dentro de una misma noticia
    norm="l2",              # todos los documentos en la hiperesfera unitaria (sin sesgo por longitud)
)

MODELOS = {
    "Naive Bayes": lambda **kw: MultinomialNB(**{"alpha": 0.1, **kw}),
    "Regresión Logística": lambda **kw: LogisticRegression(**{"C": 10, "max_iter": 2000, "random_state": SEED, **kw}),
    "SVM lineal": lambda **kw: LinearSVC(**{"C": 0.5, "random_state": SEED, **kw}),
}

# Hiperparámetro de regularización de cada modelo (se ajusta con CV en la etapa 2)
GRILLA_REG = {
    "Naive Bayes": ("alpha", [0.01, 0.05, 0.1, 0.3, 1.0]),
    "Regresión Logística": ("C", [1, 5, 10, 30, 100]),
    "SVM lineal": ("C", [0.1, 0.25, 0.5, 1.0, 2.0]),
}


# ---------------------------------------------------------------- datos y preprocesamiento
def cargar_y_preprocesar(usar_cache: bool = True):
    """Carga los splits de la cátedra y aplica el preprocesamiento del Módulo 2 (con caché en data/processed)."""
    train_df, test_df = cargar("ag_news")
    PROCESSED.mkdir(exist_ok=True)
    salida = []
    for nombre, df in (("train", train_df), ("test", test_df)):
        cache = PROCESSED / f"ag_news_{nombre}_proc.csv"
        if usar_cache and cache.exists():
            df = pd.read_csv(cache).fillna("")
        else:
            from .text_preprocessing import preprocess_series  # import diferido: carga spaCy solo si hace falta
            df = df.copy()
            df["text_nostop"] = preprocess_series(df.text, remove_stopwords=True)
            df.to_csv(cache, index=False)
        salida.append(df)
    return salida


def validar(train_df, test_df) -> pd.DataFrame:
    return pd.DataFrame({
        "filas": [len(train_df), len(test_df)],
        "nulos": [int(train_df[["text", "label"]].isna().sum().sum()), int(test_df[["text", "label"]].isna().sum().sum())],
        "duplicados": [int(train_df.text.duplicated().sum()), int(test_df.text.duplicated().sum())],
        "test_en_train": ["—", int(test_df.text.isin(train_df.text).sum())],
        "docs_vacios_tras_preproc": [int((train_df.text_nostop.str.len() == 0).sum()), int((test_df.text_nostop.str.len() == 0).sum())],
    }, index=["train", "test"])


# ---------------------------------------------------------------- modelo
def construir_pipeline(modelo: str, max_features=None, ngram_range=(1, 1), clf_params=None, **tfidf_extra) -> Pipeline:
    params = {**TFIDF_FIJOS, **tfidf_extra}
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, **params)),
        ("clf", MODELOS[modelo](**(clf_params or {}))),
    ])


def _cv():
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


def _parse(fila):
    ng = tuple(int(v) for v in str(fila["ngram_range"]).strip("()").split(","))
    mf = None if fila["max_features"] == "sin límite" else int(fila["max_features"])
    return ng, mf


def ajustar_regularizacion(X, y, exp: pd.DataFrame) -> pd.DataFrame:
    """Para cada modelo, toma su mejor configuración del vectorizador y barre su hiperparámetro de regularización."""
    filas = []
    for m, (param, valores) in GRILLA_REG.items():
        ng, mf = _parse(exp[exp.modelo == m].iloc[0])
        for v in valores:
            r = cross_validate(construir_pipeline(m, mf, ng, {param: v}), X, y, cv=_cv(), scoring=["accuracy", "f1_macro"], n_jobs=-1)
            filas.append(dict(modelo=m, ngram_range=str(ng), max_features=mf if mf else "sin límite", parametro=param, valor=v,
                              f1_macro_cv=r["test_f1_macro"].mean(), f1_std=r["test_f1_macro"].std(),
                              accuracy_cv=r["test_accuracy"].mean()))
    return pd.DataFrame(filas)


def experimentar(X, y, modelos=None, ngrams=((1, 1), (1, 2)), max_feats=(2000, 5000, 10000, None), cv=5):
    """Validación cruzada estratificada SOLO sobre train. El test no se toca en esta etapa."""
    modelos = modelos or list(MODELOS)
    skf = _cv()
    filas = []
    for m in modelos:
        for ng in ngrams:
            for mf in max_feats:
                pipe = construir_pipeline(m, mf, ng)
                t0 = time.time()
                r = cross_validate(pipe, X, y, cv=skf, scoring=["accuracy", "f1_macro"], n_jobs=-1)
                vocab = len(construir_pipeline(m, mf, ng).named_steps["tfidf"].fit(X).vocabulary_)
                filas.append(dict(modelo=m, ngram_range=str(ng), max_features=mf if mf else "sin límite",
                                  vocabulario=vocab,
                                  f1_macro_cv=r["test_f1_macro"].mean(), f1_std=r["test_f1_macro"].std(),
                                  accuracy_cv=r["test_accuracy"].mean(),
                                  seg_por_fold=round(float(np.mean(r["fit_time"])), 2)))
    return pd.DataFrame(filas).sort_values("f1_macro_cv", ascending=False).reset_index(drop=True)


def top_terminos(pipe: Pipeline, k: int = 12) -> pd.DataFrame:
    """Términos con mayor peso positivo por clase (interpretabilidad de modelos lineales)."""
    vocab = pipe.named_steps["tfidf"].get_feature_names_out()
    clf = pipe.named_steps["clf"]
    pesos = clf.feature_log_prob_ if hasattr(clf, "feature_log_prob_") else clf.coef_
    return pd.DataFrame({c: [vocab[i] for i in np.argsort(pesos[j])[::-1][:k]] for j, c in enumerate(clf.classes_)})


def evaluar(pipe: Pipeline, X_test, y_test):
    pred = pipe.predict(X_test)
    rep = classification_report(y_test, pred, digits=3, output_dict=True)
    cm = confusion_matrix(y_test, pred, labels=CLASES)
    return pred, rep, cm


def graficar_confusion(cm, path, titulo):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, data, fmt, t in [(axes[0], cm, "d", "Cantidades"), (axes[1], cm_norm, ".1%", "Normalizada por clase real (recall)")]:
        im = ax.imshow(data, cmap="Oranges")
        ax.set_xticks(range(4), CLASES); ax.set_yticks(range(4), CLASES)
        ax.set_xlabel("Clase predicha"); ax.set_ylabel("Clase real"); ax.set_title(t)
        umbral = data.max() / 2
        for i in range(4):
            for j in range(4):
                v = format(data[i, j], fmt)
                ax.text(j, i, v, ha="center", va="center", color="white" if data[i, j] > umbral else "black", fontsize=10)
        ax.grid(False)
    fig.suptitle(titulo, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return fig


# ---------------------------------------------------------------- script
def main():
    REPORTS.mkdir(exist_ok=True); FIGURES.mkdir(parents=True, exist_ok=True)
    train_df, test_df = cargar_y_preprocesar()
    print(validar(train_df, test_df), "\n")

    X_tr, y_tr = train_df.text_nostop, train_df.label
    X_te, y_te = test_df.text_nostop, test_df.label

    res = experimentar(X_tr, y_tr)
    res.to_csv(REPORTS / "experimentos_cv.csv", index=False)
    print(res.round(4).to_string(), "\n")

    reg = ajustar_regularizacion(X_tr, y_tr, res)
    reg.to_csv(REPORTS / "regularizacion_cv.csv", index=False)
    mejor = reg.sort_values("f1_macro_cv", ascending=False).iloc[0]
    print(reg.round(4).to_string(), "\n")
    ng, mf = _parse(mejor)
    final = construir_pipeline(mejor.modelo, mf, ng, {mejor.parametro: mejor.valor}).fit(X_tr, y_tr)   # fit SOLO con train
    pred, rep, cm = evaluar(final, X_te, y_te)                           # transform + predict en test
    print(classification_report(y_te, pred, digits=3))
    graficar_confusion(cm, FIGURES / "matriz_confusion.png", f"{mejor.modelo} · TF-IDF {mejor.ngram_range} · test")

    resumen = dict(modelo=mejor.modelo, ngram_range=mejor.ngram_range, max_features=str(mejor.max_features),
                   **{mejor.parametro: float(mejor.valor)},
                   f1_macro_cv=round(float(mejor.f1_macro_cv), 4),
                   test=dict(accuracy=round(accuracy_score(y_te, pred), 4), f1_macro=round(f1_score(y_te, pred, average="macro"), 4)))
    (REPORTS / "resumen.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False))
    print(json.dumps(resumen, indent=2, ensure_ascii=False))
    return resumen


if __name__ == "__main__":
    main()
