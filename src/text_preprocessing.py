"""Pipeline de preprocesamiento de texto para AG News (Data Science III · Módulo 2).

preprocess_text(text) = clean_text (regex, ruido estructural) + normalize (minúsculas + lemas spaCy).
"""
import html
import re

import spacy

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])  # solo lo necesario para lematizar

# Negaciones: spaCy las marca como stop-words, pero cambian el sentido de la frase -> se conservan
NEGACIONES = {"no", "not", "nor", "never", "n't", "nothing", "none"}
STOP_WORDS = nlp.Defaults.stop_words - NEGACIONES

# Agencias y fuentes que firman las noticias de AG News
AGENCIAS = r"Reuters|AP|AFP|Canadian Press|CP|SPACE\.com|Sports Network|NewsFactor|TechWeb|Forbes\.com|washingtonpost\.com|USATODAY\.com|CNET|PC World|Ziff Davis|MarketWatch|Business Week|eWEEK|Investor's Business Daily"

_RE_ENTIDAD_ROTA = re.compile(r"(?<![&\w])#(\d+);")      # "#39;"  -> "&#39;"  (apóstrofe)
_RE_QUOT_ROTO = re.compile(r"(?<![&\w])(quot|amp|lt|gt);")  # "quot;" -> "&quot;"
_RE_HTML = re.compile(r"<.*?>", flags=re.S)                 # etiquetas HTML (no codicioso)
_RE_URL = re.compile(r"https?://\S+|www\.\S+|\b[\w.-]+\.(?:com|net|org)\b\S*", flags=re.I)
_RE_EMAIL = re.compile(r"\S+@\S+\.\S+")
_RE_FUENTE_PAREN = re.compile(rf"\((?:{AGENCIAS})\)", flags=re.I)        # "(Reuters)"
_RE_FUENTE_GUION = re.compile(rf"(?:^|(?<=\. ))(?:{AGENCIAS})\s+-\s+", flags=re.I)  # "AP - " al inicio del cuerpo
_RE_QUOTE_PROFILE = re.compile(r"\(?\s*(?:[A-Z0-9.]+\s*:\s*)?Quote,\s*Profile,\s*Research\s*\)?")  # links de Reuters
_RE_UPDATE = re.compile(r"\bUPDATE\s*\d*\s*-\s*")                    # "UPDATE 1-" (versión del cable)
_RE_FUENTE_CIUDAD = re.compile(r"\([A-Za-z ,.]+-AP\)")                 # "(Washington-AP)"
# Dateline al inicio del cuerpo: "NEW YORK - ", "BAGHDAD, Iraq -- ", "LONDON (Reuters) - "
_RE_DATELINE = re.compile(r"(?:^|(?<=\. ))[A-Z][A-Z.'\- ]{2,}(?:,\s*[A-Z][A-Za-z.]+)*\s*(?:\([^)]*\))?\s*-{1,2}\s+")
_RE_TICKER = re.compile(r"\b[A-Z]{2,5}\.(?:N|O|OQ|PK|OB|L|TO|DE|PA|AS|T|HK|MI|F)\b")  # tickers "PEP.N", "AKLM.PK" (no toca U.S. ni U.N.)
_RE_ESPACIOS = re.compile(r"\s+")
_RE_US = re.compile(r"\bUS\b")  # "US" (país) -> "U.S.": en minúsculas spaCy lo confunde con el pronombre "us" (stop-word)


def clean_text(text: str) -> str:
    """Limpieza estructural con regex (no toca el significado del texto)."""
    text = _RE_ENTIDAD_ROTA.sub(r"&#\1;", text)
    text = _RE_QUOT_ROTO.sub(r"&\1;", text)
    text = html.unescape(html.unescape(text))   # dos veces: hay entidades doblemente escapadas (&amp;lt;)
    text = _RE_HTML.sub(" ", text)
    text = _RE_URL.sub(" ", text)
    text = _RE_EMAIL.sub(" ", text)
    text = _RE_QUOTE_PROFILE.sub(" ", text)
    text = _RE_UPDATE.sub(" ", text)
    text = _RE_FUENTE_CIUDAD.sub(" ", text)
    for _ in range(2):                          # 2 pasadas: "AP - BEIJING - ..." tiene firma + dateline encadenados
        text = _RE_FUENTE_GUION.sub("", text)   # "Canadian Press - "
        text = _RE_DATELINE.sub(" ", text)      # antes de quitar "(Reuters)", que forma parte del dateline
        text = _RE_ESPACIOS.sub(" ", text)
    text = _RE_FUENTE_PAREN.sub(" ", text)
    text = _RE_FUENTE_GUION.sub(" ", text)
    text = _RE_TICKER.sub(" ", text)
    text = _RE_ESPACIOS.sub(" ", text)
    return text.strip()


def _pre_lower(text: str) -> str:
    """Normalizaciones que dependen de las mayúsculas y por eso van antes de .lower()."""
    return _RE_US.sub("U.S.", clean_text(text))


def _lemmas(doc, remove_stopwords: bool):
    out = []
    for tok in doc:
        if tok.is_punct or tok.is_space or not any(c.isalnum() for c in tok.text):
            continue  # puntuación, espacios y símbolos sueltos ($, -, ...)
        lemma = tok.lemma_.strip()
        if remove_stopwords and (lemma in STOP_WORDS or tok.text in STOP_WORDS):
            continue
        out.append(lemma)
    return out


def preprocess_text(text: str, remove_stopwords: bool = False) -> str:
    """Limpieza regex + minúsculas + lematización spaCy. Devuelve los lemas unidos por espacios."""
    return " ".join(_lemmas(nlp(_pre_lower(text).lower()), remove_stopwords))


def preprocess_series(texts, remove_stopwords: bool = False, batch_size: int = 256):
    """Versión por lotes (nlp.pipe), mucho más rápida para corpus completos."""
    limpios = (_pre_lower(t).lower() for t in texts)
    return [" ".join(_lemmas(doc, remove_stopwords)) for doc in nlp.pipe(limpios, batch_size=batch_size)]
