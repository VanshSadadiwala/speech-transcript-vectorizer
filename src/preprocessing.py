"""Text preprocessing for raw speech-to-text (ASR) transcripts.

Steps (each can be switched on/off through PreprocessConfig):
  1. lowercase
  2. remove acoustic noise tags such as [noise], (laughs), <unk>, *cough*
  3. expand informal contractions (gonna -> going to, what's -> what is)
  4. remove spoken fillers (um, uh, you know, i mean ...)
  5. normalise numbers to the token "num"
  6. strip punctuation and collapse elongated letters (sooo -> soo)
  7. tokenize
  8. stop-word removal (NLTK list, but command keywords such as on/off/up/down
     are PROTECTED so that phrases like "turn off" survive)
  9. lemmatize (WordNet; falls back to simple suffix rules if WordNet is missing)

The module works even if the NLTK corpora cannot be downloaded (offline use).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, List

# --------------------------------------------------------------------------
# NLTK resources (optional, with graceful fallback)
# --------------------------------------------------------------------------
_FALLBACK_STOPWORDS = {
    "a", "about", "after", "again", "all", "am", "an", "and", "any", "are", "as", "at", "be",
    "because", "been", "before", "being", "between", "both", "but", "by", "can", "could", "did",
    "do", "does", "doing", "during", "each", "few", "for", "from", "further", "had", "has",
    "have", "having", "he", "her", "here", "hers", "him", "his", "how", "i", "if", "in", "into",
    "is", "it", "its", "just", "me", "more", "most", "my", "myself", "of", "once", "only", "or",
    "other", "our", "ours", "own", "same", "she", "should", "so", "some", "such", "than", "that",
    "the", "their", "them", "then", "there", "these", "they", "this", "those", "through", "to",
    "too", "under", "until", "very", "was", "we", "were", "what", "when", "where", "which",
    "while", "who", "whom", "why", "will", "with", "would", "you", "your", "yours",
    "off", "on", "up", "down", "out", "over", "above", "below", "not", "no", "nor",
}

_RESOURCES_READY = False
_STOPWORDS_NLTK: set = set()
_LEMMATIZER = None


def ensure_nltk_data(verbose: bool = False) -> dict:
    """Load (and if needed download) NLTK stopwords + WordNet. Never raises."""
    global _RESOURCES_READY, _STOPWORDS_NLTK, _LEMMATIZER
    status = {"stopwords": False, "wordnet": False}
    if _RESOURCES_READY:
        return {"stopwords": bool(_STOPWORDS_NLTK), "wordnet": _LEMMATIZER is not None}
    try:
        import nltk
        from nltk.corpus import stopwords, wordnet
        from nltk.stem import WordNetLemmatizer

        for pkg, path in (("stopwords", "corpora/stopwords"), ("wordnet", "corpora/wordnet")):
            try:
                nltk.data.find(path)
            except LookupError:
                try:
                    nltk.download(pkg, quiet=not verbose)
                except Exception:
                    pass
        try:
            _STOPWORDS_NLTK = set(stopwords.words("english"))
            status["stopwords"] = True
        except Exception:
            _STOPWORDS_NLTK = set()
        try:
            wordnet.ensure_loaded()
            _LEMMATIZER = WordNetLemmatizer()
            _LEMMATIZER.lemmatize("tests")  # force a real lookup
            status["wordnet"] = True
        except Exception:
            _LEMMATIZER = None
    except Exception:
        pass
    _RESOURCES_READY = True
    return status


# --------------------------------------------------------------------------
# Lexical resources
# --------------------------------------------------------------------------
CONTRACTIONS = {
    "gonna": "going to", "wanna": "want to", "lemme": "let me", "gimme": "give me",
    "gotta": "got to", "kinda": "kind of", "sorta": "sort of", "dunno": "do not know",
    "ya": "you", "cuz": "because", "cos": "because",
    "i'm": "i am", "i've": "i have", "i'll": "i will", "i'd": "i would",
    "you're": "you are", "you've": "you have", "you'll": "you will",
    "he's": "he is", "she's": "she is", "it's": "it is", "that's": "that is",
    "what's": "what is", "where's": "where is", "how's": "how is", "there's": "there is",
    "let's": "let us", "we're": "we are", "they're": "they are",
    "can't": "can not", "won't": "will not", "don't": "do not", "doesn't": "does not",
    "didn't": "did not", "isn't": "is not", "aren't": "are not", "wasn't": "was not",
    "couldn't": "could not", "wouldn't": "would not", "shouldn't": "should not",
}

FILLER_PHRASES = ["you know", "i mean", "sort of", "kind of"]
FILLER_WORDS = ["um", "umm", "uh", "uhh", "er", "erm", "hmm", "hm", "mm", "mmm", "ah", "eh"]

# Domain stop words for voice assistants (wake words, politeness, discourse markers)
DOMAIN_STOPWORDS = {
    "hey", "alexa", "siri", "google", "okay", "ok", "please", "thanks", "thank", "kindly",
    "gonna", "going", "want", "wanna", "need", "let", "like", "help", "yeah", "yes",
}

# Command keywords that must NEVER be dropped even though NLTK lists some as stop words
PROTECTED_KEYWORDS = {
    "on", "off", "up", "down", "out", "no", "not", "stop", "start", "play", "pause", "call",
    "set", "turn", "open", "close", "next", "previous", "back", "over", "mute",
}

# Words WordNet would wrongly lemmatize
LEMMA_EXCEPTIONS = {"conditioning", "media", "news", "bus", "gas", "its", "this", "was", "has", "is", "as"}

NOISE_TAG_RE = re.compile(r"\[[^\]]*\]|\([^)]*\)|<[^>]*>|\*[^*]*\*")
NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)?\b")
ELONGATION_RE = re.compile(r"(\w)\1{2,}")
NON_ALPHA_RE = re.compile(r"[^a-z\s]")
SPACE_RE = re.compile(r"\s+")
TOKEN_RE = re.compile(r"[a-z]+")
_CONTRACTION_RE = re.compile(r"\b(" + "|".join(re.escape(k) for k in sorted(CONTRACTIONS, key=len, reverse=True)) + r")\b")
_FILLER_PHRASE_RE = re.compile(r"\b(" + "|".join(FILLER_PHRASES) + r")\b")
_FILLER_WORD_RE = re.compile(r"\b(" + "|".join(FILLER_WORDS) + r")\b")


def _fallback_lemma(w: str) -> str:
    """Very small suffix-rule lemmatizer used only when WordNet is unavailable."""
    if w in LEMMA_EXCEPTIONS or len(w) <= 3:
        return w
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("ing") and len(w) > 5:
        stem = w[:-3]
        return stem[:-1] if len(stem) > 2 and stem[-1] == stem[-2] else stem
    if w.endswith("ed") and len(w) > 4:
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        return w[:-1]
    return w


@dataclass
class PreprocessConfig:
    lowercase: bool = True
    remove_noise_tags: bool = True
    expand_contractions: bool = True
    remove_fillers: bool = True
    normalize_numbers: bool = True
    remove_stopwords: bool = True
    lemmatize: bool = True
    protect_keywords: bool = True
    extra_stopwords: set = field(default_factory=set)


class TranscriptPreprocessor:
    """Callable preprocessing pipeline: raw transcript string -> list of tokens."""

    def __init__(self, config: PreprocessConfig | None = None):
        self.config = config or PreprocessConfig()
        self.resources = ensure_nltk_data()
        base = _STOPWORDS_NLTK or _FALLBACK_STOPWORDS
        sw = set(base) | DOMAIN_STOPWORDS | set(self.config.extra_stopwords)
        if self.config.protect_keywords:
            sw -= PROTECTED_KEYWORDS
        self.stopwords = sw

    # ---- individual steps ----
    def clean_text(self, text: str) -> str:
        c = self.config
        t = str(text)
        if c.lowercase:
            t = t.lower()
        if c.remove_noise_tags:
            t = NOISE_TAG_RE.sub(" ", t)
        t = t.replace("\u2019", "'")
        if c.expand_contractions:
            t = _CONTRACTION_RE.sub(lambda m: CONTRACTIONS[m.group(0)], t)
        if c.remove_fillers:
            t = _FILLER_PHRASE_RE.sub(" ", t)
            t = _FILLER_WORD_RE.sub(" ", t)
        if c.normalize_numbers:
            t = NUMBER_RE.sub(" num ", t)
        t = ELONGATION_RE.sub(r"\1\1", t)
        t = NON_ALPHA_RE.sub(" ", t)
        return SPACE_RE.sub(" ", t).strip()

    def tokenize(self, text: str) -> List[str]:
        return TOKEN_RE.findall(text)

    def lemma(self, w: str) -> str:
        if w in LEMMA_EXCEPTIONS or w == "num":
            return w
        if _LEMMATIZER is None:
            return _fallback_lemma(w)
        v = _LEMMATIZER.lemmatize(w, "v")
        return v if v != w else _LEMMATIZER.lemmatize(w, "n")

    def process(self, text: str) -> List[str]:
        toks = self.tokenize(self.clean_text(text))
        if self.config.remove_stopwords:
            toks = [t for t in toks if t not in self.stopwords]
        if self.config.lemmatize:
            toks = [self.lemma(t) for t in toks]
        return toks

    def process_many(self, texts: Iterable[str]) -> List[List[str]]:
        return [self.process(t) for t in texts]

    __call__ = process


def preprocessing_stats(raw_texts: List[str], token_docs: List[List[str]]) -> dict:
    """Simple before/after statistics used in the evaluation report."""
    raw_tokens = [TOKEN_RE.findall(str(t).lower()) for t in raw_texts]
    raw_vocab = {w for d in raw_tokens for w in d}
    new_vocab = {w for d in token_docs for w in d}
    n_raw = sum(len(d) for d in raw_tokens)
    n_new = sum(len(d) for d in token_docs)
    return {
        "documents": len(raw_texts),
        "tokens_before": n_raw,
        "tokens_after": n_new,
        "token_reduction_pct": round(100 * (1 - n_new / max(n_raw, 1)), 1),
        "vocab_before": len(raw_vocab),
        "vocab_after": len(new_vocab),
        "vocab_reduction_pct": round(100 * (1 - len(new_vocab) / max(len(raw_vocab), 1)), 1),
        "avg_tokens_before": round(n_raw / max(len(raw_texts), 1), 2),
        "avg_tokens_after": round(n_new / max(len(raw_texts), 1), 2),
    }


if __name__ == "__main__":
    p = TranscriptPreprocessor()
    demo = "Hey um I'm gonna need you to TURN off the air conditioning [noise] you know"
    print("RAW   :", demo)
    print("CLEAN :", p.clean_text(demo))
    print("TOKENS:", p(demo))
