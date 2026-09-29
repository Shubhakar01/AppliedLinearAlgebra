"""
AME 5151 - Applied Linear Algebra Lab, Assignment 2
Semantic Tagging with GloVe and NumPy

Pipeline
    text file --> preprocess --> W (n x 50)   \
                                               >--> S = cosine similarities (n x 20)
    20 tags   ------------------> T (20 x 50) /         computed two ways:
                                                          (a) Vec class + Python loops
                                                          (b) NumPy: normalize rows, W_hat @ T_hat.T
    S --> verify the two agree --> max-pool ranking of the 20 tags

Usage
    python glove_tagging.py --text manipal_linkedin.txt
    python glove_tagging.py --text manipal_linkedin.txt --model-path glove-wiki-gigaword-50.gz

By default the model is fetched with gensim's downloader (cached after the
first download). --model-path lets you load a local word2vec-format file.
"""

import argparse
import re
import sys

import numpy as np

from vec import Vec

MODEL_NAME = "glove-wiki-gigaword-50"
EMBEDDING_DIM = 50
NUM_TAGS = 20
MAX_WORDS = 400

# --------------------------------------------------------------------------
# Part A - tag vocabulary (exactly 20). A tag may be several words, e.g.
# "higher education"; its vector is then the mean of its words' vectors.
# --------------------------------------------------------------------------
DEFAULT_TAGS = [
    "research", "innovation", "education", "university", "students",
    "faculty", "campus", "engineering", "medicine", "technology",
    "curriculum", "collaboration", "publication", "laboratory", "scholarship",
    "mentorship", "internship", "entrepreneurship", "accreditation", "alumni",
]

# --------------------------------------------------------------------------
# Part C - stopwords (defined by hand; no external library).
# STARTER_STOPWORDS is the set given in the assignment. ADDED_STOPWORDS are
# extra function words and LinkedIn/link noise (words of length <= 2 are
# dropped by the length filter anyway, so they are not listed).
# --------------------------------------------------------------------------
STARTER_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to",
    "in", "on", "for", "is", "are", "was", "were",
    "with", "at", "by", "from",
}
ADDED_STOPWORDS = {
    # pronouns / determiners / demonstratives
    "its", "this", "that", "these", "those", "our", "you", "your", "they",
    "their", "them", "she", "his", "her", "him", "who", "whom", "which",
    "what", "when", "where", "why", "how", "there", "here",
    # auxiliaries / conjunctions / prepositions / common adverbs
    "been", "being", "have", "has", "had", "does", "did", "will", "would",
    "can", "could", "should", "may", "might", "must", "not", "but", "than",
    "then", "also", "about", "into", "over", "more", "most", "other", "such",
    "only", "own", "same", "very", "just", "all", "any", "both", "each",
    "few", "too", "out", "off", "down", "again", "once", "while", "during",
    "because", "between", "through", "after", "before", "above", "below",
    "under", "some", "many", "much", "every", "across", "within", "without",
    "per", "via", "etc",
    # LinkedIn copy-paste / link noise
    "hashtag", "http", "https", "www", "com", "lnkd", "linkedin",
}
STOPWORDS = frozenset(STARTER_STOPWORDS | ADDED_STOPWORDS)

# Anything that is not a letter/digit/whitespace, plus the underscore
# (which \w would otherwise keep), counts as punctuation. This also removes
# curly quotes, dashes and emoji that appear in real LinkedIn text.
_PUNCTUATION_RE = re.compile(r"[^\w\s]|_")


# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------
def load_model(model_path=None):
    """Load glove-wiki-gigaword-50 (from a local file if model_path is given)."""
    if model_path:
        from gensim.models import KeyedVectors
        return KeyedVectors.load_word2vec_format(model_path)
    import gensim.downloader as api
    return api.load(MODEL_NAME)


# --------------------------------------------------------------------------
# Part C - preprocessing
# --------------------------------------------------------------------------
def preprocess_text(text, stopwords=STOPWORDS, min_length=3):
    """
    Apply preprocessing steps 1-5 in the required order:
      1. lowercase
      2. remove punctuation (each punctuation character becomes a space, so
         "state-of-the-art" -> 4 tokens and "university's" -> "university", "s")
      3. split into tokens
      4. remove stopwords
      5. remove tokens with length <= 2  (keep length >= min_length = 3)
    Repeated tokens are kept. Returns the list of remaining tokens x_1..x_m.
    Step 6 (vocabulary check) happens in build_text_matrix.
    """
    text = text.lower()
    text = _PUNCTUATION_RE.sub(" ", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in stopwords]
    tokens = [t for t in tokens if len(t) >= min_length]
    return tokens


# --------------------------------------------------------------------------
# Part D - tag matrix and text matrix
# --------------------------------------------------------------------------
def build_tag_matrix(model, tags):
    """
    Construct the tag matrix.

    Returns:
        tag_names: list[str] of exactly 20 tag names.
        T: float64 NumPy array of shape (20, 50).

    Single-word tag: row = model[tag].
    Multi-word tag ("higher education"): row = arithmetic mean of the
    vectors of its component words.

    Raises ValueError (never silently drops a tag) if there are not exactly
    20 distinct tags or if any tag/component word is not in the vocabulary.
    """
    tag_names = [t.strip() for t in tags]
    if len(tag_names) != NUM_TAGS:
        raise ValueError(f"Expected exactly {NUM_TAGS} tags, got {len(tag_names)}")
    if any(t == "" for t in tag_names):
        raise ValueError("Empty tag name")
    if len(set(tag_names)) != len(tag_names):
        raise ValueError("Duplicate tags in tag vocabulary")

    missing = []
    for tag in tag_names:
        for word in tag.lower().split():
            if word not in model.key_to_index:
                missing.append((word, tag))
    if missing:
        details = ", ".join(
            f"{w!r}" if w == t.lower() else f"{w!r} (in tag {t!r})" for w, t in missing
        )
        raise ValueError(f"Tag not in vocabulary: {details}")

    rows = []
    for tag in tag_names:
        vectors = np.asarray([model[w] for w in tag.lower().split()], dtype=np.float64)
        rows.append(vectors.mean(axis=0))
    T = np.vstack(rows)
    return tag_names, T


def build_text_matrix(model, tokens):
    """
    Construct the text matrix.

    Returns:
        in_vocab_tokens: tokens found in the model vocabulary, in original
                         order, repetitions preserved.
        oov_tokens: tokens not in the vocabulary (order/repetitions preserved).
        W: float64 array of shape (n, 50); row i is the vector of
           in_vocab_tokens[i]. If n == 0, W has shape (0, 50).
    """
    in_vocab_tokens = [t for t in tokens if t in model.key_to_index]
    oov_tokens = [t for t in tokens if t not in model.key_to_index]
    if in_vocab_tokens:
        W = np.asarray([model[t] for t in in_vocab_tokens], dtype=np.float64)
    else:
        W = np.empty((0, model.vector_size), dtype=np.float64)
    return in_vocab_tokens, oov_tokens, W


# --------------------------------------------------------------------------
# Part E - similarity matrix with the Vec class (reference implementation)
# --------------------------------------------------------------------------
def _check_matrices(W, T):
    if W.ndim != 2 or T.ndim != 2:
        raise ValueError("W and T must both be 2-D arrays")
    if W.shape[1] != T.shape[1]:
        raise ValueError(
            f"Dimension mismatch: W has {W.shape[1]} columns, T has {T.shape[1]}"
        )


def similarity_matrix_vector_class(W, T):
    """
    Compute the full cosine-similarity matrix with the Vec class and loops.

    S[i, j] = Vec(W[i]).cosine_similarity(Vec(T[j]))

    Two Vec instances are constructed for every (i, j) pair. NumPy is used
    only to store the result and to hand the rows over as Python floats;
    no NumPy dot/norm/matmul is used to compute a similarity.
    Returns S of shape (n_words, n_tags).
    """
    _check_matrices(W, T)
    n_words, n_tags = W.shape[0], T.shape[0]
    S = np.zeros((n_words, n_tags), dtype=np.float64)
    for i in range(n_words):
        for j in range(n_tags):
            u = Vec(W[i].tolist())   # .tolist() -> plain Python floats
            v = Vec(T[j].tolist())
            S[i, j] = u.cosine_similarity(v)
    return S


# --------------------------------------------------------------------------
# Part F - similarity matrix with NumPy
# --------------------------------------------------------------------------
def similarity_matrix_numpy(W, T):
    """
    Compute all text-word/tag cosine similarities at once:
        W_hat = rows of W divided by their norms
        T_hat = rows of T divided by their norms
        S     = W_hat @ T_hat.T          (n x 50) @ (50 x 20) -> (n x 20)
    Raises ValueError if any row of W or T has zero norm.
    """
    _check_matrices(W, T)
    W_norm = np.linalg.norm(W, axis=1, keepdims=True)
    T_norm = np.linalg.norm(T, axis=1, keepdims=True)
    if np.any(W_norm == 0):
        raise ValueError("W contains a zero-norm row; cosine similarity is undefined")
    if np.any(T_norm == 0):
        raise ValueError("T contains a zero-norm row; cosine similarity is undefined")
    W_hat = W / W_norm
    T_hat = T / T_norm
    return W_hat @ T_hat.T


# --------------------------------------------------------------------------
# Part H - max-pool ranking
# --------------------------------------------------------------------------
def rank_tags(tag_names, S, in_vocab_tokens):
    """
    score[j] = max_i S[i, j].

    Returns a list of (tag, score, best_token) tuples sorted by score in
    descending order. Scores that agree to 12 decimal places count as tied
    and keep the original tag order; this matters because several exact
    word-for-tag matches all have cosine 1 up to ~1e-16 rounding noise, and
    that noise should not decide the ranking. best_token is the text token
    that produced the maximum (the first one if several tie).
    """
    S = np.asarray(S)
    if S.ndim != 2 or S.shape[0] == 0:
        raise ValueError("S must be a non-empty 2-D array")
    if S.shape[0] != len(in_vocab_tokens):
        raise ValueError("S has a different number of rows than in_vocab_tokens")
    if S.shape[1] != len(tag_names):
        raise ValueError("S has a different number of columns than tag_names")

    scores = S.max(axis=0)
    best_rows = S.argmax(axis=0)
    order = np.argsort(-np.round(scores, 12), kind="stable")
    return [
        (tag_names[j], float(scores[j]), in_vocab_tokens[int(best_rows[j])])
        for j in order
    ]


def print_ranking(ranking, top_k=8, out=print):
    """Print the top_k ranked tags and note identical vs different matches."""
    out(f"{'Rank':<5}{'Tag':<20}{'Score':<9}{'Best-matching text word':<26}Match type")
    out("-" * 76)
    identical, different = [], []
    for rank, (tag, score, word) in enumerate(ranking[:top_k], start=1):
        kind = "identical word" if word == tag else "different word"
        (identical if word == tag else different).append(tag)
        out(f"{rank:<5}{tag:<20}{score:<9.4f}{word:<26}{kind}")
    out()
    out(f"Top-{top_k} tags matched by an identical word : {identical if identical else 'none'}")
    out(f"Top-{top_k} tags matched by a different word  : {different if different else 'none'}")


# --------------------------------------------------------------------------
# Full pipeline
# --------------------------------------------------------------------------
def run_pipeline(text, model, tags=DEFAULT_TAGS, top_k=8, out=print,
                 rtol=1e-7, atol=1e-9):
    """
    Run Parts C-H on `text`, printing everything the submission requires.
    Exits with an explanatory message if no in-vocabulary word remains.
    Returns a dict of the intermediate results (used by the unit tests).
    """
    n_raw = len(text.split())
    if n_raw > MAX_WORDS:
        out(f"WARNING: the text has {n_raw} words; the assignment allows at most {MAX_WORDS}.")

    tokens = preprocess_text(text)
    tag_names, T = build_tag_matrix(model, tags)
    in_vocab_tokens, oov_tokens, W = build_text_matrix(model, tokens)

    out("=== Token statistics ===")
    out(f"Tokens before preprocessing (whitespace-split raw text): {n_raw}")
    out(f"Tokens after preprocessing (steps 1-5)                 : {len(tokens)}")
    out(f"In-vocabulary tokens (n)                               : {len(in_vocab_tokens)}")
    out(f"Out-of-vocabulary tokens                               : {len(oov_tokens)}")
    distinct_oov = list(dict.fromkeys(oov_tokens))
    out(f"Distinct out-of-vocabulary tokens ({len(distinct_oov)}): {distinct_oov}")
    out()

    out("=== Checkpoint ===")
    out(f"T.shape = {T.shape}")
    out(f"W.shape = {W.shape}")
    out()

    if len(in_vocab_tokens) == 0:
        sys.exit(
            "Error: the input text contains no usable words (n = 0): after "
            "preprocessing, no token is in the embedding vocabulary. "
            "Provide a text with real words and re-run."
        )

    S_vector = np.array(similarity_matrix_vector_class(W, T))
    S_numpy = similarity_matrix_numpy(W, T)

    out("=== Verification ===")
    out(f"S_vector.shape = {S_vector.shape}, S_numpy.shape = {S_numpy.shape}")
    same = np.allclose(S_vector, S_numpy, rtol=rtol, atol=atol)
    out(f"np.allclose(S_vector, S_numpy, rtol={rtol}, atol={atol}) = {same}")
    max_diff = np.max(np.abs(S_vector - S_numpy))
    out(f"max |S_vector - S_numpy| = {max_diff:.3e}")
    out()

    ranking = rank_tags(tag_names, S_numpy, in_vocab_tokens)
    out(f"=== Top {top_k} tags (max pooling) ===")
    print_ranking(ranking, top_k=top_k, out=out)

    return {
        "tokens": tokens, "tag_names": tag_names, "T": T, "W": W,
        "in_vocab_tokens": in_vocab_tokens, "oov_tokens": oov_tokens,
        "S_vector": S_vector, "S_numpy": S_numpy,
        "allclose": bool(same), "max_diff": float(max_diff), "ranking": ranking,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Semantic tagging with GloVe (AME 5151, Assignment 2)")
    parser.add_argument("--text", default="manipal_linkedin.txt",
                        help="plain-text input file (default: manipal_linkedin.txt)")
    parser.add_argument("--model-path", default=None,
                        help="local word2vec-format GloVe file (default: gensim downloader)")
    parser.add_argument("--top", type=int, default=8, help="how many top tags to display")
    args = parser.parse_args(argv)

    try:
        with open(args.text, encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        sys.exit(f"Error: text file not found: {args.text}")
    if not text.strip():
        sys.exit(f"Error: text file is empty: {args.text}")

    model = load_model(args.model_path)
    run_pipeline(text, model, DEFAULT_TAGS, top_k=args.top)


if __name__ == "__main__":
    main()
