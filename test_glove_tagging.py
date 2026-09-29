"""
Unit tests for glove_tagging.py (AME 5151, Assignment 2).

Run with:
    python -m unittest test_glove_tagging -v
or:
    python test_glove_tagging.py

Most tests use a small deterministic FakeModel (random 50-d vectors), so they
run instantly and offline. The build_* and similarity functions only need
`model.key_to_index`, `model[word]` and `model.vector_size`, which is exactly
what FakeModel provides. The class TestWithRealGloVe repeats the key checks
on the real glove-wiki-gigaword-50 model; it is skipped automatically if the
model cannot be loaded (set GLOVE_PATH to a local file to avoid downloading).
"""

import os
import unittest

import numpy as np

from vec import Vec
import glove_tagging as gt

TAGS = gt.DEFAULT_TAGS  # 20 single-word tags
EXTRA_WORDS = ["quantum", "learning", "banana", "higher", "hello"]


class FakeModel:
    """Minimal stand-in for a gensim KeyedVectors object."""

    vector_size = 50

    def __init__(self, words, seed=0):
        rng = np.random.default_rng(seed)
        self.key_to_index = {w: i for i, w in enumerate(words)}
        self._vectors = rng.normal(size=(len(words), self.vector_size)).astype(np.float32)

    def __getitem__(self, word):
        return self._vectors[self.key_to_index[word]]


def make_model():
    return FakeModel(TAGS + EXTRA_WORDS)


class TestPreprocessing(unittest.TestCase):

    def test_steps_in_order(self):
        text = "The Research, at MANIPAL: it's #Innovative (2026)!! state-of-the-art AI."
        self.assertEqual(
            gt.preprocess_text(text),
            ["research", "manipal", "innovative", "2026", "state", "art"],
        )

    def test_repeated_tokens_are_kept(self):
        self.assertEqual(gt.preprocess_text("research research Research"),
                         ["research"] * 3)

    def test_length_filter_removes_two_letter_tokens(self):
        self.assertEqual(gt.preprocess_text("ai ml data"), ["data"])

    def test_linkedin_noise_removed(self):
        self.assertEqual(gt.preprocess_text("hashtag#Research https://lnkd.in/abc"),
                         ["research", "abc"])

    def test_curly_quotes_dashes_and_emoji_are_punctuation(self):
        self.assertEqual(gt.preprocess_text("\u201cResearch\u201d \u2014 students \U0001F393"),
                         ["research", "students"])


class TestTagMatrix(unittest.TestCase):

    def setUp(self):
        self.model = make_model()

    def test_1_tag_matrix_shape(self):
        """Required test 1: T.shape == (20, 50)."""
        names, T = gt.build_tag_matrix(self.model, TAGS)
        self.assertEqual(T.shape, (20, 50))
        self.assertEqual(len(names), 20)

    def test_single_word_row_equals_model_vector(self):
        names, T = gt.build_tag_matrix(self.model, TAGS)
        j = names.index("medicine")
        np.testing.assert_allclose(T[j], self.model["medicine"].astype(np.float64))

    def test_multiword_tag_is_mean_of_component_vectors(self):
        tags = TAGS[:-1] + ["higher education"]
        names, T = gt.build_tag_matrix(self.model, tags)
        expected = (self.model["higher"].astype(np.float64)
                    + self.model["education"].astype(np.float64)) / 2
        np.testing.assert_allclose(T[names.index("higher education")], expected)

    def test_7_oov_single_word_tag_raises(self):
        """Required test 7: an OOV tag raises; it is not silently dropped."""
        bad = TAGS[:-1] + ["notaword"]
        with self.assertRaisesRegex(ValueError, "notaword"):
            gt.build_tag_matrix(self.model, bad)

    def test_7_oov_component_of_multiword_tag_raises(self):
        bad = TAGS[:-1] + ["higher notaword"]
        with self.assertRaisesRegex(ValueError, "notaword"):
            gt.build_tag_matrix(self.model, bad)

    def test_dropping_a_tag_is_not_allowed(self):
        with self.assertRaises(ValueError):
            gt.build_tag_matrix(self.model, TAGS[:19])   # only 19 tags
        with self.assertRaises(ValueError):
            gt.build_tag_matrix(self.model, TAGS + ["hello"])  # 21 tags

    def test_duplicate_tags_rejected(self):
        with self.assertRaises(ValueError):
            gt.build_tag_matrix(self.model, TAGS[:-1] + [TAGS[0]])


class TestTextMatrix(unittest.TestCase):

    def setUp(self):
        self.model = make_model()

    def test_2_text_matrix_shape(self):
        """Required test 2: W.shape == (n, 50)."""
        tokens = ["research", "quantum", "banana", "hello"]
        in_vocab, oov, W = gt.build_text_matrix(self.model, tokens)
        self.assertEqual(len(in_vocab), 4)
        self.assertEqual(W.shape, (4, 50))

    def test_rows_match_tokens_and_repetitions_kept(self):
        tokens = ["quantum", "research", "quantum"]
        in_vocab, oov, W = gt.build_text_matrix(self.model, tokens)
        self.assertEqual(in_vocab, tokens)
        self.assertEqual(W.shape, (3, 50))
        np.testing.assert_allclose(W[0], W[2])
        np.testing.assert_allclose(W[1], self.model["research"].astype(np.float64))

    def test_6_oov_token_excluded_and_reported(self):
        """Required test 6: OOV tokens are not rows of W and are reported."""
        tokens = ["quantum", "zzzunknown", "research", "zzzunknown"]
        in_vocab, oov, W = gt.build_text_matrix(self.model, tokens)
        self.assertNotIn("zzzunknown", in_vocab)
        self.assertEqual(oov, ["zzzunknown", "zzzunknown"])
        self.assertEqual(W.shape, (2, 50))

    def test_no_in_vocab_tokens_gives_empty_matrix(self):
        in_vocab, oov, W = gt.build_text_matrix(self.model, ["zzz", "yyy"])
        self.assertEqual(in_vocab, [])
        self.assertEqual(W.shape, (0, 50))


class TestSimilarity(unittest.TestCase):

    def setUp(self):
        self.model = make_model()
        self.names, self.T = gt.build_tag_matrix(self.model, TAGS)
        self.tokens = ["research", "quantum", "learning", "banana", "quantum"]
        self.in_vocab, _, self.W = gt.build_text_matrix(self.model, self.tokens)
        self.S_vec = np.array(gt.similarity_matrix_vector_class(self.W, self.T))
        self.S_np = gt.similarity_matrix_numpy(self.W, self.T)

    def test_3_similarity_shapes(self):
        """Required test 3: S.shape == (n, 20) for both implementations."""
        n = len(self.in_vocab)
        self.assertEqual(self.S_vec.shape, (n, 20))
        self.assertEqual(self.S_np.shape, (n, 20))

    def test_4_pair_agrees_between_vec_class_and_numpy(self):
        """Required test 4: one (word, tag) pair, Vec class vs NumPy matrix."""
        i = self.in_vocab.index("quantum")
        j = self.names.index("medicine")
        cos = Vec(self.W[i].tolist()).cosine_similarity(Vec(self.T[j].tolist()))
        self.assertAlmostEqual(cos, self.S_np[i, j], places=10)
        # and against a plain textbook formula
        manual = float(self.W[i] @ self.T[j]
                       / (np.linalg.norm(self.W[i]) * np.linalg.norm(self.T[j])))
        self.assertAlmostEqual(cos, manual, places=10)

    def test_5_allclose(self):
        """Required test 5: np.allclose(S_vector, S_numpy) is True."""
        self.assertTrue(np.allclose(self.S_vec, self.S_np, rtol=1e-7, atol=1e-9))
        self.assertLess(np.max(np.abs(self.S_vec - self.S_np)), 1e-9)

    def test_entries_lie_in_minus_one_to_one(self):
        self.assertTrue(np.all(self.S_np <= 1 + 1e-12))
        self.assertTrue(np.all(self.S_np >= -1 - 1e-12))

    def test_word_identical_to_tag_scores_one(self):
        i = self.in_vocab.index("research")
        j = self.names.index("research")
        self.assertAlmostEqual(self.S_np[i, j], 1.0, places=9)
        self.assertAlmostEqual(self.S_vec[i, j], 1.0, places=9)

    def test_repeated_word_gives_identical_rows(self):
        np.testing.assert_allclose(self.S_np[1], self.S_np[4])

    def test_numpy_rejects_zero_norm_rows(self):
        W0 = self.W.copy()
        W0[0] = 0.0
        with self.assertRaises(ValueError):
            gt.similarity_matrix_numpy(W0, self.T)
        T0 = self.T.copy()
        T0[3] = 0.0
        with self.assertRaises(ValueError):
            gt.similarity_matrix_numpy(self.W, T0)

    def test_vec_class_rejects_zero_norm_rows(self):
        W0 = self.W.copy()
        W0[0] = 0.0
        with self.assertRaises(ValueError):
            gt.similarity_matrix_vector_class(W0, self.T)

    def test_dimension_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            gt.similarity_matrix_numpy(self.W[:, :10], self.T)


class TestVecCosine(unittest.TestCase):
    """Sanity checks of Vec.dot / Vec.cosine_similarity on easy vectors."""

    def test_dot(self):
        self.assertEqual(Vec([1, 2, 3]).dot(Vec([4, 5, 6])), 32)

    def test_orthogonal_parallel_antiparallel(self):
        self.assertAlmostEqual(Vec([1, 0]).cosine_similarity(Vec([0, 5])), 0.0)
        self.assertAlmostEqual(Vec([1, 2]).cosine_similarity(Vec([2, 4])), 1.0)
        self.assertAlmostEqual(Vec([1, 2]).cosine_similarity(Vec([-1, -2])), -1.0)

    def test_scale_invariance(self):
        a, b = Vec([1.0, 2.0, 3.0]), Vec([3.0, 1.0, 2.0])
        self.assertAlmostEqual(a.cosine_similarity(b),
                               (10 * a).cosine_similarity(b), places=9)

    def test_zero_vector_raises(self):
        with self.assertRaises(ValueError):
            Vec([0, 0]).cosine_similarity(Vec([1, 1]))

    def test_dimension_mismatch_raises(self):
        with self.assertRaises(TypeError):
            Vec([1, 2]).dot(Vec([1, 2, 3]))


class TestRanking(unittest.TestCase):

    def test_max_pool_order_and_best_token(self):
        S = np.array([[0.10, 0.90],
                      [0.50, 0.20],
                      [0.30, 0.95]])
        ranking = gt.rank_tags(["a", "b"], S, ["x", "y", "z"])
        self.assertEqual([r[0] for r in ranking], ["b", "a"])
        self.assertAlmostEqual(ranking[0][1], 0.95)
        self.assertEqual(ranking[0][2], "z")
        self.assertAlmostEqual(ranking[1][1], 0.50)
        self.assertEqual(ranking[1][2], "y")

    def test_single_strong_word_dominates_max_pool(self):
        # tag "medicine": one very similar word, all others unrelated
        S = np.array([[0.05], [0.02], [0.97], [0.01]])
        (tag, score, word), = gt.rank_tags(["medicine"], S, ["a", "b", "c", "d"])
        self.assertAlmostEqual(score, 0.97)
        self.assertEqual(word, "c")

    def test_ties_up_to_rounding_noise_keep_tag_order(self):
        # both columns are "1" up to floating-point noise; tag order decides
        S = np.array([[0.9999999999999999, 1.0000000000000002, 0.5]])
        ranking = gt.rank_tags(["a", "b", "c"], S, ["w"])
        self.assertEqual([r[0] for r in ranking], ["a", "b", "c"])

    def test_shape_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            gt.rank_tags(["a", "b"], np.zeros((3, 2)), ["x", "y"])
        with self.assertRaises(ValueError):
            gt.rank_tags(["a"], np.zeros((2, 2)), ["x", "y"])

    def test_empty_S_rejected(self):
        with self.assertRaises(ValueError):
            gt.rank_tags(["a"], np.zeros((0, 1)), [])


class TestPipeline(unittest.TestCase):

    def setUp(self):
        self.model = make_model()
        self.silent = lambda *args, **kwargs: None

    def test_full_pipeline_on_small_text(self):
        text = "Research and innovation: quantum learning, quantum banana! zzzunknown"
        res = gt.run_pipeline(text, self.model, TAGS, top_k=8, out=self.silent)
        self.assertEqual(res["T"].shape, (20, 50))
        self.assertEqual(res["W"].shape, (len(res["in_vocab_tokens"]), 50))
        self.assertEqual(res["S_numpy"].shape, (len(res["in_vocab_tokens"]), 20))
        self.assertTrue(res["allclose"])
        self.assertIn("zzzunknown", res["oov_tokens"])
        self.assertEqual(len(res["ranking"]), 20)
        # 'research' and 'innovation' appear verbatim -> score 1 for those tags
        top = {tag: (score, word) for tag, score, word in res["ranking"]}
        self.assertAlmostEqual(top["research"][0], 1.0, places=9)
        self.assertEqual(top["research"][1], "research")

    def test_no_usable_words_terminates_with_message(self):
        with self.assertRaises(SystemExit) as cm:
            gt.run_pipeline("the and of zzzunknown", self.model, TAGS, out=self.silent)
        self.assertIn("no usable words", str(cm.exception))


def _load_real_model():
    path = os.environ.get("GLOVE_PATH")
    if path:
        from gensim.models import KeyedVectors
        return KeyedVectors.load_word2vec_format(path)
    import gensim.downloader as api
    return api.load(gt.MODEL_NAME)


class TestWithRealGloVe(unittest.TestCase):
    """Key checks on the real glove-wiki-gigaword-50 model (skipped if unavailable)."""

    @classmethod
    def setUpClass(cls):
        try:
            cls.model = _load_real_model()
        except Exception as exc:  # no gensim, no network, bad path ...
            raise unittest.SkipTest(f"real GloVe model not available: {exc}")

    def test_default_tags_shape_and_membership(self):
        names, T = gt.build_tag_matrix(self.model, TAGS)
        self.assertEqual(T.shape, (20, 50))

    def test_multiword_tag_in_real_vocabulary(self):
        names, T = gt.build_tag_matrix(self.model, TAGS[:-1] + ["higher education"])
        expected = (np.asarray(self.model["higher"], dtype=np.float64)
                    + np.asarray(self.model["education"], dtype=np.float64)) / 2
        np.testing.assert_allclose(T[-1], expected)

    def test_real_pipeline_agreement(self):
        text = ("Manipal students and faculty collaborate on innovative research "
                "in engineering, medicine and technology. zzqxjv")
        res = gt.run_pipeline(text, self.model, TAGS, out=lambda *a, **k: None)
        self.assertTrue(res["allclose"])
        self.assertLess(res["max_diff"], 1e-9)
        self.assertEqual(res["S_numpy"].shape, (len(res["in_vocab_tokens"]), 20))
        self.assertIn("zzqxjv", res["oov_tokens"])

    def test_real_oov_tag_raises(self):
        with self.assertRaises(ValueError):
            gt.build_tag_matrix(self.model, TAGS[:-1] + ["zzqxjv"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
