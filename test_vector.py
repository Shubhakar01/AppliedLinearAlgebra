"""
Applied Linear Algebra Assignment 1
Unit tests for Vec.mean, Vec.demean and Vec.std .
S Shubhakar
261100660015
"""

import math
import unittest

from vec import Vec


class TestMean(unittest.TestCase):

    def test_known_value(self):
        v = Vec([2, 4, 4, 4, 5, 5, 7, 9])
        self.assertAlmostEqual(v.mean(), 5.0)

    def test_constant_vector(self):
        v = Vec([3, 3, 3, 3])
        self.assertAlmostEqual(v.mean(), 3.0)

    def test_single_element(self):
        v = Vec([42])
        self.assertAlmostEqual(v.mean(), 42.0)

    def test_mean_is_linear_under_scaling(self):
        # mean(k * v) == k * mean(v)
        v = Vec([1, 2, 3, 4, 5])
        k = 3.5
        scaled = k * v
        self.assertAlmostEqual(scaled.mean(), k * v.mean(), places=4)

    def test_mean_is_additive(self):
        # mean(v + w) == mean(v) + mean(w) for same-length vectors
        v = Vec([1, 2, 3])
        w = Vec([4, 5, 6])
        self.assertAlmostEqual((v + w).mean(), v.mean() + w.mean(), places=4)

    def test_empty_vector_raises(self):
        v = Vec([])
        with self.assertRaises(ZeroDivisionError):
            v.mean()


class TestDemean(unittest.TestCase):

    def test_demeaned_values(self):
        v = Vec([2, 4, 4, 4, 5, 5, 7, 9])
        expected = [-3, -1, -1, -1, 0, 0, 2, 4]
        for got, want in zip(v.demean().elements, expected):
            self.assertAlmostEqual(got, want)

    def test_demeaned_vector_has_zero_mean(self):
        # mean(demean(v)) == 0 for any v
        for data in ([1, 2, 3, 4, 5], [10, -3, 7.5, 2.25], [0, 0, 0, 1]):
            v = Vec(data)
            self.assertAlmostEqual(v.demean().mean(), 0.0, places=4)

    def test_sum_of_deviations_is_zero(self):
        v = Vec([2, 4, 4, 4, 5, 5, 7, 9])
        self.assertAlmostEqual(sum(v.demean().elements), 0.0, places=4)

    def test_constant_vector_demeans_to_zero_vector(self):
        v = Vec([7, 7, 7, 7])
        for x in v.demean().elements:
            self.assertAlmostEqual(x, 0.0)

    def test_demean_is_idempotent(self):
        v = Vec([1, 2, 3, 4, 5])
        once = v.demean()
        twice = once.demean()
        for a, b in zip(once.elements, twice.elements):
            self.assertAlmostEqual(a, b, places=4)

    def test_demean_does_not_mutate_original(self):
        v = Vec([1, 2, 3])
        original = list(v.elements)
        v.demean()
        self.assertEqual(v.elements, original)


class TestStd(unittest.TestCase):

    def test_known_value(self):
        v = Vec([2, 4, 4, 4, 5, 5, 7, 9])
        self.assertAlmostEqual(v.std(), 2.0, places=4)

    def test_constant_vector_has_zero_std(self):
        # A constant vector has no spread, so its std is 0
        v = Vec([5, 5, 5, 5, 5])
        self.assertAlmostEqual(v.std(), 0.0)

    def test_std_is_non_negative(self):
        # std is a square root of an average of squares, so it can never be negative
        for data in ([1, 2, 3], [-5, -5, 10], [0.1, 0.2, 0.3, 0.4]):
            v = Vec(data)
            self.assertGreaterEqual(v.std(), 0.0)

    def test_std_is_translation_invariant(self):
        v = Vec([1, 2, 3, 4, 5])
        c = 100
        shifted = Vec([x + c for x in v.elements])
        self.assertAlmostEqual(shifted.std(), v.std(), places=4)

    def test_std_scales_with_absolute_value_of_k(self):
        # Scaling every entry by k scales the std by |k|: std(k*v) == |k| * std(v)
        v = Vec([1, 2, 3, 4, 5])
        for k in (3, -3, 0.5):
            scaled = k * v
            self.assertAlmostEqual(scaled.std(), abs(k) * v.std(), places=3)

    def test_std_matches_manual_formula(self):
        data = [10, 12, 23, 23, 16, 23, 21, 16]
        v = Vec(data)
        mu = sum(data) / len(data)
        expected = math.sqrt(sum((x - mu) ** 2 for x in data) / len(data))
        self.assertAlmostEqual(v.std(), expected, places=4)

    def test_empty_vector_raises(self):
        v = Vec([])
        with self.assertRaises(ZeroDivisionError):
            v.std()


if __name__ == "__main__":
    unittest.main(verbosity=2)
