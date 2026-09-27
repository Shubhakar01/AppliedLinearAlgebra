import sys
import math
import random
from typing import Self


"""
A custom vector class implementation for educational purposes.
"""

class Vec:
    def __init__(self, src=None) -> Self:
        """
        Construct a Vec from an iterable of numbers (int or float).
        If src is None, constructs an empty vector.
        Raises TypeError if any element is not an int or float.
        """
        if src is None:
            self.elements = []
        else:
            elements = list(src)
            for x in elements:
                if not isinstance(x, (int, float)):
                    raise TypeError(f"Scalar must be a number: {type(x)}")
            self.elements = elements

    def __add__(self, t: Self) -> Self:
        """
        Elementwise vector addition: self + t.
        Both operands must be Vec instances of the same dimension.
        Returns a new Vec; does not mutate self or t.
        """
        if not isinstance(t, Vec):
            raise TypeError(f"Expected Vec: {type(t)}")
        if len(self.elements) != len(t):
            raise TypeError(f"Type error - vectors must be of same dimensions")

        return Vec([round(x + y, 5) for x, y in zip(self.elements, t.elements)])

    def __rmul__(self, scalar: int | float) -> Self:
        """
        Scalar multiplication: scalar * self (e.g. 2.2 * v).
        Returns a new Vec with each element scaled; does not mutate self.
        """
        if not isinstance(scalar, (int, float)):
            raise TypeError(f"Vector multiplication with invalid type: {type(scalar)}")
        #
        return Vec([round(x * scalar, 5) for x in self.elements])

    def __imul__(self, scalar: int | float) -> Self:
        """
        In-place scalar multiplication: self *= scalar.
        Mutates self and returns it.
        """
        if not isinstance(scalar, (int, float)):
            raise TypeError(f"Vector multiplication with invalid type: {type(scalar)}")

        for i, val in enumerate(self.elements):
            self.elements[i] = round(val * scalar, 5)
        #
        return self

    def __repr__(self) -> str:
        """Return the vector's elements as a printable string."""
        return repr(self.elements)

    def __len__(self) -> int:
        """Return the dimension (number of elements) of the vector."""
        return len(self.elements)

    def __sub__(self, t: Self) -> Self:
        """
        Elementwise vector subtraction: self - t.
        Both operands must be Vec instances of the same dimension.
        Returns a new Vec; does not mutate self or t.
        """
        if not isinstance(t, Vec):
            raise TypeError(f"Expected Vec: {type(t)}")
        if len(self.elements) != len(t):
            raise TypeError(f"Type error - vectors must be of same dimensions")

        return Vec([round(x - y, 5) for x, y in zip(self.elements, t.elements)])

    def __neg__(self) -> Self:
        """
        Unary negation: -self.
        Returns a new Vec with every element negated; does not mutate self.
        """
        return Vec([round(-x, 5) for x in self.elements])

    def __radd__(self, other):
        """
        Reflected addition: other + self, used when `other` does not know
        how to add a Vec (e.g. int.__add__ returns NotImplemented first).

        Special-cased for `other == 0` so that the builtin sum() works on
        an iterable of Vecs (sum() starts its accumulator at 0 by default,
        so it calls 0 + vecs[0], which routes here). Any other scalar is
        rejected, to stay consistent with __add__, which only supports
        Vec + Vec (not vector + scalar broadcasting).
        """
        if isinstance(other, (int, float)) and other == 0:
            return Vec(self.elements)
        raise TypeError(f"Expected Vec: {type(other)}")

    def __iadd__(self, other):
        """
        In-place vector addition: self += other.
        `other` must be a Vec instance of the same dimension.
        Mutates self and returns it.
        """
        if not isinstance(other, Vec):
            raise TypeError(f"Expected Vec: {type(other)}")
        if len(self.elements) != len(other):
            raise TypeError(f"Type error - vectors must be of same dimensions")

        for i, (x, y) in enumerate(zip(self.elements, other.elements)):
            self.elements[i] = round(x + y, 5)
        return self

    # return a vector of @n zeroes. precondition: @n > 0
    @staticmethod
    def zeros(n: int) -> Self:
        """Return a new Vec of length n, every element 0. Precondition: n > 0."""
        if n <= 0:
            raise ValueError(f"n must be > 0: {n}")
        return Vec([0] * n)

    # return a vector of @n. precondition: @n > 0
    @staticmethod
    def ones(n: int) -> Self:
        """Return a new Vec of length n, every element 1. Precondition: n > 0."""
        if n <= 0:
            raise ValueError(f"n must be > 0: {n}")
        return Vec([1] * n)

    # return a vector of @n uniformly distributed numbers in [0, 1]. precondition: @n > 0
    @staticmethod
    def uniform(n: int) -> Self:
        """Return a new Vec of length n with elements drawn uniformly from [0, 1]. Precondition: n > 0."""
        if n <= 0:
            raise ValueError(f"n must be > 0: {n}")
        return Vec([random.uniform(0, 1) for _ in range(n)])

    # Calculates the Euclidean norm (L2 norm) of the vector.
    # sqrt(e[0]^2 + e[1]^2 + e[2]^2 + ... + e[n-1]^2)
    def norm(self) -> float:
        """Return the Euclidean (L2) norm of the vector."""
        return round(math.sqrt(sum(x * x for x in self.elements)), 5)

    def mean(self) -> float:
        """
        Return the arithmetic mean of the vector's entries:
        sum(entries) / len(entries).
        Raises ZeroDivisionError on an empty vector.
        """
        if len(self.elements) == 0:
            raise ZeroDivisionError("Cannot compute mean of an empty vector")
        return round(sum(self.elements) / len(self.elements), 5)

    def demean(self) -> Self:
        """
        Return a new Vec obtained by subtracting self's mean from every entry.
        Does not mutate self.
        """
        mu = self.mean()
        return Vec([round(x - mu, 5) for x in self.elements])

    def std(self) -> float:
        """
        Return the (population) standard deviation of the vector's entries:
        sqrt(average of the squared deviations from the mean).
        Built on top of demean().
        """
        if len(self.elements) == 0:
            raise ZeroDivisionError("Cannot compute std of an empty vector")
        demeaned = self.demean()
        variance = sum(x * x for x in demeaned.elements) / len(demeaned)
        return round(math.sqrt(variance), 5)


"""
(1) Understand the basic design of the vector abstraction. Review the implementation.
(2) Document each function.
(3) Implement all unimplemented methods.
(4) Create appropriate tests for this implementation, increasing the confidence about its correctness.
(5) Test this implementation by importing the class in a sepatate python script.

(6) Measure the performance of each of these functions on vectors of varying lengths.
    Try 2k to 64k dimension vectors and time the results.
    How would you do the measurements?
(7) Measure the performance on your machine. Check it on colab.

(8) use numpy and compare the performance.
"""


if sys.version_info < (3, 8):
    sys.exit("Error: This script requires Python 3.8 or higher.")

if __name__ == "__main__":
    #z1 = Vec.zeros(10)
    v1 = Vec([0, 1, 1.03])
    print(v1)
    v3 = 2.2 * v1
    v3 *= 5
    # v3 = 1 + v3  # NOTE: __radd__ only special-cases 0 (to support sum()); a
    #                genuine scalar + Vec broadcast is intentionally unsupported,
    #                for the same reason __add__ only accepts Vec + Vec.
    print(v3)
    v2 = v1 + v3
    print(v1 + v3)
    print(-(v1 + v3))

    # sum() relies on __radd__ under the hood: it starts from 0 and does
    # 0 + vecs[0], then result + vecs[1], etc.
    total = sum([v1, v3])
    print("sum([v1, v3]):", total)

    # quick smoke test of the new mean/demean/std methods
    sample = Vec([2, 4, 4, 4, 5, 5, 7, 9])
    print("mean:", sample.mean())
    print("demean:", sample.demean())
    print("std:", sample.std())
