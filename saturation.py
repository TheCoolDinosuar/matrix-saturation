from copy import deepcopy
from itertools import combinations, product
import random
import numpy as np
from numpy.typing import NDArray

Pattern = NDArray[np.int_]


class Matrix:
    """ Only supports 2d matrices right now. """

    def __init__(self, matrix: Pattern) -> None:
        self.matrix: NDArray[np.int_] = matrix
        self.weight: int = np.concatenate(matrix).sum()
        self._rng = np.random.default_rng()

    def __str__(self):
        return str(self.matrix).replace('0', ' ').replace('1', '*')

    @staticmethod
    def from_str_list(entries: list[str]) -> Pattern:
        """ Function to ease input of large 0-1 matrices into code. """
        return np.array([
            [int(c) for c in row]
            for row in entries
        ])

    @staticmethod
    def kron(X: "Matrix", Y: "Matrix") -> "Matrix":
        """ Kronecker product of X and Y. """
        return Matrix(np.kron(X.matrix, Y.matrix))

    def contains_pattern(self, pattern: Pattern) -> bool:
        """ Checks if matrix contains pattern as a submatrix. """
        for m_dim, p_dim in zip(self.matrix.shape, pattern.shape):
            if m_dim < p_dim:
                return False
        for rows, cols in product(
                combinations(range(self.matrix.shape[0]), pattern.shape[0]),
                combinations(range(self.matrix.shape[1]), pattern.shape[1])):
            good = True
            for x, y in product(range(len(rows)), range(len(cols))):
                if self.matrix[rows[x]][cols[y]] < pattern[x][y]:
                    good = False
                    break
            if good:
                return True
        return False

    def is_saturating(self, pattern: Pattern) -> tuple[bool, list[tuple[int, int]]]:
        """ Checks if matrix is saturating for pattern. """
        if self.contains_pattern(pattern):
            return False, list()
        allowed_spots = list()
        saturating = True
        for a, b in product(range(self.matrix.shape[0]), range(self.matrix.shape[1])):
            if self.matrix[a][b] != 1:
                self.matrix[a][b] = 1
                if not self.contains_pattern(pattern):
                    allowed_spots.append((a, b))
                    saturating = False
                self.matrix[a][b] = 0
        return saturating, allowed_spots

    def is_vertical_witness(self, pattern: Pattern, row_idx: int) -> bool:
        """ Checks if matrix is a vertical witness for pattern.
            row_idx: is index of empty row claimed to be expandable. """
        if self.contains_pattern(pattern):
            return False
        for col_idx in range(self.matrix.shape[1]):
            self.add_one((row_idx, col_idx))
            if not self.contains_pattern(pattern):
                return False
            self.remove_one((row_idx, col_idx))
        return True

    def is_horizontal_witness(self, pattern: Pattern, col_idx: int) -> bool:
        """ Checks if matrix is a horizontal witness for pattern.
            col_idx: index of empty col claimed to be expandable. """
        if self.contains_pattern(pattern):
            return False
        for row_idx in range(self.matrix.shape[0]):
            self.add_one((row_idx, col_idx))
            if not self.contains_pattern(pattern):
                return False
            self.remove_one((row_idx, col_idx))
        return True

    def is_witness(self, pattern: Pattern, row_idx: int, col_idx: int) -> bool:
        """ Checks if matrix is a witness for pattern. """
        return self.is_vertical_witness(pattern, row_idx) \
            and self.is_horizontal_witness(pattern, col_idx)

    def add_one(self, loc: tuple[int, int] | None = None) -> bool:
        """ Randomly chooses an empty spot in the matrix to change from 0 to 1,
            or a given element with indices loc. If successful, returns True,
            else False. """
        if self.weight == self.matrix.size:
            return False
        if loc is None:
            while True:
                row = self._rng.integers(0, self.matrix.shape[0])
                col = self._rng.integers(0, self.matrix.shape[1])
                if self.matrix[row][col] == 0:
                    self.matrix[row][col] = 1
                    self.weight += 1
                    return True
        if self.matrix[loc] == 1:
            return False
        self.matrix[loc] = 1
        self.weight += 1
        return True

    def add_row(self, idx: int) -> None:
        """ Adds an empty row at index idx. """
        self.matrix = np.concatenate((
            self.matrix[:idx],
            np.zeros((1, self.matrix.shape[1]), dtype=int),
            self.matrix[idx:])
        )

    def remove_one(self, loc: tuple[int, int]) -> bool:
        """ Removes a 1 entry at the element at position loc. """
        if self.matrix[loc] == 0:
            return False
        self.matrix[loc] = 0
        self.weight -= 1
        return True


def ssat_bounded(P: Pattern) -> bool:
    """ Returns True if ssat(P) is O(1).
        cond_1: first row of P has 1 entry that's the only 1 in its column.
        cond_2: same as cond_1 but last row.
        cond_3: same as cond_1 but columns.
        cond_4: same as cond_2 but columns.
        cond_5: exists a 1 entry that's the only entry in both row and column. """
    cond_1, cond_2, cond_3, cond_4, cond_5 = (False for _ in range(5))
    row_sums = P.sum(axis=1)
    col_sums = P.sum(axis=0)
    for col in range(P.shape[1]):
        if P[0][col] == 1 == col_sums[col]:
            cond_1 = True
            break
    for col in range(P.shape[1]):
        if P[-1][col] == 1 == col_sums[col]:
            cond_2 = True
            break
    for row in range(P.shape[0]):
        if P[row][0] == 1 == row_sums[row]:
            cond_3 = True
            break
    for row in range(P.shape[0]):
        if P[row][-1] == 1 == row_sums[row]:
            cond_4 = True
            break
    for i, j in product(range(P.shape[0]), range(P.shape[1])):
        if P[i][j] == 1 == row_sums[i] == col_sums[j]:
            cond_5 = True
            break
    return cond_1 and cond_2 and cond_3 and cond_4 and cond_5


def is_decomposable(P: Pattern) -> bool:
    """ Checks if a pattern P is decomposable. That is, if it is of the form
        [ A 0 ] or [ 0 A ]
        [ 0 B ]    [ B 0 ] 
        for some matrices A and B. """
    for i, j in product(range(1, P.shape[0]), range(1, P.shape[1])):
        if P[i:, :j].sum() == 0 and P[:i, j:].sum() == 0 \
                and P[i:, j:].sum() != 0 and P[:i, :j].sum() != 0:
            return True
        if P[:i, :j].sum() == 0 and P[i:, j:].sum() == 0 \
                and P[i:, :j].sum() != 0 and P[:i, j:].sum() != 0:
            return True
    return False


def is_q1_like(P: Pattern) -> bool:
    """ Checks if a pattern is Q_1 like. That is, if the outer elements of P
        form the shape
        [   *  ] or [  *   ]
        [ *    ]    [    * ]
        [    * ]    [ *    ]
        [  *   ]    [   *  ]. """
    if P[0].sum() == 1 and P[-1].sum() == 1 \
        and P.sum(axis=0)[0] == 1 and P.sum(axis=0)[-1].sum() == 1 \
        and P[0][0] == 0 and P[0][-1] == 0 \
            and P[-1][0] == 0 and P[-1][-1] == 0:
        t, b, l, r = 0, 0, 0, 0
        while P[0][t] != 1:
            t += 1
        while P[-1][b] != 1:
            b += 1
        while P[l][0] != 1:
            l += 1
        while P[r][-1] != 1:
            r += 1
        if t > b:
            return l < r
        if t < b:
            return l > r
    return False


def has_empty_row_or_col(P: Pattern) -> bool:
    """ Checks if P has an empty row or column. """
    return 0 in P.sum(axis=0) or 0 in P.sum(axis=1)


def estimate_sat(P: Pattern,
                 a: int,
                 b: int | None = None,
                 num_trials: int = 1,
                 starting_matrix: Matrix | None = None) -> int:
    """ Tries to fill a matrix with 1s until it becomes saturating for P to
        estimate the sat function for P. Extremely slow and probably not worth
        using. """
    if b is None:
        b = a
    if starting_matrix is None:
        starting_matrix = Matrix(np.zeros((a, b), dtype=int))
    best_seen = a * b  # Just an upper bound, could be math.inf
    pre_computed = starting_matrix.is_saturating(P)
    for _ in range(num_trials):
        M = starting_matrix
        saturating, allowed = deepcopy(pre_computed)
        while not saturating:
            random_spot = random.sample(allowed, 1)[0]
            allowed.remove(random_spot)
            M.matrix[random_spot] = 1
            M.weight += 1
            allowed_spots = list()
            saturating = True
            for x, y in allowed:
                M.matrix[x][y] = 1
                if not M.contains_pattern(P):
                    allowed_spots.append((x, y))
                    saturating = False
                M.matrix[x][y] = 0
            allowed = allowed_spots
        best_seen = min(best_seen, M.weight)
    return best_seen
