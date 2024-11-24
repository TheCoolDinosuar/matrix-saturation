from itertools import combinations, product
import numpy as np
from numpy.typing import NDArray

Pattern = NDArray[np.int_]

MatrixIndex = tuple[int, int]
PatternCopy = tuple[MatrixIndex, ...]

def solve_sat(m: int, n: int, P: Pattern) -> int:
    """ Solves for sat(m, n, P) using a binary linear program. """
    # Initializes all data like A_(i,j) and copies of P
    weight = sum(P)
    A = dict[MatrixIndex, list[PatternCopy]]()
    P_copies = list[PatternCopy]()
    for mat_idx in product(range(m), range(n)):
        A[mat_idx] = []
    for rows, cols in product(combinations(range(m), P.shape[0]),
                              combinations(range(n), P.shape[1])):
        P_copy = tuple(
            (rows[x], cols[y]) \
            for x, y in product(range(len(rows)), range(len(cols))) \
            if P[x][y]
        )
        P_copies.append(P_copy)
        for mat_idx in P_copy:
            A[mat_idx].append(P_copy)
        
    # Set up the program and constraints
    program = MixedIntegerLinearProgram()
    x = program.new_variable(binary=True)
    y = program.new_variable(binary=True)

    for P_copy in P_copies:
        program.add_constraint(
            y[P_copy] <= 1
        )
        program.add_constraint(
            y[P_copy] >= 0
        )
        
    for mat_idx in product(range(m), range(n)):
        program.add_constraint(
            x[mat_idx] <= 1
        )
        program.add_constraint(
            x[mat_idx] >= 0
        )


    for P_copy in P_copies:
        program.add_constraint(
            program.sum(x[mat_idx] for mat_idx in P_copy) <= weight - 1
        ) # Doesn't contain P
        program.add_constraint(
            (weight - 1) * y[P_copy] - sum(x[mat_idx] for mat_idx in P_copy) <= 0
        )
        program.add_constraint(
            y[P_copy] - sum(x[mat_idx] for mat_idx in P_copy) >= 2 - weight
        ) # Conditions on x and y
    for mat_idx in product(range(m), range(n)):
        if A[mat_idx]:
            program.add_constraint(
                program.sum(y[P_copy] for P_copy in A[mat_idx]) >= 1
            ) # At least one new copy of P if mat_idx changed to 1
        else: # If mat_idx not in any copy of P, x[mat_idx] is set to 1
            program.add_constraint(
                x[mat_idx] <= 1
            )
            program.add_constraint(
                x[mat_idx] >= 1
            )

    program.set_objective(
        -1 * sum(x[mat_idx] for mat_idx in product(range(m), range(n)))
    )

    return -1 * program.solve()

if __name__ == "__main__":
    P = np.array(
        [[1, 0, 0],
         [0, 1, 0],
         [0, 0, 1]],
        dtype=int
    )
    print(solve_sat(4, 4, P))
