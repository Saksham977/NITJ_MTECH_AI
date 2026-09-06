import math

import numpy as np

from Eigen_Values import get_user_matrix, eigenvalues_scratch, format_eigenvalue


def null_space_vector(matrix, lam, tol=1e-4):
    
    lam = complex(lam)
    if abs(lam.imag) < tol:
        lam = complex(lam.real, 0.0)

    n = len(matrix)
    M = [[complex(matrix[i][j]) - (lam if i == j else 0) for j in range(n)] for i in range(n)]

    pivot_cols = []
    row = 0
    for col in range(n):
        pivot = None
        best = tol
        for r in range(row, n):
            if abs(M[r][col]) > best:
                best = abs(M[r][col])
                pivot = r
        if pivot is None:
            continue

        M[row], M[pivot] = M[pivot], M[row]
        piv_val = M[row][col]
        M[row] = [x / piv_val for x in M[row]]

        for r in range(n):
            if r != row:
                factor = M[r][col]
                M[r] = [M[r][k] - factor * M[row][k] for k in range(n)]

        pivot_cols.append(col)
        row += 1
        if row == n:
            break

    free_cols = [c for c in range(n) if c not in pivot_cols]
    if not free_cols:
        return None

    free_col = free_cols[0]
    v = [0j] * n
    v[free_col] = 1.0 + 0j
    for i, pc in enumerate(pivot_cols):
        v[pc] = -M[i][free_col]

    norm = math.sqrt(sum(abs(x) ** 2 for x in v))
    if norm > tol:
        v = [x / norm for x in v]
    return v


def eigenvectors_numpy(matrix):
    arr = np.array(matrix, dtype=float)
    return np.linalg.eig(arr)


def format_vector(v, tol=1e-9):
    parts = []
    for x in v:
        x = complex(x)
        if abs(x.imag) < tol:
            parts.append(f"{x.real:.4f}")
        else:
            sign = "+" if x.imag >= 0 else "-"
            parts.append(f"{x.real:.4f}{sign}{abs(x.imag):.4f}i")
    return "[" + ", ".join(parts) + "]"


def main():
    matrix = get_user_matrix()

    print("\n--- Eigenvalues & eigenvectors (from scratch without python libraries) ---")
    for lam in eigenvalues_scratch(matrix):
        print(f"Eigenvalue: {format_eigenvalue(lam)}")
        v = null_space_vector(matrix, lam)
        if v is None:
            print("  Could not isolate a distinct eigenvector numerically "
                  "(possible repeated/defective eigenvalue).")
        else:
            print(f"  Eigenvector: {format_vector(v)}")

    print("\n--- Eigenvalues & eigenvectors (using numpy library) ---")
    values, vectors = eigenvectors_numpy(matrix)
    for idx, lam in enumerate(values):
        print(f"Eigenvalue: {format_eigenvalue(lam)}")
        print(f"  Eigenvector: {format_vector(vectors[:, idx])}")


if __name__ == "__main__":
    main()
