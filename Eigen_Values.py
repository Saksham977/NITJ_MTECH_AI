import numpy as np

def get_user_matrix():
    """Gets matrix dimensions and values from the user row by row."""
    n = int(input("Enter the size of the matrix (n for n*n matrix): "))
    print(f"Enter {n} numbers for each row separated by spaces:")
    
    matrix = []
    for i in range(n):
        row_input = input(f"Row {i + 1}: ").split()
        # Convert each string entry into a float number
        row_numbers = []
        for val in row_input:
            row_numbers.append(float(val))
        matrix.append(row_numbers)
    return matrix


def multiply_matrices(A, B):
    """Multiplies two matrices using simple, explicit loops."""
    n = len(A)
    # Create a blank result matrix filled with zeros
    result = []
    for i in range(n):
        result.append([0.0] * n)
        
    # Standard row-by-column multiplication loop
    for i in range(n):
        for j in range(n):
            total = 0.0
            for k in range(n):
                total += A[i][k] * B[k][j]
            result[i][j] = total
    return result


def get_matrix_trace(A):
    """Calculates the sum of the main diagonal numbers (top-left to bottom-right)."""
    total = 0.0
    for i in range(len(A)):
        total += A[i][i]
    return total


def find_polynomial_coefficients(matrix):
    """
    Faddeev-LeVerrier algorithm:
    Converts a matrix into polynomial coefficients using matrix multiplication and trace.
    """
    n = len(matrix)
    # Create an identity matrix (1s on the diagonal, 0s elsewhere)
    I = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(1.0 if i == j else 0.0)
        I.append(row)

    # Initialize intermediate calculation matrix M with zeros
    M = []
    for i in range(n):
        M.append([0.0] * n)

    coefficients = [1.0] # The leading x^n coefficient is always 1
    c = 1.0

    for k in range(1, n + 1):
        # 1. Multiply original matrix with our temporary tracking matrix M
        AM = multiply_matrices(matrix, M)
        
        # 2. Add the previous coefficient 'c' to the diagonal of AM
        for i in range(n):
            AM[i][i] += c
            
        # 3. M becomes this updated matrix for the next iteration step
        M = AM
        
        # 4. Compute the next coefficient using the trace of (matrix * M)
        next_AM = multiply_matrices(matrix, M)
        c = -get_matrix_trace(next_AM) / k
        coefficients.append(c)
        
    return coefficients


def evaluate_polynomial(coefficients, x):
    """Evaluates a polynomial at value x using Horner's efficient method."""
    result = 0j
    for c in coefficients:
        result = result * x + c
    return result


def differentiate_polynomial(coefficients):
    """Returns the coefficients of the derivative of the given polynomial."""
    n = len(coefficients) - 1
    derivative = []
    for i in range(n):
        power = n - i
        derivative.append(coefficients[i] * power)
    return derivative


def deflate_polynomial(coefficients, root):
    """Synthetic division: divides the polynomial by (x - root), dropping the remainder."""
    quotient = [coefficients[0]]
    for c in coefficients[1:]:
        quotient.append(quotient[-1] * root + c)
    return quotient[:-1]


def newton_raphson_root(coefficients, initial_guess, tol=1e-12, max_iter=200):
    """
    Newton-Raphson method:
    Refines a single complex guess towards a root using quadratic convergence,
    x_{k+1} = x_k - f(x_k) / f'(x_k).
    """
    derivative = differentiate_polynomial(coefficients)
    x = initial_guess
    for _ in range(max_iter):
        fx = evaluate_polynomial(coefficients, x)
        fpx = evaluate_polynomial(derivative, x)
        if fpx == 0:
            x += 1e-6 + 1e-6j
            continue
        delta = fx / fpx
        x = x - delta
        if abs(delta) < tol:
            break
    return x


def find_roots_newton_raphson(coefficients):
    """
    Finds all roots of a polynomial one at a time using Newton-Raphson,
    deflating the polynomial after each root is found.
    """
    n = len(coefficients) - 1
    if n == 0:
        return []

    working_coeffs = coefficients[:]
    roots = []
    guess = 0.4 + 0.9j
    for _ in range(n):
        root = newton_raphson_root(working_coeffs, guess)
        # Polish the root against the original polynomial for better accuracy
        root = newton_raphson_root(coefficients, root)
        roots.append(root)
        working_coeffs = deflate_polynomial(working_coeffs, root)
        guess *= (0.4 + 0.9j)

    return roots


def eigenvalues_scratch(matrix):
    """Computes eigenvalues of a matrix from scratch (Faddeev-LeVerrier + Newton-Raphson)."""
    coeffs = find_polynomial_coefficients(matrix)
    return find_roots_newton_raphson(coeffs)


def format_eigenvalue(value):
    """Alias for format_output, kept for callers that name it by eigenvalue."""
    return format_output(value)


def format_output(value):
    """Cleans up the complex number strings for user display."""
    v = complex(value)
    # If the imaginary part is extremely close to zero, treat it as a real number
    if abs(v.imag) < 1e-4:
        return f"{v.real:.4f}"
    
    sign = "+" if v.imag >= 0 else "-"
    return f"{v.real:.4f} {sign} {abs(v.imag):.4f}i"


def main():
    # 1. Get input matrix
    matrix = get_user_matrix()

    # 2. Method 1: Scratch implementation
    print("\n--- Eigenvalues from Scratch ---")
    coeffs = find_polynomial_coefficients(matrix)
    scratch_eigenvalues = find_roots_newton_raphson(coeffs)
    for val in scratch_eigenvalues:
        print(format_output(val))

    # 3. Method 2: NumPy library shortcut
    print("\n--- Eigenvalues using NumPy ---")
    numpy_matrix = np.array(matrix, dtype=float)
    numpy_eigenvalues = np.linalg.eigvals(numpy_matrix)
    for val in numpy_eigenvalues:
        print(format_output(val))


if __name__ == "__main__":
    main()
