import numpy as np

def get_user_matrix():
    n = int(input("Enter the size of the matrix (n for an n*n matrix): "))
    if n <= 0:
        raise ValueError("Matrix size must be a positive integer.")

    print(f"Enter {n} numbers for each row separated by spaces:")

    matrix = []
    for i in range(n):
        row_input = input(f"Row {i + 1}: ").split()
        if len(row_input) != n:
            raise ValueError(f"Each row must contain exactly {n} numbers.")

        row_numbers = []
        for val in row_input:
            row_numbers.append(float(val))
        matrix.append(row_numbers)
    return matrix


def print_matrix(matrix):
    """Prints a matrix with cleanly formatted decimal places."""
    for row in matrix:
        formatted_row = []
        for val in row:
            formatted_row.append(f"{val:.4f}")
        print(formatted_row)


def get_determinant(matrix):
    """Calculates the determinant using simple recursion and matrix minors."""
    n = len(matrix)

    if n == 0:
        raise ValueError("Matrix must not be empty.")
    if any(len(row) != n for row in matrix):
        raise ValueError("Matrix must be square.")
    
    # Base cases for small matrices
    if n == 1:
        return matrix[0][0]
    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    det = 0.0
    for col in range(n):
        # Create a smaller sub-matrix (minor) by skipping row 0 and current column
        minor = []
        for r in range(1, n):
            sub_row = []
            for c in range(n):
                if c != col:
                    sub_row.append(matrix[r][c])
            minor.append(sub_row)
            
        # Alternating sign pattern (+ - + - ...)
        sign = (-1) ** col
        det += sign * matrix[0][col] * get_determinant(minor)
        
    return det


def calculate_inverse_gauss(matrix):
    """
    Computes matrix inverse using Gauss-Jordan Elimination with partial pivoting.
    Spreads out single-line logic into straightforward, visible math loops.
    """
    n = len(matrix)
    if n == 0 or any(len(row) != n for row in matrix):
        raise ValueError("Matrix must be a non-empty square matrix.")
    
    # 1. Create the Augmented Matrix [Matrix | Identity Matrix]
    augmented = []
    for i in range(n):
        # Copy original row numbers
        row = list(matrix[i])
        # Add 1s and 0s for the Identity matrix on the right half
        for j in range(n):
            if i == j:
                row.append(1.0)
            else:
                row.append(0.0)
        augmented.append(row)

    # 2. Main Gauss-Jordan Elimination loop
    for col in range(n):
        
        # --- Partial Pivoting ---
        # Find the row with the largest absolute value in the current column
        pivot_row = col
        max_val = abs(augmented[col][col])
        for r in range(col + 1, n):
            if abs(augmented[r][col]) > max_val:
                max_val = abs(augmented[r][col])
                pivot_row = r

        # A zero pivot after row swapping means the matrix is singular.
        if max_val == 0.0:
            return None

        # Swap current row with the pivot row to put the largest number on top
        temp = augmented[col]
        augmented[col] = augmented[pivot_row]
        augmented[pivot_row] = temp

        # --- Row Normalization ---
        # Divide the entire pivot row by its main diagonal element to make it equal 1.0
        pivot_element = augmented[col][col]
        for c in range(2 * n):
            augmented[col][c] = augmented[col][c] / pivot_element

        # --- Row Elimination ---
        # Subtract the pivot row from all other rows to make their values in this column 0
        for r in range(n):
            if r != col:
                factor = augmented[r][col]
                for c in range(2 * n):
                    augmented[r][c] = augmented[r][c] - (factor * augmented[col][c])

    # 3. Extract the right half of the augmented matrix (which is now the inverse)
    inverse = []
    for i in range(n):
        inverse_row = augmented[i][n:] # Take elements from column index n to the end
        inverse.append(inverse_row)
        
    return inverse


def main():
    # 1. Input gathering
    matrix = get_user_matrix()

    # 2. Display the determinant for reference. Invertibility is decided by
    #    the pivot checks below so small, valid determinants are not rejected.
    print("\n--- Determinant ---")
    det = get_determinant(matrix)
    print(f"Determinant: {det:.6f}")

    # 3. Method 1: Scratch implementation
    print("\n--- Inverse (from scratch, Gauss Elimination) ---")
    inv_scratch = calculate_inverse_gauss(matrix)
    if inv_scratch is None:
        print("Matrix is singular - inverse does not exist.")
        return
    else:
        print_matrix(inv_scratch)

    # 4. Method 2: NumPy library shortcut
    print("\n--- Inverse (using NumPy library) ---")
    numpy_arr = np.array(matrix, dtype=float)
    inv_numpy = np.linalg.inv(numpy_arr)
    print_matrix(inv_numpy.tolist())


if __name__ == "__main__":
    main()
