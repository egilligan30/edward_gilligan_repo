"""
Matrix Operations Toolkit
------------------------
Purpose:
    A comprehensive toolkit for symbolic matrix operations using SymPy, including:
    - Determinants via recursive cofactor expansion
    - Manual RREF computation
    - Matrix inversion using augmented matrices
    - Eigenvalue and eigenvector computation
    - Diagonalization of square matrices

Outline:
    1. get_determinant: Compute determinant recursively
    2. get_invertible_matrix: Prompt user until an invertible matrix is provided
    3. rref: Manual computation of Reduced Row Echelon Form
    4. find_inverse: Compute matrix inverse using augmented matrix and RREF
    5. user_input_matrix: General matrix input helper
    6. manual_factor_polynomial: Factor characteristic polynomial and find roots
    7. compute_eigenvalues: Compute eigenvalues manually
    8. extract_nullspace_from_rref: Construct basis vectors for null space
    9. compute_eigenvectors: Compute eigenvectors for all eigenvalues
   10. diagonalize_matrix: Diagonalize matrix using eigenvectors and eigenvalues
   11. get_matrix_for_eigen: Prompt user for a square matrix for eigen computations
"""

import sympy as sp


# ---------------------- Determinant ----------------------
def get_determinant(M, depth=0):
    """
    Compute the determinant of a square matrix M recursively using cofactor expansion.
    Prints step-by-step details for learning purposes.

    Parameters:
        M (sp.Matrix): The matrix for which to compute the determinant.
        depth (int): Recursion depth for indentation in print statements.

    Returns:
        det (sp.Expr): Determinant of the matrix.
    """
    n = M.rows
    indent = "  " * depth

    print(f"\n{indent}Computing determinant of {n}×{n} matrix:")
    sp.pprint(M)
    print(f"{indent}" + "-" * 50)

    # 1x1 — nothing to expand, just return the single element
    if n == 1:
        val = M[0, 0]
        print(f"{indent}Base case (1×1): det = {val}")
        return val

    # 2x2 — use the classic ad - bc shortcut
    if n == 2:
        a, b = M[0, 0], M[0, 1]
        c, d = M[1, 0], M[1, 1]
        det = a*d - b*c
        print(f"{indent}2×2: det = {a}×{d} - {b}×{c} = {det}")
        return det

    # Anything bigger — expand along the first row, alternating signs
    det = 0
    term_strings = []

    for col in range(n):
        sign = (-1) ** col
        coeff = M[0, col]
        submatrix = M.minor_submatrix(0, col)

        print(f"\n{indent}→ Expanding on a[1,{col+1}] = {coeff} with sign {sign}")
        print(f"{indent}  Submatrix:")
        sp.pprint(submatrix)

        subdet = get_determinant(submatrix, depth + 1)
        term = sign * coeff * subdet
        det += term
        term_strings.append(f"{sign}×{coeff}×{subdet}")

        print(f"{indent}  Term = {term_strings[-1]} = {term}")
        print(f"{indent}  Running total = {det}")
        print(f"{indent}" + "-" * 50)

    # At the top level, print a summary and sanity-check against SymPy
    if depth == 0:
        print(f"\nSum of terms: {' + '.join(term_strings)} = {det}")
        print(f"\nFinal Determinant = {det}")

        builtin_det = M.det()
        print(f"sympy.det(): {builtin_det}")
        if builtin_det != det:
            print("Mismatch! Custom result doesn't match sympy.")
        else:
            print("Validation: Custom result matches sympy.")

    return det


# ---------------------- Matrix Input & Validation ----------------------
def user_input_matrix(prompt="Enter matrix"):
    """
    Prompts user to input a square matrix of size n x n.
    Converts input strings to symbolic expressions using SymPy.

    Returns:
        sp.Matrix: User-input square matrix
    """
    n = int(input(f"{prompt} - size n x n, n = "))
    matrix_data = []

    for i in range(n):
        while True:
            row_input = input(f"Row {i+1}: ")
            row_values = row_input.strip().split()
            if len(row_values) != n:
                print(f"Please enter exactly {n} numbers.")
            else:
                try:
                    # sympify handles fractions, sqrt, etc. — not just plain ints
                    row = [sp.sympify(val) for val in row_values]
                    matrix_data.append(row)
                    break
                except Exception:
                    print("Invalid input. Please enter numbers or valid expressions.")

    return sp.Matrix(matrix_data)


def get_invertible_matrix():
    """
    Repeatedly prompts user to input a square matrix until it is invertible.
    Computes determinant to ensure non-singularity.

    Returns:
        sp.Matrix: Invertible square matrix
    """
    # Keep asking until we get something we can actually invert
    while True:
        A = user_input_matrix("Input matrix for inversion")
        det = get_determinant(A)
        print(f"\nDeterminant: {det}")

        if det == 0:
            print("\nMatrix is not invertible (det = 0). Please input a different matrix.\n")
        else:
            print("\nMatrix is invertible (det ≠ 0). Proceeding to inversion...")
            return A


# ---------------------- Manual RREF ----------------------
def rref(A):
    """
    Computes the Reduced Row Echelon Form (RREF) of a matrix manually.
    Prints each elementary row operation step.

    Returns:
        sp.Matrix: Matrix in RREF
    """
    A = A.copy()
    rows, cols = A.shape
    pivot_row = 0
    step = 1

    print("\n=== INITIAL MATRIX ===")
    sp.pprint(A)
    print("=" * 60)

    # Forward pass — get zeros below each pivot
    for pivot_col in range(cols):
        if pivot_row >= rows:
            break

        # Find the first non-zero entry in this column to use as pivot
        max_row = None
        for r in range(pivot_row, rows):
            if A[r, pivot_col] != 0:
                max_row = r
                break
        if max_row is None:
            continue

        if max_row != pivot_row:
            print(f"\nStep {step}: Swap row {pivot_row+1} with row {max_row+1}")
            A.row_swap(pivot_row, max_row)
            sp.pprint(A)
            step += 1

        # Scale the pivot row so the leading entry becomes 1
        pivot_val = A[pivot_row, pivot_col]
        if pivot_val != 1:
            print(f"\nStep {step}: Normalize row {pivot_row+1} (divide by {pivot_val})")
            A.row_op(pivot_row, lambda x, _: x / pivot_val)
            sp.pprint(A)
            step += 1

        for r in range(pivot_row + 1, rows):
            factor = A[r, pivot_col]
            if factor != 0:
                print(f"\nStep {step}: row {r+1} - ({factor})*row {pivot_row+1}")
                A.row_op(r, lambda x, j: x - factor * A[pivot_row, j])
                sp.pprint(A)
                step += 1

        pivot_row += 1

    # Backward pass — clear out entries above each pivot too
    for pivot_row in reversed(range(rows)):
        pivot_cols = [c for c in range(cols) if A[pivot_row, c] == 1]
        if not pivot_cols:
            continue
        pivot_col = pivot_cols[0]

        for r in range(pivot_row):
            factor = A[r, pivot_col]
            if factor != 0:
                print(f"\nStep {step}: row {r+1} - ({factor})*row {pivot_row+1}")
                A.row_op(r, lambda x, j: x - factor * A[pivot_row, j])
                sp.pprint(A)
                step += 1

    print("\n=== FINAL RREF MATRIX ===")
    sp.pprint(A)
    return A


# ---------------------- Matrix Inversion ----------------------
def find_inverse(A):
    """
    Computes the inverse of a square matrix A using augmented matrix method.
    Steps:
        1. Form augmented matrix [A | I]
        2. Reduce to RREF to get [I | A⁻¹]
    Prints each major step for educational purposes.

    Parameters:
        A (sp.Matrix): Invertible square matrix

    Returns:
        sp.Matrix: Inverse of A
    """
    n, m = A.shape
    if n != m:
        raise ValueError("Matrix must be square to compute its inverse.")

    # Augment A with the identity — RREF will turn the left side into I
    # and the right side into A⁻¹
    print("\n=== AUGMENTED MATRIX [ A | I ] ===")
    I = sp.eye(n)
    augmented = A.row_join(I)
    sp.pprint(augmented)

    print("\n=== REDUCING TO [ I | A⁻¹ ] VIA RREF ===")
    inverse_augmented = rref(augmented)

    print("\n=== FINAL INVERSE MATRIX (Right Half) ===")
    inverse = inverse_augmented[:, n:]  # everything to the right of the dividing line
    sp.pprint(inverse)

    # Quick sanity check — A times its inverse should give us back the identity
    print("\n=== VALIDATION: A * A⁻¹ = I ===")
    product = A * inverse
    simplified_product = product.applyfunc(sp.simplify)
    sp.pprint(simplified_product)
    if simplified_product == sp.eye(n):
        print("Validation passed: A × A⁻¹ = I")
    else:
        print("Validation failed: A × A⁻¹ ≠ I")

    return inverse


# ---------------------- Characteristic Polynomial & Eigenvalues ----------------------
def manual_factor_polynomial(poly, var):
    """
    Factor a polynomial and compute its roots.
    Prints factorization, roots, and multiplicities.

    Parameters:
        poly (sp.Expr): Polynomial expression
        var (sp.Symbol): Variable in polynomial

    Returns:
        dict: Dictionary mapping root -> algebraic multiplicity
    """
    print("\n=== FACTORING CHARACTERISTIC POLYNOMIAL ===")
    print(f"Polynomial: {poly}")
    factored = sp.factor(poly)
    print(f"Factored form: {factored}")

    roots = sp.solve(poly, var)
    eigenvals_with_mult = {}
    for root in roots:
        root_simplified = sp.simplify(root)
        eigenvals_with_mult[root_simplified] = roots.count(root)

    print("Roots and multiplicities:")
    for root, mult in eigenvals_with_mult.items():
        print(f"  λ = {root}, multiplicity = {mult}")
    print("=" * 60)

    return eigenvals_with_mult


def compute_eigenvalues(A):
    """
    Compute eigenvalues of a square matrix A manually using:
        det(A - λI) = 0

    Returns:
        eigenvalues (list): Eigenvalues with repetitions
        eigenvals_dict (dict): Root -> algebraic multiplicity
        char_poly (sp.Expr): Characteristic polynomial
    """
    if A.rows != A.cols:
        raise ValueError("Matrix must be square for eigenvalues.")

    n = A.rows
    lam = sp.Symbol('lambda')
    I = sp.eye(n)

    # The characteristic polynomial comes from det(A - λI) = 0
    print("\n=== COMPUTING CHARACTERISTIC POLYNOMIAL ===")
    char_matrix = A - lam * I
    print("Characteristic matrix (A - λI):")
    sp.pprint(char_matrix)

    char_poly_raw = get_determinant(char_matrix)
    char_poly = sp.expand(char_poly_raw)
    print(f"\nExpanded characteristic polynomial: {char_poly}")

    eigenvals_dict = manual_factor_polynomial(char_poly, lam)

    # Flatten into a plain list, repeating each eigenvalue by its multiplicity
    eigenvalues = []
    for eigval, mult in eigenvals_dict.items():
        eigenvalues.extend([eigval] * mult)

    print("Summary of eigenvalues (with multiplicities):")
    for eigval, mult in eigenvals_dict.items():
        print(f"λ = {eigval}, algebraic multiplicity = {mult}")
    return eigenvalues, eigenvals_dict, char_poly


# ---------------------- Null Space Extraction ----------------------
def extract_nullspace_from_rref(rref_matrix):
    """
    Constructs basis vectors for the null space from RREF matrix.

    Parameters:
        rref_matrix (sp.Matrix): RREF of (A - λI)

    Returns:
        list: List of basis vectors (sp.Matrix) for null space
    """
    rows, cols = rref_matrix.shape
    pivot_cols = []
    pivot_rows = []

    # Walk through to find which columns have pivots and which are free
    for row in range(rows):
        for col in range(cols):
            if rref_matrix[row, col] != 0:
                pivot_cols.append(col)
                pivot_rows.append(row)
                break

    free_cols = [c for c in range(cols) if c not in pivot_cols]

    print(f"Pivot columns: {[c+1 for c in pivot_cols]}")
    print(f"Free columns: {[c+1 for c in free_cols]}")

    if not free_cols:
        print("No free variables: Null space = {0}")
        return []

    # One basis vector per free variable — set that free var to 1, solve for the rest
    basis_vectors = []
    for i, free_col in enumerate(free_cols):
        vec = [0] * cols
        vec[free_col] = 1

        for row in reversed(range(len(pivot_rows))):
            pivot_col = pivot_cols[row]
            sum_terms = sum(rref_matrix[row, col] * vec[col] for col in range(pivot_col+1, cols))
            vec[pivot_col] = -sum_terms
        basis_vectors.append(sp.Matrix(vec))

    return basis_vectors


# ---------------------- Eigenvectors ----------------------
def compute_eigenvectors(A, eigenvalues):
    """
    Compute eigenvectors for all eigenvalues of matrix A.
    Uses manual RREF and null space extraction.

    Returns:
        dict: eigenvalue -> list of basis eigenvectors
    """
    n = A.shape[0]
    eigenvectors_dict = {}
    unique_eigenvalues = list(set(eigenvalues))

    for eigval in unique_eigenvalues:
        print(f"\nEigenvectors for λ = {eigval}")
        I = sp.eye(n)
        # Null space of (A - λI) gives us the eigenvectors for this eigenvalue
        null_matrix = A - eigval * I
        rref_matrix = rref(null_matrix)
        eigenvects = extract_nullspace_from_rref(rref_matrix)
        eigenvectors_dict[eigval] = eigenvects

    return eigenvectors_dict


# ---------------------- Diagonalization ----------------------
def diagonalize_matrix(A):
    """
    Diagonalizes matrix A if possible.
    Returns matrices P (eigenvectors) and D (diagonal eigenvalues).
    """
    n = A.shape[0]
    if A.rows != A.cols:
        raise ValueError("Matrix must be square to diagonalize.")

    eigenvalues, eigenvals_dict, _ = compute_eigenvalues(A)
    eigenvectors_dict = compute_eigenvectors(A, eigenvalues)

    # Need as many independent eigenvectors as the matrix has dimensions
    total_eigenvectors = sum(len(vecs) for vecs in eigenvectors_dict.values())
    if total_eigenvectors < n:
        print("Matrix is NOT diagonalizable (insufficient independent eigenvectors).")
        return None, None

    # P's columns are the eigenvectors; D's diagonal entries are the matching eigenvalues
    P_columns = []
    D_entries = []
    for eigval in eigenvals_dict.keys():
        for vec in eigenvectors_dict[eigval]:
            P_columns.append(vec)
            D_entries.append(eigval)
    P = sp.Matrix.hstack(*P_columns)
    D = sp.diag(*D_entries)

    P_inv = find_inverse(P)
    if P * D * P_inv != A:
        print("Warning: Verification failed for A = P D P⁻¹")
    else:
        print("Matrix successfully diagonalized: A = P D P⁻¹")

    return P, D


def get_matrix_for_eigen():
    """
    Prompts the user to input a square matrix for eigenvalue and diagonalization computations.
    Ensures valid input (correct size, numeric/symbolic entries).

    Returns:
        sp.Matrix object containing the user's input.
    """
    while True:
        try:
            print("\n" + "="*60)
            print("      MATRIX INPUT (FOR EIGENVALUES/DIAGONALIZATION)")
            print("="*60)

            n = int(input("Enter the size n of the n x n matrix: "))
            print(f"\nEnter the matrix row by row (each row should have {n} numbers separated by spaces):")
            matrix = []

            for i in range(n):
                # sympify lets users type things like sqrt(2) or 1/3 and have it work
                row = list(map(sp.sympify, input(f"Row {i+1}: ").strip().split()))
                if len(row) != n:
                    raise ValueError(f"Each row must have exactly {n} elements.")
                matrix.append(row)

            A = sp.Matrix(matrix)
            print("\nMatrix entered:")
            sp.pprint(A)
            return A

        except Exception as e:
            print("Invalid input:", e)
            print("Please try again.\n")


# ============================================================
# DRIVER CODE
# ============================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("       MATRIX OPERATIONS TOOLKIT")
    print("="*60)
    print("\nAvailable operations:")
    print("1. Compute determinant (step-by-step cofactor expansion)")
    print("2. Find inverse matrix (via manual RREF)")
    print("3. Compute eigenvalues (manual characteristic polynomial)")
    print("4. Compute eigenvectors (manual null space extraction)")
    print("5. Diagonalize matrix (complete manual process)")
    print("="*60)

    choice = input("\nEnter your choice (1-5): ").strip()

    if choice == "1":
        A = user_input_matrix()
        get_determinant(A)

    elif choice == "2":
        A = get_invertible_matrix()
        find_inverse(A)

    elif choice == "3":
        A = get_matrix_for_eigen()
        compute_eigenvalues(A)

    elif choice == "4":
        A = get_matrix_for_eigen()
        eigenvalues, eigenvals_dict, _ = compute_eigenvalues(A)
        compute_eigenvectors(A, list(eigenvals_dict.keys()))

    elif choice == "5":
        A = get_matrix_for_eigen()
        P, D = diagonalize_matrix(A)

    else:
        print("Invalid choice! Please restart and select a number 1-5.")