"""
fem_utils.py - helper functions copied unchanged from type_II-elas.ipynb
so other notebooks can import them without re-running that notebook.

    from fem_utils import plot_errors, plot_sparseness, generate_error_table_general
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ufl import sqrt
from dolfinx.fem import assemble_matrix, form


def compute_von_mises_stress(sigma):
    s11 = sigma[0, 0]
    s22 = sigma[1, 1]
    s12 = sigma[0, 1]

    # Von Mises stress formula for plane strain
    von_mises = sqrt(s11**2 + s22**2 - s11 * s22 + 3 * s12**2)
    return von_mises


def generate_l_shaped_mesh(x_min: float, x_max: float, y_min: float, y_max: float, h: float, q: int):
    import pygmsh   # only needed for this function
    z = 0.0
    x_mid = (x_min + x_max)/2
    y_mid = (y_min + y_max)/2
    h_vtcs = h
    with pygmsh.geo.Geometry() as geom:
        points = [
            geom.add_point([x_min, y_min, z], mesh_size=h_vtcs),
            geom.add_point([x_max, y_min, z], mesh_size=h_vtcs),
            geom.add_point([x_max, y_mid, z], mesh_size=h_vtcs),
            geom.add_point([x_mid, y_mid, z], mesh_size=h_vtcs**q),
            geom.add_point([x_mid, y_max, z], mesh_size=h_vtcs),
            geom.add_point([x_min, y_max, z], mesh_size=h_vtcs),
        ]

        lines = [
            geom.add_line(points[0], points[1]),  # Bottom horizontal
            geom.add_line(points[1], points[2]),  # Right vertical
            geom.add_line(points[2], points[3]),  # Inner horizontal
            geom.add_line(points[3], points[4]),  # Inner vertical
            geom.add_line(points[4], points[5]),  # Top horizontal (left part)
            geom.add_line(points[5], points[0]),  # Left vertical
        ]

        # Create a line loop to define the L-shaped domain boundary
        line_loop = geom.add_curve_loop(lines)

        surface = geom.add_plane_surface(line_loop)

        mesh = geom.generate_mesh()

    return mesh


def plot_errors(h, e, l, x_label='$h$', y_label=r'$\epsilon_{L_2}$', title=''):
    """
    Plots the error vs mesh size on a log-log plot.

    Args:
    h : array-like
        Vector of mesh sizes.
    e : 2D array-like
        Matrix where each row corresponds to the errors for a specific label.
    l : array-like
        List of labels for each row in the error matrix.
    """

    plt.figure(figsize=(8, 6))

    for i, label in enumerate(l):
        plt.loglog(h[i], e[i], label=label, marker='o')

    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.title(title, fontsize=14)

    plt.legend()

    plt.grid(True, which="both", ls="--", linewidth=0.5)

    plt.show()


def plot_sparseness(a):
    A = assemble_matrix(form(a))

    A_scipy = A.to_scipy()

    no_non_zeros = A_scipy.nnz
    no_dofs = A_scipy.shape[0]

    print(f"Number of degrees of freedom: {no_dofs}")
    print(f"Number of non-zeros: {no_non_zeros}")
    print(f"Covering fraction: {100*no_non_zeros/no_dofs/no_dofs:.2f}%")

    # Plot the sparsity pattern
    plt.figure(figsize=(8, 8))
    plt.spy(A_scipy, markersize=1)
    plt.show()


def generate_error_table_general(ms, e, l):
    """
    Generates a pandas DataFrame containing mesh sizes, errors, and convergence rates.
    This version handles any number of error rows.

    Args:
    - ms: List of mesh sizes.
    - e: List of lists containing error data for different methods.
    - l: List of column labels (first label is for mesh sizes).

    Returns:
    - DataFrame containing errors and convergence rates.
    """

    # Create a pandas DataFrame with mesh sizes
    data = pd.DataFrame({l[0]: ms})

    # Add the error columns to the DataFrame
    for i in range(1, len(l)):
        data[l[i]] = e[i-1]

    # Function to calculate convergence rates
    def compute_convergence_rates(errors, mesh_sizes):
        rates = []
        for i in range(1, len(errors)):
            rate = np.log(errors[i] / errors[i-1]) / np.log(mesh_sizes[i] / mesh_sizes[i-1])
            rates.append(rate)
        rates.insert(0, np.nan)  # Insert NaN for the first entry where rate can't be computed
        return rates

    # Add the convergence rate columns
    for i in range(1, len(l)):
        rate_col = f'rate_{l[i]}'
        data[rate_col] = compute_convergence_rates(e[i-1], ms)

    return data
