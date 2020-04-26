#!/usr/bin/env python
# -*- coding: utf-8 -*-
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt


def vectors_uniform(k):
    """Uniformly generates k vectors."""
    vectors = []
    for a in np.linspace(0, 2 * np.pi, k, endpoint=False):
        vectors.append(2 * np.array([np.sin(a), np.cos(a)]))
    return vectors


def visualize_transformation(A, vectors):
    """Plots original and transformed vectors for a given 2x2 transformation matrix A and a list of 2D vectors."""
    for i, v in enumerate(vectors):
        # Plot original vector.
        plt.quiver(
            0.0,
            0.0,
            v[0],
            v[1],
            width=0.008,
            color="blue",
            scale_units="xy",
            angles="xy",
            scale=1,
            zorder=4,
        )
        plt.text(v[0] / 2 + 0.25, v[1] / 2, "v{0}".format(i), color="blue")

        # Plot transformed vector.
        tv = A.dot(v)
        plt.quiver(
            0.0,
            0.0,
            tv[0],
            tv[1],
            width=0.005,
            color="magenta",
            scale_units="xy",
            angles="xy",
            scale=1,
            zorder=4,
        )
        plt.text(tv[0] / 2 + 0.25, tv[1] / 2, "v{0}'".format(i), color="magenta")
    plt.xlim([-6, 6])
    plt.ylim([-6, 6])
    plt.margins(0.05)
    # Plot eigenvectors
    plot_eigenvectors(A)
    plt.show()


def visualize_vectors(vectors, color="green"):
    """Plots all vectors in the list."""
    for i, v in enumerate(vectors):
        plt.quiver(
            0.0,
            0.0,
            v[0],
            v[1],
            width=0.006,
            color=color,
            scale_units="xy",
            angles="xy",
            scale=1,
            zorder=4,
        )
        plt.text(v[0] / 2 + 0.25, v[1] / 2, "eigv{0}".format(i), color=color)


def plot_eigenvectors(A):
    """Plots all eigenvectors of the given 2x2 matrix A."""
    _, eigvec = np.linalg.eig(A)
    visualize_vectors(eigvec.T)


def EVD_decomposition(A):
    eig_val, eigvec = np.linalg.eig(A)
    L = np.diag(eig_val)
    K = eigvec

    ## Jeżeli macież o rozmiarze 2x2
    if A.shape == (2, 2):
        K_inv = (
            1
            / (K[0, 0] * K[1, 1] - K[0, 1] * K[1, 0])
            * np.array([[K[1, 1], -K[0, 1]], [-K[1, 0], K[0, 0]]])
        )
    else:
        K_inv = np.linalg.inv(K)

    # Ostatnia macierz nie jest diagonalizowalna
    if np.isclose(np.linalg.det(K), 0.0):
        print(f"Matrix\n{A}\nis not digonalizable")
    else:
        print(
            f"For A =\n{A}\nK @ L @ K_inv=\n{K @ L @ K_inv}\nK=\n{K}\nL=\n{L}\nK_inv=\n{K_inv}"
        )
    pass


def plot_attractors(A, vectors):
    # TODO: Zad. 4.3. Uzupełnij funkcję tak by generowała wykres z atraktorami.
    """Plots original and transformed vectors for a given 2x2 transformation matrix A and a list of 2D vectors."""

    plt.xlim([-6, 6])
    plt.ylim([-6, 6])
    plt.margins(0.05)
    # Plot eigenvectors
    _, eigvec = np.linalg.eig(A)
    idx2color = ["orange", "blue", "red", "green"]
    eigvec = eigvec.T
    eigvec = np.concatenate([-eigvec, eigvec])

    is_all_eig = False
    if A[0][0] != 0 and not False in (A / A[0][0] == np.eye(A.shape[0])):
        is_all_eig = True
    for vector in vectors:
        vector = vector / np.linalg.norm(vector)
        tmp_vec = vector.copy()
        for _ in range(20):
            tmp_vec = A @ tmp_vec
            tmp_vec /= np.linalg.norm(tmp_vec)
        if is_all_eig:
            color = "black"
        else:
            color = idx2color[
                int(
                    np.argmin(
                        np.linalg.norm(
                            (eigvec / np.linalg.norm(eigvec, axis=1).reshape((-1, 1)))
                            - tmp_vec,
                            axis=1,
                        )
                    )
                )
            ]
        plt.quiver(
            0.0,
            0.0,
            vector[0],
            vector[1],
            width=0.003,
            color=color,
            scale_units="xy",
            angles="xy",
            scale=1,
            zorder=4,
        )
    for vector, color in zip(eigvec, idx2color):
        plt.quiver(
            0.0,
            0.0,
            *vector,
            width=0.006,
            color=color,
            scale_units="xy",
            angles="xy",
            scale=1,
            zorder=5,
        )
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.show()


def show_eigen_info(A, vectors):
    EVD_decomposition(A)
    visualize_transformation(A, vectors)
    plot_attractors(A, vectors)


if __name__ == "__main__":
    vectors = vectors_uniform(k=16)

    A = np.array([[2, 0],
                  [0, 2]])
    show_eigen_info(A, vectors)


    A = np.array([[-1, 2],
                  [2, 1]])
    show_eigen_info(A, vectors)


    A = np.array([[3, 1],
                  [0, 2]])
    show_eigen_info(A, vectors)


    A = np.array([[2, -1],
                  [1, 4]])
    show_eigen_info(A, vectors)
