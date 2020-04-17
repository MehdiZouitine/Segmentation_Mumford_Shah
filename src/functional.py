import numpy as np
import math
from typing import Union, Tuple, List
import pandas as pd
from tqdm import tqdm_notebook
from itertools import combinations
import matplotlib.pyplot as plt


def munford_shah_fonctional(
    u: np.ndarray, w: np.ndarray, omega: List[tuple], l: float, m: float,
) -> float:

    """Short summary:

        Function that evals the Munford-Shah fonctional.

    Parameters
    ----------
    u : np.ndarray
        Description of parameter `u`.
    w : np.ndarray
        Description of parameter `w`.
    omega : List[tuple]
        Description of parameter `omega`.
    l : float
        Description of parameter `l`.
    m : float
        Description of parameter `m`.

    Returns
    -------
    float
        Description of returned object.

    """

    return (
        get_perimeter(omega)
        + l * np.sum(np.abs(image_gradient(w)) ** 2)
        + m * np.linalg.norm(w - u, 2) ** 2
    )


def dl(pixel1: np.ndarray, pixel2: np.ndarray, frontier: List[tuple]) -> int:

    """Short summary:

        Fonction that compute piece of frontier perimeter.

    Parameters
    ----------
    pixel1 : np.ndarray
        Description of parameter `pixel1`.
    pixel2 : np.ndarray
        Description of parameter `pixel2`.
    frontier : List[tuple]
        Description of parameter `frontier`.

    Returns
    -------
    int
        Description of returned object.

    """

    # Pour le dl nous choisissons la variation total de fonction caracteristique de notre omega (On choisira pour ce cas d'usage la 4-connexité)
    if (
        (pixel1 in frontier)
        and (pixel2 in frontier)
        and (pixel1[0] != pixel2[0])
        and (pixel1[1] != pixel2[1])
    ):
        return 1
    return 0


def H_eps(t: float, eps: float) -> float:

    """Short summary:

        Function that eval H_eps function (see chapter 6).

    Parameters
    ----------
    t : float
        Description of parameter `t`.
    eps : float
        Description of parameter `eps`.

    Returns
    -------
    float
        Description of returned object.

    """

    if t >= eps:
        return 1

    elif t >= -eps and t <= eps:
        return (1 / 2) * (1 + (t / eps) + (1 / math.pi) * math.sin(math.pi * t / eps))

    else:
        return 0


def H_eps_derivative(t: float, eps: float) -> float:

    """Short summary:

        Function that eval H_eps derivative function (see chapter 6).

    Parameters
    ----------
    t : float
        Description of parameter `t`.
    eps : float
        Description of parameter `eps`.

    Returns
    -------
    float
        Description of returned object.

    """

    if t >= eps:
        return 0

    elif t >= -eps and t <= eps:
        return (1 / 2 * eps) * (1 + math.cos(math.pi * t / eps))

    else:
        return 0


def grad_w_part(
    w: np.ndarray, u: np.ndarray, frontier: List[List], lambda_: float, mu: float
) -> np.ndarray:

    image_size = w.shape

    h1_term = np.zeros(image_size)

    data_term = 2 * (w - u)

    w = np.pad(w, 1, mode="constant")

    for i in range(image_size[0]):

        for j in range(image_size[1]):

            if [i, j] not in frontier:

                h1_term[i, j] = -2 * (w[i + 1, j] + w[i, j + 1] - 2 * w[i, j])

    return lambda_ * h1_term + mu * data_term


def grad_phi_part(phi, omega_frontier, eps):
    omega_term = np.zeros(phi.shape)
    for tuple1 in omega_frontier:
        for tuple2 in omega_frontier:
            omega_term[tuple1[0]][tuple1[1]] += (
                dl(tuple1, tuple2, omega_frontier)
                * H_eps_derivative(phi[tuple1[0], tuple1[1]], eps)
                * (1 - 2 * H_eps(phi[tuple2[0], tuple2[1]], eps))
            )
    return omega_term
