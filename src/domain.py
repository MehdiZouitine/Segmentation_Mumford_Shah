import numpy as np
import math
from typing import Union, Tuple, List, Dict
import pandas as pd
from tqdm import tqdm_notebook
from itertools import combinations
import matplotlib.pyplot as plt


EPS = 0.1


def in_shape(img, pixel):
    return (
        pixel[0] > -1
        and pixel[0] < img.shape[0]
        and pixel[1] > -1
        and pixel[1] < img.shape[1]
    )


sign = lambda x: math.copysign(1, x)  # two will work


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


def dl(pixel1: np.ndarray, pixel2: np.ndarray, frontier: List[List]) -> int:

    """Short summary:

        Fonction that compute piece of frontier perimeter.

    Parameters
    ----------
    pixel1 : np.ndarray
        Description of parameter `pixel1`.
    pixel2 : np.ndarray
        Description of parameter `pixel2`.
    frontier : List[List]
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


def dl2(
    pixel1: np.ndarray, pixel2: np.ndarray, frontier: List[tuple], w: np.ndarray
) -> float:

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
    w : np.ndarray
        Description of parameter `w`.

    Returns
    -------
    float
        Description of returned object.

    """

    # Pour le dl nous choisissons la variation total de fonction caracteristique de notre omega (On choisira pour ce cas d'usage la 4-connexité)
    if (pixel1 in frontier) and (pixel2 in frontier):
        partial_x = w[pixel1[0] + 1, pixel1[1]] - w[pixel1[0], pixel1[1]]
        partial_y = w[pixel1[0], pixel1[1] + 1] - w[pixel1[0], pixel1[1]]

        return 1 - (partial_x ** 2 + partial_y ** 2)
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

    """Short summary:
       Fonction that compute gradient of w part of the gradient.


    Parameters
    ----------
    w : np.ndarray
        Description of parameter `w`.
    u : np.ndarray
        Description of parameter `u`.
    frontier : List[List]
        Description of parameter `frontier`.
    lambda_ : float
        Description of parameter `lambda_`.
    mu : float
        Description of parameter `mu`.

    Returns
    -------
    np.ndarray
        Description of returned object.

    """

    image_size = w.shape

    h1_term = np.zeros(image_size)
    data_term = 2 * (w - u)

    w = np.pad(w, 1, mode="edge")

    for i in range(image_size[0]):
        for j in range(image_size[1]):
            if [i, j] not in frontier:
                h1_term[i, j] = -2 * (w[i + 1, j] + w[i, j + 1] - 2 * w[i, j])

    return lambda_ * h1_term + mu * data_term


def grad_phi_part(
    phi: np.ndarray, w: np.ndarray, omega_frontier: List[List], eps: float
):
    """Short summary:
       Fonction that compute gradient of phi part of the gradient.

    Parameters
    ----------
    phi : np.ndarray
        Description of parameter `phi`.
    w : np.ndarray
        Description of parameter `w`.
    omega_frontier : List[List]
        Description of parameter `omega_frontier`.
    eps : float
        Description of parameter `eps`.

    Returns
    -------
    type
        Description of returned object.

    """
    w = np.pad(w, 1, mode="edge")
    omega_term = np.zeros(phi.shape)
    comb = combinations(omega_frontier, 2)
    for (tuple1, tuple2) in comb:
        omega_term[tuple1[0]][tuple1[1]] += (
            dl2(tuple1, tuple2, omega_frontier, w)
            * H_eps_derivative(phi[tuple1[0], tuple1[1]], eps)
            * (1 - 2 * H_eps(phi[tuple2[0], tuple2[1]], eps))
        )
    return omega_term


def get_frontier_phi(omega: List[List], phi: np.ndarray) -> List[List]:
    """Short summary:
       Fonction that compute the frontier of omega.

    Parameters
    ----------
    omega : List[List]
        Description of parameter `omega`.
    phi : np.ndarray
        Description of parameter `phi`.

    Returns
    -------
    List[List]
        Description of returned object.

    """
    frontier = []
    for pixel in omega:
        i = pixel[0]
        j = pixel[1]

        if (
            not in_shape(phi, (i + 1, j))
            or not in_shape(phi, (i, j + 1))
            or not in_shape(phi, (i - 1, j))
            or not in_shape(phi, (i, j - 1))
        ):
            frontier.append([i, j])

        elif (
            sign(phi[i, j]) == -sign(phi[i + 1, j])
            or sign(phi[i, j]) == -sign(phi[i, j + 1])
            or sign(phi[i, j]) == -sign(phi[i - 1, j])
            or sign(phi[i, j]) == -sign(phi[i, j - 1])
        ):
            frontier.append([i, j])
    return frontier


def gradient_descent(
    u: np.ndarray,
    step_w: float,
    step_phi: float,
    eps: float,
    lambda_: float,
    mu: float,
    it: int,
    verbose: bool,
    mode: str,
) -> Dict[str, Union[np.ndarray, List]]:

    """Short summary:

       Function that compute simple gradient descent algorithm.

    Parameters
    ----------
    u : np.ndarray
        Description of parameter `u`.
    step_w : float
        Description of parameter `step_w`.
    step_phi : float
        Description of parameter `step_phi`.
    eps : float
        Description of parameter `eps`.
    lambda_ : float
        Description of parameter `lambda_`.
    mu : float
        Description of parameter `mu`.
    it : int
        Description of parameter `it`.
    verbose : bool
        Description of parameter `verbose`.
    mode : str
        Description of parameter `mode`.

    Returns
    -------
    Dict[str,Union[np.ndarray,List]]
        Description of returned object.

    """

    phi = np.random.uniform(-1, 1, u.shape)
    positive_part = np.argwhere(phi >= 0).tolist()
    frontier = get_frontier_phi(omega=positive_part, phi=phi)
    norm_grad_phi = []
    norm_grad_w = []
    w = u

    if mode == "standard":
        for i in range(it):
            print(f"itération {i}/{it}")
            grad_w = grad_w_part(w, u, frontier, lambda_, mu)
            w = w + step_w * grad_w

            grad_phi = grad_phi_part(phi=phi, w=w, omega_frontier=frontier, eps=eps)
            phi = phi + step_phi * grad_phi

            positive_part = np.argwhere(phi >= 0).tolist()
            frontier = get_frontier_phi(omega=positive_part, phi=phi)

            norm_grad_phi.append(np.linalg.norm(grad_phi))
            norm_grad_w.append(np.linalg.norm(grad_w))
            if verbose:
                print(f"itération {i} : w gradient: {norm_grad_w[-1]}")
                print(f"itération {i} : phi gradient: {norm_grad_phi[-1]}")

    return {
        "w": w,
        "phi": phi,
        "frontier": frontier,
        "norm_grad_phi": norm_grad_phi,
        "norm_grad_w": norm_grad_w,
    }
