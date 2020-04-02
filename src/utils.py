# Code de toutes les fonctions utiles au projet, histoire de pas tout foutre dans le notebook.
import numpy as np
import math
from typing import Union, Tuple, List
import pandas as pd
from tqdm import tqdm_notebook

EPS = 0.1


def get_neighborhood(
    image: np.ndarray, pixel: Tuple, neighborhood_type: str = "4_connex"
) -> List[tuple]:
    """Short summary:

        Function that compute the neighborhood of a pixel using (4 or 8 connexity).
        In our case we will use 4 connexity.

    Parameters
    ----------
    image : np.ndarray
        Description of parameter `image`.
    pixel : Tuple
        Description of parameter `pixel`.
    neighborhood_type : str
        Description of parameter `neighborhood_type`.

    Returns
    -------
    List[tuple]
        Description of returned object.

    """

    shape = np.shape(image)
    real_neighborhood = []

    if neighborhood_type == "4_connex":

        neighborhood = [
            (pixel[0], pixel[1] + 1),
            (pixel[0], pixel[1] - 1),
            (pixel[0] + 1, pixel[1]),
            (pixel[0] - 1, pixel[1]),
        ]

        for neighbour in neighborhood:
            if neighbour[0] < shape[0] and neighbour[1] < shape[1]:
                real_neighborhood.append(neighbour)
    elif neighborhood_type == "8_connex":

        neighborhood = [
            (pixel[0], pixel[1] + 1),
            (pixel[0], pixel[1] - 1),
            (pixel[0] + 1, pixel[1]),
            (pixel[0] - 1, pixel[1]),
            (pixel[0] + 1, pixel[1] + 1),
            (pixel[0] - 1, pixel[1] - 1),
            (pixel[0] + 1, pixel[1] - 1),
            (pixel[0] - 1, pixel[1] + 1),
        ]

        for neighbour in neighborhood:
            if neighbour[0] < shape[0] and neighbour[1] < shape[1]:
                real_neighborhood.append(neighbour)

    else:
        raise ValueError(
            "The neighborhood method must be chosen among 4_connex or 8_connex"
        )

    return real_neighborhood


def get_frontier(
    image: np.ndarray, omega: List[tuple], neighborhood_type: str
) -> List[Tuple]:

    """Short summary:

        Function that find the frontier of a given neighborhood.

    Parameters
    ----------
    image : np.ndarray
        Description of parameter `image`.
    omega : List[tuple]
        Description of parameter `omega`.
    neighborhood_type : str
        Description of parameter `neighborhood_type`.

    Returns
    -------
    List[Tuple]
        Description of returned object.

    """

    frontier = []

    for elt in omega:

        elt_neighborhood = get_neighborhood(image, elt, neighborhood_type)

        for potential_frontier_elt in elt_neighborhood:
            if potential_frontier_elt not in omega:
                frontier.append(elt)

    return list(set(frontier))


def get_perimeter(frontier: List[Tuple]) -> int:
    """Short summary:

        Function that compute the perimeter of a given frontier.
        We probably won't use it.


    Parameters
    ----------
    frontier : List[Tuple]
        Description of parameter `frontier`.

    Returns
    -------
    int
        Description of returned object.

    """

    return len(frontier) - 1


def image_gradient(w: np.ndarray) -> np.ndarray:
    """Short summary:

        Function that compute image gradient (given by difference beetween 2 neighbours)
        on each axis (dx and dy). It use pandas shift function to be faster.


    Parameters
    ----------
    w : np.ndarray
        Description of parameter `w`.

    Returns
    -------
    np.ndarray
        Description of returned object.

    """
    return np.array(
        [
            (pd.DataFrame(w) - pd.DataFrame(w).shift(1, axis=0))
            .fillna(method="bfill", axis=0)
            .to_numpy(),
            (pd.DataFrame(w).shift(-1, axis=1) - pd.DataFrame(w))
            .fillna(method="ffill", axis=1)
            .to_numpy(),
        ]
    )


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
    if (pixel1 in frontier) and (pixel2 in frontier) and (pixel1 != pixel2):
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
    w: np.ndarray, u: np.ndarray, omega: List[tuple], lambda_: float, mu: float
) -> np.ndarray:

    image_size = w.shape
    frontier = get_frontier(image=u, omega=omega, neighborhood_type="4_connex")
    h1_term = np.zeros(image_size)
    data_term = np.zeros(image_size)
    w = np.pad(w, 1, mode="constant")
    print(w)
    for i in range(image_size[0]):
        for j in range(image_size[1]):
            if (i, j) in frontier:
                h1_term[i, j] = -2 * (w[i + 1, j] + w[i, j + 1] - 2 * w[i, j])
            else:
                h1_term[i, j] = 0

            data_term[i, j] = 2 * (w[i, j] - u[i, j])

    return lambda_ * h1_term + mu * data_term


def grad_phi_part():
    return 0


def gradient_descent(
    phi_0: np.ndarray,
    u: np.ndarray,
    step: float,
    trhld: float,
    omega: List[Tuple],
    eps: float,
) -> np.ndarray:

    """Short summary:

       Function that compute simple gradient descent algorithm.

    Parameters
    ----------
    phi_0 : np.ndarray
        Description of parameter `phi_0`.
    u : np.ndarray
        Description of parameter `u`.
    step : float
        Description of parameter `step`.
    trhld : float
        Description of parameter `trhld`.
    omega : List[Tuple]
        Description of parameter `omega`.
    eps : float
        Description of parameter `eps`.

    Returns
    -------
    np.ndarray
        Description of returned object.

    """

    phi = phi_0
    while np.linalg.norm(grad_munford_shah(phi, u, eps, omega)) > trhld:
        phi = phi - step * grad_munford_shah(phi, u, eps, omega)
        print(np.linalg.norm(grad_munford_shah(phi, u, eps, omega)))

    return phi
