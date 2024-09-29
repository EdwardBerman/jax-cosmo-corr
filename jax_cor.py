import jax.numpy as jnp
from jax import grad, jit, vmap, jacfwd, jacrev
import jax.random as random
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass

'''
$Y = \left\{y_1, \hdots, y_n\right\}$ are the object positions,
$c$ is the number of clusters in $Y$; $2 \leq c < n$,
$m$ is the weighting exponent (or fuzziness); $1 \leq m < \infty$,
$U$ is the fuzzy c-partition of $Y$; $U \in \mathbb{R}^{c \times N}$
$v = (v_1, \hdots v_c)$ is a vector of centers,
$v_i = (v_{i1}, \hdots, v_{in})$, is the center of cluster $i$.
'''

@dataclass
class galaxy:
    coord: jnp.ndarray
    quantity_matrix: jnp.ndarray

@dataclass
class galaxy_pair:
    galaxy1: galaxy
    galaxy2: galaxy
    distance: float

def vincenty_formula(coord1: jnp.ndarray, coord2: jnp.ndarray) -> float:
    phi1, lambda1 = coord1
    phi2, lambda2 = coord2

    phi1 = phi1 * jnp.pi / 180
    lambda1 = lambda1 * jnp.pi / 180
    phi2 = phi2 * jnp.pi / 180
    lambda2 = lambda2 * jnp.pi / 180

    delta_lambda = lambda2 - lambda1
    c1 = (jnp.cos(phi2) * jnp.sin(delta_lambda))**2
    c2 = (jnp.cos(phi1) * jnp.sin(phi2) - jnp.sin(phi1) * jnp.cos(phi2) * jnp.cos(delta_lambda))**2
    c3 = jnp.sin(phi1) * jnp.sin(phi2) + jnp.cos(phi1) * jnp.cos(phi2) * jnp.cos(delta_lambda)

    y = jnp.sqrt(c1 + c2)
    x = c3
    Δσ = jnp.arctan2(y, x)

    distance = Δσ * (180 / jnp.pi) * 60

    return distance


def update_centers(U: jnp.ndarray, Y: jnp.ndarray, fuzziness: float) -> jnp.ndarray:
    '''
    \hat{v}_i = \frac{\sum_{k=1}^N \left(\hat{u}_{ik}\right)^m y_k}{\sum_{k=1}^N \left(\hat{u}_{ik}\right)^m} ; 1 \leq i \leq c
    '''
    c, N = U.shape
    new_centers = []
    for i in range(c):
        sum1 = 0
        sum2 = 0
        for j in range(N):
            sum1 += U[i, j]**fuzziness * Y[j]
            sum2 += U[i, j]**fuzziness
        new_centers.append(sum1 / (sum2 + 1e-6))
    return jnp.array(new_centers)

def update_membership(Y: jnp.ndarray, v: jnp.ndarray, fuzziness: float, distance_metric: Optional[Callable[[jnp.ndarray, jnp.ndarray], float]] = vincenty_formula) -> jnp.ndarray:
    '''
    \hat{u}_{ik} = \left( \sum_{j=1}^c \left(\frac{\hat{d}_{ik}}{\hat{d}_{jk}}\right)^{\frac{2}{m-1}} \right)^{-1}
    '''
    c = v.shape[0]
    N = Y.shape[0]
    U = jnp.zeros((c, N))
    for i in range(c):
        for k in range(N):
            sum1 = 0
            for j in range(c):
                sum1 += (distance_metric(Y[k], v[i]) / distance_metric(Y[k], v[j]))**(2 / (fuzziness - 1))
            U = U.at[i, k].set(1 / (sum1 + 1e-6))
    return U

def sigmoid_weighting(lower_bound: float, upper_bound: float, galaxy_pair: galaxy_pair, sharpness: Optional[float] = 10) -> float:
    distance = galaxy_pair.distance
    sigmoid_lower_bound = 1 / (1 + jnp.exp(-sharpness * (distance - lower_bound)) + 1e-6)
    sigmoid_upper_bound = 1 / (1 + jnp.exp(-sharpness * (upper_bound - distance)) + 1e-6)
    return sigmoid_lower_bound * sigmoid_upper_bound

class fuzzy_c_means:
    def __init__(self, c: int, m: float, Y: jnp.ndarray, distance_metric: Optional[Callable[[jnp.ndarray, jnp.ndarray], float]] = vincenty_formula):
        self.c = c
        self.m = m
        self.Y = Y
        self.distance_metric = distance_metric
        self.U = random.uniform(random.PRNGKey(0), (c, Y.shape[0]))
        self.U = self.U / jnp.sum(self.U, axis=0)
        self.v = random.uniform(random.PRNGKey(0), (c, Y.shape[1]))

    def fit(self, max_iter: int = 1000, tol: float = 1e-6):
        for _ in range(max_iter):
            new_centers = update_centers(self.U, self.Y, self.m)
            new_membership = update_membership(self.Y, new_centers, self.m, self.distance_metric)
            if jnp.sum(jnp.abs(new_membership - self.U)) < tol:
                break
            self.U = new_membership
            self.v = new_centers

    def predict(self, Y: jnp.ndarray) -> jnp.ndarray:
        return update_membership(Y, self.v, self.m, self.distance_metric)

@dataclass
class correlator_config:
    lower_bound: float
    upper_bound: float
    sharpness: float
    number_bins: int
    verbose: Optional[bool] = False
    fuzziness: Optional[float] = 2.0
    max_iter: Optional[int] = 1000
    tol: Optional[float] = 1e-6
    distance_metric: Optional[Callable[[jnp.ndarray, jnp.ndarray], float]] = vincenty_formula

class cosmic_correlator:
    def __init__(self, galaxies: List[galaxy], number_clusters: int, config: correlator_config):
        self.galaxies = galaxies
        self.number_clusters = number_clusters

        self.lower_bound = config.lower_bound
        self.upper_bound = config.upper_bound
        self.sharpness = config.sharpness
        self.distance_metric = config.distance_metric
        self.number_bins = config.number_bins
        self.m = config.fuzziness
        self.verbose = config.verbose
        self.max_iter = config.max_iter
        self.tol = config.tol

        self.Y = jnp.array([galaxy.coord for galaxy in galaxies])
        self.quantity_matrix = jnp.array([galaxy.quantity_matrix for galaxy in galaxies])
        self.U = random.uniform(random.PRNGKey(0), (self.number_clusters, self.Y.shape[0]))
        self.U = self.U / jnp.sum(self.U, axis=0)
        self.v = random.uniform(random.PRNGKey(0), (self.number_clusters, self.Y.shape[1]))
        self.fcm = fuzzy_c_means(self.number_clusters, self.m, self.Y, self.distance_metric)

    def fcm_fit(self):
        self.fcm.fit(self.max_iter, self.tol)
        self.U = self.fcm.U
        self.v = self.fcm.v

config = correlator_config(lower_bound=0, upper_bound=10, sharpness=10, number_bins=10, verbose=True)
galaxy1 = galaxy(coord=jnp.array([0, 0]), quantity_matrix=jnp.array([1, 2, 3]))
galaxy2 = galaxy(coord=jnp.array([0, 1]), quantity_matrix=jnp.array([1, 2, 3]))
galaxy3 = galaxy(coord=jnp.array([1, 0]), quantity_matrix=jnp.array([1, 2, 3]))
galaxy4 = galaxy(coord=jnp.array([1, 1]), quantity_matrix=jnp.array([1, 2, 3]))
galaxies = [galaxy1, galaxy2, galaxy3, galaxy4]
correlator = cosmic_correlator(galaxies, 2, config)
correlator.fcm_fit()
print(correlator.U)
print(correlator.v)
