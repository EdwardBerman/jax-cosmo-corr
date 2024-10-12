import jax.numpy as jnp
from jax import grad, jit, vmap, jacfwd, jacrev, lax
import jax.random as random
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass
from functools import partial

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
    quantities: jnp.ndarray

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

def calculate_direction(x_1, x_2, y_1, y_2, z_1, z_2) -> jnp.complex64:
    euclidean_distance_squared = (x_2 - x_1)**2 + (y_2 - y_1)**2 + (z_2 - z_1)**2
    cosA = (z_1 - z_2) + 0.5 * z_2 * euclidean_distance_squared
    sinA = y_1 * x_2 - x_1 * y_2
    r = jnp.complex64(sinA + 1j * -cosA)  # Use jnp.complex128 to create a complex number
    return r

def real_and_imag(z) -> jnp.ndarray:
    return jnp.array([jnp.real(z), jnp.imag(z)])

def fuzzy_shear_estimator(galaxy1_coord: jnp.ndarray, galaxy2_coord: jnp.ndarray, galaxy_distance: float, galaxy1_quantities: jnp.ndarray, galaxy2_quantities: jnp.ndarray) -> float:
    ra1, dec1 = galaxy1_coord
    ra2, dec2 = galaxy2_coord
    x1, y1, z1 = jnp.cos(ra1 * jnp.pi / 180) * jnp.cos(dec1 * jnp.pi / 180), jnp.sin(ra1 * jnp.pi / 180) * jnp.cos(dec1 * jnp.pi / 180), jnp.sin(dec1 * jnp.pi / 180)
    x2, y2, z2 = jnp.cos(ra2 * jnp.pi / 180) * jnp.cos(dec2 * jnp.pi / 180), jnp.sin(ra2 * jnp.pi / 180) * jnp.cos(dec2 * jnp.pi / 180), jnp.sin(dec2 * jnp.pi / 180)

    r21 = calculate_direction(x2, x1, y2, y1, z2, z1)
    phi21 = jnp.real(jnp.conj(r21) * r21 / (jnp.abs(r21)**2 + 1e-6))

    r12 = calculate_direction(x1, x2, y1, y2, z1, z2)
    phi12 = jnp.real(jnp.conj(r12) * r12 / (jnp.abs(r12)**2 + 1e-6))

    object_one_shear_one = jnp.array([galaxy1_quantities[0], galaxy1_quantities[1]])
    object_one_shear_two = jnp.array([galaxy1_quantities[2], galaxy1_quantities[3]])
    object_two_shear_one = jnp.array([galaxy2_quantities[0], galaxy2_quantities[1]])
    object_two_shear_two = jnp.array([galaxy2_quantities[2], galaxy2_quantities[3]])

    object_one_shear_one = real_and_imag(-jnp.exp(2j * phi12) * (object_one_shear_one[0] + 1j * object_one_shear_one[1]))
    object_two_shear_two = real_and_imag(-jnp.exp(2j * phi21) * (object_two_shear_two[0] + 1j * object_two_shear_two[1]))
    object_one_shear_two = real_and_imag(-jnp.exp(2j * phi12) * (object_one_shear_two[0] + 1j * object_one_shear_two[1]))
    object_two_shear_one = real_and_imag(-jnp.exp(2j * phi21) * (object_two_shear_one[0] + 1j * object_two_shear_one[1]))

    return jnp.dot(object_one_shear_one, object_two_shear_two) + jnp.dot(object_one_shear_two, object_two_shear_one)

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

def sigmoid_weighting(lower_bound: float, upper_bound: float, distance: float, sharpness: Optional[float] = 10) -> float:
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
    fuzziness: Optional[float] = 5.0
    max_iter: Optional[int] = 1000
    tol: Optional[float] = 1e-6
    distance_metric: Optional[Callable[[jnp.ndarray, jnp.ndarray], float]] = vincenty_formula

class cosmic_correlator:
    def __init__(self, galaxies: List[galaxy], number_clusters: int, config: correlator_config):
        self.galaxies = galaxies
        self.number_clusters = number_clusters

        self.lower_bound = config.lower_bound
        self.upper_bound = config.upper_bound
        self.number_bins = config.number_bins
        self.bins = jnp.linspace(self.lower_bound, self.upper_bound, self.number_bins)

        self.sharpness = config.sharpness
        self.distance_metric = config.distance_metric
        self.m = config.fuzziness
        self.verbose = config.verbose
        self.max_iter = config.max_iter
        self.tol = config.tol

        self.Y = jnp.array([galaxy.coord for galaxy in galaxies])
        self.quantity_matrix = jnp.array([galaxy.quantities for galaxy in galaxies]).T
        self.U = random.uniform(random.PRNGKey(0), (self.number_clusters, self.Y.shape[0]))
        self.U = self.U / jnp.sum(self.U, axis=0)
        self.v = random.uniform(random.PRNGKey(0), (self.number_clusters, self.Y.shape[1]))
        self.fcm = fuzzy_c_means(self.number_clusters, self.m, self.Y, self.distance_metric)

        self.cluster_assignments = None


    def fcm_fit(self):
        self.fcm.fit(self.max_iter, self.tol)
        self.U = self.fcm.U
        self.v = self.fcm.v

    def _set_static_assignments(self):
        self.cluster_assignments = jnp.argmax(self.U, axis=0)
        self.U = jnp.zeros((self.number_clusters, self.Y.shape[0]))
        for i, cluster in enumerate(self.cluster_assignments):
            self.U = self.U.at[cluster, i].set(1)

    def _weight_quantities(self):
        q_tilde = jnp.dot(self.quantity_matrix, self.U.T)
        q_tilde = q_tilde / jnp.sum(self.U, axis=1)
        return q_tilde

    @jax.custom_jvp
    def correlate(self) -> jnp.ndarray:
        correlation = jnp.zeros((self.number_bins))
        total_weight = jnp.zeros((self.number_bins))
        self.fcm_fit()
        self._set_static_assignments()
        weighted_quantities = self._weight_quantities()
        new_galaxies = [galaxy(coord=self.v[i], quantities=weighted_quantities[i]) for i in range(self.number_clusters)]
        distances = [self.distance_metric(galaxy1.coord, galaxy2.coord) for galaxy1 in new_galaxies for galaxy2 in new_galaxies]
        for k in range(len(self.bins) - 1):
            lower_bound = self.bins[k]
            upper_bound = self.bins[k + 1]
            for i, galaxy1 in enumerate(new_galaxies):
                for j, galaxy2 in enumerate(new_galaxies):
                    def true_fn(correlation):
                        distance = vincenty_formula(galaxy1.coord, galaxy2.coord)
                        weight = sigmoid_weighting(lower_bound, upper_bound, distance, sharpness=self.sharpness)
                        total_weight = total_weight.at[k].set(total_weight[k] + weight)
                        shear_estimation = fuzzy_shear_estimator(galaxy1.coord, galaxy2.coord, distance, galaxy1.quantities, galaxy2.quantities)
                        return correlation.at[k].set(correlation[k] + weight * shear_estimation)
                
                    def false_fn(correlation):
                        return correlation 

                    correlation = lax.cond(i < j, true_fn, false_fn, correlation)
            correlation = jnp.where(total_weight != 0, correlation / total_weight, 0)

                
        return correlation

    @correlate.defjvp
    def correlate_jvp(primals, tangents):
        self, = primals
        tangent_self, = tangents
        primal_out = self.correlate()
        
        correlation = jnp.zeros((self.number_bins))
        total_weight = jnp.zeros((self.number_bins))
        weighted_quantities = self._weight_quantities()
        original_galaxies = self.galaxies
        new_galaxies = [galaxy(coord=self.v[i], quantities=weighted_quantities[i]) for i in range(self.number_clusters)]

        def correlate_binned_galaxies(self, new_galaxies):
            for i, galaxy1 in enumerate(new_galaxies):
                for j, galaxy2 in enumerate(new_galaxies):
                    def true_fn(correlation):
                        distance = vincenty_formula(galaxy1.coord, galaxy2.coord)
                        weight = sigmoid_weighting(self.lower_bound, self.upper_bound, distance, sharpness=self.sharpness)
                        total_weight = total_weight.at[k].set(total_weight[k] + weight)
                        shear_estimation = fuzzy_shear_estimator(galaxy1.coord, galaxy2.coord, distance, galaxy1.quantities, galaxy2.quantities)
                        return correlation.at[k].set(correlation[k] + weight * shear_estimation)
                    
                    def false_fn(correlation):
                        return correlation 

                    correlation = lax.cond(i < j, true_fn, false_fn, correlation)
            correlation = correlation / total_weight

        shear_grad = grad(correlate_binned_galaxies)(self, new_galaxies)

        for i, galaxy in enumerate(galaxies):
            cluster_assignment = jnp.argmax(self.U[:, i])
            tangent_out = shear_grad[cluster_assignment]

        return primal_out, tangent_out




config = correlator_config(lower_bound=0, upper_bound=200, sharpness=1, number_bins=2, verbose=True)
galaxy1 = galaxy(coord=jnp.array([0, 0]), quantities=jnp.array([1, 2, 3, 4]))
galaxy2 = galaxy(coord=jnp.array([0, 1]), quantities=jnp.array([1, 2, 3, 4]))
galaxy3 = galaxy(coord=jnp.array([1, 0]), quantities=jnp.array([1, 2, 3, 4]))
galaxy4 = galaxy(coord=jnp.array([1, 1]), quantities=jnp.array([1, 2, 3, 4]))
galaxy5 = galaxy(coord=jnp.array([2, 2]), quantities=jnp.array([1, 2, 3, 4]))
galaxy6 = galaxy(coord=jnp.array([2, 3]), quantities=jnp.array([1, 2, 3, 4]))
galaxy7 = galaxy(coord=jnp.array([3, 2]), quantities=jnp.array([1, 2, 3, 4]))
galaxy8 = galaxy(coord=jnp.array([3, 3]), quantities=jnp.array([1, 2, 3, 4]))
galaxy9 = galaxy(coord=jnp.array([4, 4]), quantities=jnp.array([1, 2, 3, 4]))
galaxy10 = galaxy(coord=jnp.array([4, 5]), quantities=jnp.array([1, 2, 3, 4]))

galaxies = [galaxy1, galaxy2, galaxy3, galaxy4, galaxy5, galaxy6, galaxy7, galaxy8, galaxy9, galaxy10]
correlator = cosmic_correlator(galaxies, 3, config)
correlator.fcm_fit()
print(correlator.U)
print(correlator.v)
print(correlator._weight_quantities())
print(correlator.correlate())

