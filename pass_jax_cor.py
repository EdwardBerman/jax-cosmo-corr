import jax.numpy as jnp
import jax
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

def fuzzy_shear_estimator(galaxy1_coord: jnp.ndarray, galaxy2_coord: jnp.ndarray, galaxy1_quantities: jnp.ndarray, galaxy2_quantities: jnp.ndarray) -> float:
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

def fit(U, Y, m, max_iter: int = 1000, tol: float = 1e-6):
    for _ in range(max_iter):
        new_centers = update_centers(U, Y, m)
        new_membership = update_membership(Y, new_centers, m, vincenty_formula)
        if jnp.sum(jnp.abs(new_membership - U)) < tol:
            break
        U = new_membership
        v = new_centers
    return U, v

def sigmoid_weighting(lower_bound: float, upper_bound: float, distance: float, sharpness: float) -> float:
    sigmoid_lower_bound = 1 / (1 + jnp.exp(-sharpness * (distance - lower_bound)) + 1e-6)
    sigmoid_upper_bound = 1 / (1 + jnp.exp(-sharpness * (upper_bound - distance)) + 1e-6)
    return sigmoid_lower_bound * sigmoid_upper_bound

@jax.custom_vjp
def correlate_fuzzy_c_means(galaxies_coords: jnp.ndarray, galaxies_quantities: jnp.ndarray, number_clusters: int, 
                            fuzziness: float, lower_bound: float, upper_bound: float, sharpness: float, 
                            number_bins: int, max_iter: Optional[int] = 1000, tol: Optional[float] = 1e-6) -> jnp.ndarray:
    N = galaxies_coords.shape[0]
    U = random.uniform(random.PRNGKey(0), (number_clusters, N))
    U = U / jnp.sum(U, axis=0)

    U, centers = fit(U, galaxies_coords, fuzziness, max_iter, tol)

    weighted_quantities = jnp.dot(galaxies_quantities, U.T) / (jnp.sum(U, axis=1) + 1e-6)
    bins = jnp.linspace(lower_bound, upper_bound, number_bins)

    correlation = jnp.zeros(number_bins - 1)
    distances = jnp.zeros(number_bins - 1)
    total_weight = jnp.zeros(number_bins - 1)
    
    for k in range(bins.shape[0] - 1):
        lower_bound = bins[k]
        upper_bound = bins[k + 1]
        
        for i in range(number_clusters):
            for j in range(i+1, number_clusters):
                distance = vincenty_formula(centers[i], centers[j])
                weight = sigmoid_weighting(lower_bound, upper_bound, distance, sharpness)
                
                total_weight = total_weight.at[k].set(total_weight[k] + weight)
                
                shear_estimation = fuzzy_shear_estimator(centers[i], centers[j], weighted_quantities[i], weighted_quantities[j])
                correlation = correlation.at[k].set(correlation[k] + weight * shear_estimation)

        distances = distances.at[k].set((lower_bound + upper_bound) / 2)
    
    correlation = jnp.where(total_weight != 0, correlation / total_weight, 0)
    
    return correlation

def correlate_fuzzy_c_means_fwd(galaxies_coords: jnp.ndarray, galaxies_quantities: jnp.ndarray, number_clusters: int, 
                            fuzziness: float, lower_bound: float, upper_bound: float, sharpness: float, 
                            number_bins: int, distance_metric: Optional[Callable] = vincenty_formula,
                            max_iter: Optional[int] = 1000, tol: Optional[float] = 1e-6) -> Tuple[jnp.ndarray, Tuple]:
    result = correlate_fuzzy_c_means(galaxies_coords, galaxies_quantities, number_clusters, fuzziness, lower_bound, upper_bound, sharpness, number_bins, distance_metric, max_iter, tol)
    return result, (galaxies_coords, galaxies_quantities, number_clusters, fuzziness, lower_bound, upper_bound, sharpness, number_bins, distance_metric, max_iter, tol)

def correlate_fuzzy_c_means_bwd(res, grad):
    galaxies_coords, galaxies_quantities, number_clusters, fuzziness, lower_bound, upper_bound, sharpness, number_bins, distance_metric, max_iter, tol = res
    N = galaxies_coords.shape[0]
    U = random.uniform(random.PRNGKey(0), (number_clusters, N))
    U = U / jnp.sum(U, axis=0)
    
    for _ in range(max_iter):
        centers = update_centers(U, galaxies_coords, fuzziness)
        new_U = update_membership(galaxies_coords, centers, fuzziness, distance_metric)
        if jnp.sum(jnp.abs(U - new_U)) < tol:
            break
        U = new_U

    U_new = jnp.zeros((number_clusters, N))
    for i in range(N):
        cluster_assignment = jnp.argmax(U[:, i])
        U_new = U_new.at[cluster_assignment, i].set(1)
        
    weighted_quantities = jnp.dot(U, galaxies_quantities.T) / (jnp.sum(U, axis=1) + 1e-6)
    bins = jnp.linspace(lower_bound, upper_bound, number_bins)
    correlation = jnp.zeros(number_bins - 1)
    total_weight = jnp.zeros(number_bins - 1)

    def correlate_centers(centers, weighted_quantities, bins, correlation, total_weight, sharpness, lower_bound, upper_bound, number_clusters):
        bins = jax.lax.stop_gradient(bins)
        correlation = jax.lax.stop_gradient(correlation)
        sharpness = jax.lax.stop_gradient(sharpness)
        lower_bound = jax.lax.stop_gradient(lower_bound)
        upper_bound = jax.lax.stop_gradient(upper_bound)
        number_clusters = jax.lax.stop_gradient(number_clusters)
        for k in range(bins.shape[0] - 1):
            lower_bound = bins[k]
            upper_bound = bins[k + 1]

            for i in range(number_clusters):
                for j in range(i+1, number_clusters):
                    distance = vincenty_formula(centers[i], centers[j])
                    weight = sigmoid_weighting(lower_bound, upper_bound, distance, sharpness)

                    total_weight = total_weight.at[k].set(total_weight[k] + weight)

                    shear_estimation = fuzzy_shear_estimator(centers[i], centers[j], weighted_quantities[i], weighted_quantities[j])
                    correlation = correlation.at[k].set(correlation[k] + weight * shear_estimation)

        correlation = jnp.where(total_weight != 0, correlation / total_weight, 0)
        return correlation

    grad_correlation = grad(correlate_centers, argnums=(0, 1))(centers, weighted_quantities, bins, correlation, total_weight, sharpness, lower_bound, upper_bound, distance_metric, number_clusters)

    return (grad_correlation[0], grad_correlation[1], None, None, None, None, None, None, None, None, None)

correlate_fuzzy_c_means.defvjp(correlate_fuzzy_c_means_fwd, correlate_fuzzy_c_means_bwd)


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
galaxies_coords = jnp.array([galaxy.coord for galaxy in galaxies])
galaxies_quantities = jnp.array([galaxy.quantities for galaxy in galaxies]).T

#correlation = correlate_fuzzy_c_means(galaxies_coords, galaxies_quantities, number_clusters=2, fuzziness=1.5, lower_bound=0, upper_bound=10, sharpness=1, number_bins=10)
#grad_correlation = grad(correlate_fuzzy_c_means, argnums=(0, 1), allow_int=True)(galaxies_coords, galaxies_quantities, number_clusters=2, fuzziness=1.5, lower_bound=0, upper_bound=10, sharpness=1, number_bins=10)
#config = correlator_config(lower_bound=0, upper_bound=200, sharpness=1, number_bins=2, verbose=True)
jac_correlation = jacrev(correlate_fuzzy_c_means, argnums=(0, 1), allow_int=True)(galaxies_coords, galaxies_quantities, number_clusters=2, fuzziness=1.5, lower_bound=0, upper_bound=200, sharpness=0.005, number_bins=2)
#print(correlation)
print(jac_correlation)

