import jax.numpy as jnp
import diffrax
import jax
from jax import jvp
from jax import debug
from jax import grad, jit, vmap, jacfwd, jacrev, lax
import jax.random as random
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass
from functools import partial
import ipdb

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
    v_init = random.uniform(random.PRNGKey(0), (U.shape[0], Y.shape[1]))
    v = v_init
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

def gaussian_weighting(lower_bound: float, upper_bound: float, distance: float, sharpness: float = 1.0) -> float:
    center = (lower_bound + upper_bound) / 2.0  
    normalized_distance = (distance - center) / (upper_bound - lower_bound)  
    gaussian = jnp.exp(-sharpness * (normalized_distance ** 2))  
    return gaussian

@jax.custom_jvp
def correlate_fuzzy_c_means(U, Y, quantities, m, lower_bound, upper_bound, sharpness, number_bins) -> jnp.ndarray:
    U, v = fit(U, Y, m)
    c, N = U.shape
    new_centers = v 
    new_quantities = jnp.dot(quantities.T, U)
    correlation = 0.0
    total_weight = 0.0

    for i in range(c):
        for j in range(c):
            if i < j:
                distance = vincenty_formula(new_centers[i], new_centers[j])
                total_weight += sigmoid_weighting(lower_bound, upper_bound, distance, sharpness)
                correlation += sigmoid_weighting(lower_bound, upper_bound, distance, sharpness) * fuzzy_shear_estimator(new_centers[i], new_centers[j], new_quantities[i], new_quantities[j])
                
    return correlation / total_weight

@correlate_fuzzy_c_means.defjvp
def correlate_fuzzy_c_means_jvp(primals, tangents) -> Tuple[jnp.ndarray, jnp.ndarray]:
    U, Y, quantities, m, lower_bound, upper_bound, sharpness, number_bins = primals
    dU, dY, dquantities, dm, dlower_bound, dupper_bound, dsharpness, dnumber_bins = tangents
    U, v = fit(U, Y, m)
    c, N = U.shape
    new_centers = v 
    new_quantities = (jnp.dot(quantities, U.T) / jnp.sum(U, axis=1)).T
    quantities = quantities.T

    gradients = []
    dcorrelation_dY = jnp.zeros_like(Y) 
    dcorrelation_d_quantity = jnp.zeros_like(quantities)  

    def weighted_correlation(new_center_i, new_center_j, new_quantity_i, new_quantity_j):
        distance = vincenty_formula(new_center_i, new_center_j)
        weight = gaussian_weighting(lower_bound, upper_bound, distance, sharpness)
        shear_estimate = fuzzy_shear_estimator(
            new_center_i, new_center_j, new_quantity_i, new_quantity_j
        )
        return weight * shear_estimate

    for i in range(dcorrelation_dY.shape[0]):
        cluster_assignment = jnp.argmax(U[:, i])
        for j in range(c):
            for k in range(c):
                if j < k and j == cluster_assignment:
                    gradients = jax.grad(weighted_correlation, argnums=(0, 1, 2, 3))(
                        new_centers[j], new_centers[k], new_quantities[j], new_quantities[k]
                    )
                    gradients = tuple(jnp.nan_to_num(grad, nan=0.0) for grad in gradients)
                    dcorrelation_dY = dcorrelation_dY.at[i, :].add(gradients[0])
                    dcorrelation_d_quantity = dcorrelation_d_quantity.at[i, :].add(gradients[2])
                    gradients = jax.grad(weighted_correlation, argnums=(0, 1, 2, 3))(
                        new_centers[k], new_centers[j], new_quantities[k], new_quantities[j]
                    )
                    gradients = tuple(jnp.nan_to_num(grad, nan=0.0) for grad in gradients)
                    dcorrelation_dY = dcorrelation_dY.at[i, :].add(gradients[0])
                    dcorrelation_d_quantity = dcorrelation_d_quantity.at[i, :].add(gradients[2])


    primals_out = correlate_fuzzy_c_means(U, Y, v, m, lower_bound, upper_bound, sharpness, number_bins)
    tangent_out = jnp.sum(jnp.multiply(dcorrelation_dY, dY)) + jnp.sum(jnp.multiply(dcorrelation_d_quantity, dquantities.T + jnp.ones_like(dquantities.T)))
    
    return primals_out, tangent_out


def gradient_correlate_fuzzy_c_means(U, Y, quantities, m, lower_bound, upper_bound, sharpness, number_bins) -> jnp.ndarray:
    U, v = fit(U, Y, m)
    c, N = U.shape
    new_centers = v 
    new_quantities = jnp.dot(quantities, U.T)
    return

galaxy1 = galaxy(coord=jnp.array([0.0, 0.0]), quantities=jnp.array([1.0, 2.0, 3.0, 4.0]))
galaxy2 = galaxy(coord=jnp.array([0.0, 1.0]), quantities=jnp.array([1.0, 2.0, 3.0, 4.0]))
galaxy3 = galaxy(coord=jnp.array([1.0, 0.0]), quantities=jnp.array([1.0, 2.0, 3.0, 4.0]))
galaxy4 = galaxy(coord=jnp.array([1.0, 1.0]), quantities=jnp.array([1.0, 2.0, 3.0, 4.0]))
galaxy5 = galaxy(coord=jnp.array([2.0, 2.0]), quantities=jnp.array([1.0, 2.0, 3.0, 4.0]))
galaxy6 = galaxy(coord=jnp.array([2.0, 3.0]), quantities=jnp.array([1.0, 2.0, 3.0, 4.0]))
galaxy7 = galaxy(coord=jnp.array([3.0, 2.0]), quantities=jnp.array([1.0, 2.0, 3.0, 4.0]))
galaxy8 = galaxy(coord=jnp.array([3.0, 3.0]), quantities=jnp.array([1.0, 2.0, 3.0, 4.0]))
galaxy9 = galaxy(coord=jnp.array([4.0, 4.0]), quantities=jnp.array([1.0, 2.0, 3.0, 4.0]))
galaxy10 = galaxy(coord=jnp.array([4.0, 5.0]), quantities=jnp.array([1.0, 2.0, 3.0, 4.0]))

galaxies = [galaxy1, galaxy2, galaxy3, galaxy4, galaxy5, galaxy6, galaxy7, galaxy8, galaxy9, galaxy10]
galaxies_coords = jnp.array([galaxy.coord for galaxy in galaxies])
galaxies_quantities = jnp.array([galaxy.quantities for galaxy in galaxies]).T


U_init = random.uniform(random.PRNGKey(0), (3, 10))
U_init = U_init / jnp.sum(U_init, axis=0)
grad_correlation = grad(correlate_fuzzy_c_means, argnums=(1, 2), allow_int=True)(U_init, galaxies_coords, galaxies_quantities, 1.5, 0, 200, 1.0, 3)

def gravitational_ode_3d(t, state, args):
    pos_x, pos_y, pos_z, vel_x, vel_y, vel_z = state
    G, M = args
    r = jnp.sqrt(pos_x**2 + pos_y**2 + pos_z**2)
    ax = -G * M * pos_x / r**3
    ay = -G * M * pos_y / r**3
    az = -G * M * pos_z / r**3
    return jnp.array([vel_x, vel_y, vel_z, ax, ay, az])

G = 1.0  
M = 1.0  
R = 1.0  

pos_x = R
pos_y = 0.0
pos_z = 0.0

vel_mag = jnp.sqrt(G * M / R)
vel_x = 0.0
vel_y = vel_mag
vel_z = 0.0

state0 = jnp.array([pos_x, pos_y, pos_z, vel_x, vel_y, vel_z])

t0 = 0.0
t1 = 2 * jnp.pi * jnp.sqrt(R**3 / (G * M)) 

args = (G, M)
term = diffrax.ODETerm(gravitational_ode_3d)
solver = diffrax.Dopri5()
t_eval = jnp.linspace(t0, t1, 1000)

solution = diffrax.diffeqsolve(
    term,
    solver,
    t0=t0,
    t1=t1,
    dt0=0.1,
    y0=state0,
    args=args,
    saveat=diffrax.SaveAt(ts=t_eval)
)




print(grad_correlation)

