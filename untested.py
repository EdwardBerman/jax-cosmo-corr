import jax
import jax.numpy as jnp
from jax import grad, jacrev, vmap
import diffrax
from functools import partial

# ----------------------------------------------------------------------
# 1) Simple 3D gravitational ODE (Kepler-like)
# ----------------------------------------------------------------------
def gravitational_ode_3d(t, state, args):
    """
    state = (pos_x, pos_y, pos_z, vel_x, vel_y, vel_z)
    args  = (G, M)
    Returns d(state)/dt
    """
    pos_x, pos_y, pos_z, vel_x, vel_y, vel_z = state
    G, M = args
    # small epsilon to avoid dividing by zero
    r = jnp.sqrt(pos_x**2 + pos_y**2 + pos_z**2) + 1e-12
    ax = -G * M * pos_x / r**3
    ay = -G * M * pos_y / r**3
    az = -G * M * pos_z / r**3
    return jnp.array([vel_x, vel_y, vel_z, ax, ay, az])

def xyz_to_ra_dec(x, y, z):
    """
    Convert cartesian (x,y,z) -> (RA, Dec) in degrees.
    RA: [-180, 180], Dec: [-90, 90].
    """
    r = jnp.sqrt(x**2 + y**2 + z**2) + 1e-12
    dec = jnp.arcsin(z / r) * (180.0 / jnp.pi)
    ra = jnp.arctan2(y, x) * (180.0 / jnp.pi)
    return jnp.array([ra, dec])

def generate_end_positions_for_given_G(G, initial_states, t_eval, M=1.0):
    """
    Solve the system for each object's initial_states, from t0->t1,
    then convert final (x,y,z) -> (RA, Dec).
    """
    term = diffrax.ODETerm(gravitational_ode_3d)
    solver = diffrax.Dopri5()

    t0 = t_eval[0]
    t1 = t_eval[-1]
    dt0 = 0.1

    # By default, diffrax uses a custom_vjp-based adjoint for ODEs,
    # which does *not* allow forward-mode autodiff.
    # However, we only need reverse-mode autodiff here, so it's fine.
    # We'll keep the default adjacency. If you ever *must* do forward-mode,
    # you could specify adjoint=diffrax.DirectAdjoint(), but that can be slower.
    @partial(jax.jit, static_argnums=())
    def single_solve(y0):
        sol = diffrax.diffeqsolve(
            term,
            solver,
            t0=t0,
            t1=t1,
            dt0=dt0,
            y0=y0,
            args=(G, M),
            saveat=diffrax.SaveAt(ts=t_eval),
        )
        final_xyz = sol.ys[-1, :3]
        return xyz_to_ra_dec(final_xyz[0], final_xyz[1], final_xyz[2])

    return vmap(single_solve)(initial_states)


# ----------------------------------------------------------------------
# 2) Final RA/Dec vs G (10 objects -> shape (10, 2))
# ----------------------------------------------------------------------
def final_ra_dec_vs_G(G, initial_states, t_eval):
    """
    Returns shape (10,2) array of final (RA, Dec) for each object,
    after integrating the ODE with gravitational constant G.
    """
    return generate_end_positions_for_given_G(G, initial_states, t_eval)


# ----------------------------------------------------------------------
# 3) Dummy correlation function that depends on Y=(RA,Dec)
# ----------------------------------------------------------------------
def correlate_fuzzy_c_means(U, Y, quantities, m, lower_bound, upper_bound, sharpness, number_bins):
    """
    Returns a scalar correlation that depends on Y (shape Nx2).
    For demonstration, we sum pairwise distances, plus some quantity term.
    """
    def euclidean_distance(p1, p2):
        return jnp.sqrt(jnp.sum((p1 - p2)**2))

    N = Y.shape[0]
    total_dist = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            total_dist += euclidean_distance(Y[i], Y[j])

    # Just add something from 'quantities' to ensure it's used
    quantity_term = jnp.sum(quantities)
    return total_dist + 0.01 * quantity_term

def correlation_wrt_positions(Y):
    """
    Wrap the correlation so we can do grad w.r.t. Y.
    Y: shape (N, 2)
    """
    # Suppose we fix membership U, fuzziness, etc.
    c = 3
    N = Y.shape[0]
    U_init = jax.random.uniform(jax.random.PRNGKey(0), shape=(c, N))
    U_init = U_init / jnp.sum(U_init, axis=0, keepdims=True)

    # Suppose shear quantities: shape (4, N)
    quantities = jnp.ones((4, N)) * 0.001

    # Dummy parameters
    m = 1.5
    lower_bound = 0.0
    upper_bound = 54.0
    sharpness = 1.0
    number_bins = 1

    return correlate_fuzzy_c_means(U_init, Y, quantities, m, lower_bound, upper_bound, sharpness, number_bins)


# ----------------------------------------------------------------------
# MAIN SCRIPT
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Example: 10 objects in near-circular orbits
    # --------------------------------------------
    G_true = 1.0
    M = 1.0
    R = 1.0
    num_objects = 10

    # Random angles from 0..2π
    angles = jax.random.uniform(jax.random.PRNGKey(42), shape=(num_objects,),
                                minval=0, maxval=2*jnp.pi)

    # Positions
    pos_x = R * jnp.cos(angles)
    pos_y = R * jnp.sin(angles)
    pos_z = jnp.zeros(num_objects)

    # Speeds for near-circular orbits
    vel_mag = jnp.sqrt(G_true * M / R)
    vel_x = -vel_mag * jnp.sin(angles)
    vel_y =  vel_mag * jnp.cos(angles)
    vel_z = jnp.zeros(num_objects)

    # Pack into (10,6)
    initial_states = jnp.column_stack((pos_x, pos_y, pos_z, vel_x, vel_y, vel_z))

    # Time array from t=0 to one full orbit
    T_orbit = 2 * jnp.pi * jnp.sqrt(R**3 / (G_true * M))
    t_eval = jnp.linspace(0.0, T_orbit, 200)

    # 1) Get final (RA, Dec) with G=1.0
    # -----------------------------------
    final_positions = final_ra_dec_vs_G(G_true, initial_states, t_eval)
    print("\nFinal (RA, Dec) with G=1.0, shape:", final_positions.shape)
    print(final_positions)

    # 2) Jacobian of (RA,Dec) wrt G using jacrev
    #    shape => (10,2) because the output is (10,2)
    # -------------------------------------------
    def ra_dec_given_G(G):
        return final_ra_dec_vs_G(G, initial_states, t_eval)

    # Use jacrev for reverse-mode autodiff
    jac_fn = jacrev(ra_dec_given_G)
    d_ra_dec_dG = jac_fn(G_true)
    print("\nJacobian d(RA,Dec)/dG at G=1.0, shape:", d_ra_dec_dG.shape)
    print(d_ra_dec_dG)

    # 3) Correlation gradient w.r.t. positions (10,2)
    # -----------------------------------------------
    # correlation_wrt_positions returns a scalar
    d_correlation_dY_fn = grad(correlation_wrt_positions)
    grad_corr = d_correlation_dY_fn(final_positions)  # shape (10,2)
    print("\nGradient of correlation wrt final positions, shape:", grad_corr.shape)
    print(grad_corr)
