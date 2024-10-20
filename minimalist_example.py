import diffrax as dfx
import jax
import jax.numpy as jnp

def f(t, y, args):
    c = args[0]
    return -c * y

term = dfx.ODETerm(f)
solver = dfx.Dopri5()
y0 = jnp.array([2., 3.])

t0, t1 = 0, 1
dt0 = 0.1

def solve_ode(c):
    solution = dfx.diffeqsolve(term, solver, t0=t0, t1=t1, dt0=dt0, y0=y0, args=(c,))
    return solution.ys

c_initial = 1.0  # Initial value for the constant
grad_fn = jax.grad(lambda c: jnp.sum(solve_ode(c)))  # Sum the solution to get a scalar
gradient = grad_fn(c_initial)

print("Gradient of the solution with respect to the constant c:", gradient)

def gravitational_ode_3d(t, state, args):
    pos_x, pos_y, pos_z, vel_x, vel_y, vel_z = state
    G, M = args
    r = jnp.sqrt(pos_x**2 + pos_y**2 + pos_z**2)
    ax = -G * M * pos_x / r**3
    ay = -G * M * pos_y / r**3
    az = -G * M * pos_z / r**3
    return jnp.array([vel_x, vel_y, vel_z, ax, ay, az])

term = dfx.ODETerm(gravitational_ode_3d)
solver = dfx.Dopri5()
y0 = jnp.array([1., 0., 0., 0., 0.5, 0.])

t0, t1 = 0, 1
dt0 = 0.1

G = 1.0
M = 1.0
args = (G, M)

grad_fn = jax.grad(lambda G: jnp.sum(dfx.diffeqsolve(term,
                                                        solver,
                                                        t0=t0,
                                                        t1=t1,
                                                        dt0=dt0,
                                                        y0=y0,
                                                        args=(G, M)).ys))
gradient = grad_fn(G)
print("Gradient of the solution with respect to the constant G:", gradient)

