import diffrax as dfx
import jax
import jax.numpy as jnp

# Define the function with an additional parameter (constant `c`)
def f(t, y, args):
    c = args[0]
    return -c * y

# Create the ODE term and solver
term = dfx.ODETerm(f)
solver = dfx.Dopri5()
y0 = jnp.array([2., 3.])

# Define the initial time and final time
t0, t1 = 0, 1
dt0 = 0.1

# Set up a function that takes `c` as input and solves the ODE
def solve_ode(c):
    # Call diffeqsolve with `c` as the argument
    solution = dfx.diffeqsolve(term, solver, t0=t0, t1=t1, dt0=dt0, y0=y0, args=(c,))
    return solution.ys

# Take the gradient of the solution with respect to `c`
c_initial = 1.0  # Initial value for the constant
grad_fn = jax.grad(lambda c: jnp.sum(solve_ode(c)))  # Sum the solution to get a scalar
gradient = grad_fn(c_initial)

print("Gradient of the solution with respect to the constant c:", gradient)

