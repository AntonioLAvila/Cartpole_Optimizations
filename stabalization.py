import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
import cvxpy as cp
from cartpole_dyanmics import BetterLinearizedCartpole
from utils import play_back_trajectory, play_back_trajectory_polynomial
from pydrake.all import StartMeshcat, PiecewisePolynomial
import time

# system
cartpole = BetterLinearizedCartpole()
A = cartpole.A_zoh
B = cartpole.B_zoh
f = cartpole.f
dt = cartpole.dt

# costs/constraints
Q = np.diag([10, 10, 1, 1])
R = np.eye(1)
Qf = np.diag([100, 100, 10, 10])
f_max = 10

# mpc variables, state = [x, th, xd, thd]
x0 = np.array([0, pi-0.3, 0, 0])
x_star = np.array([0, pi, 0, 0])
xf = x_star
N = 20
x = cp.Variable((4, N+1))
u = cp.Variable((1, N))
x_init = cp.Parameter(4)

# sim variables
n_sim = 70
t_traj = [0]
x_traj = [x0]
u_traj = []

def linear_MPC():
    cost = 0
    constraints = [x[:,0] == x_init]    # initial constraint
    for t in range(N):
        constraints += [cp.norm(u[:,t], 'inf') <= f_max]    # force constraint
        constraints += [x[:,t+1] - x_star == A@(x[:,t] - x_star) + B@u[:,t]]    # dynamics constraint (x̄ = Ax̄ + Bū)
        cost += cp.quad_form(xf - x[:,t], Q) + cp.quad_form(u[:,t], R)  # quadratic costs
    cost += cp.quad_form(xf - x[:,N], Qf)    # final state cost
    prob = cp.Problem(cp.Minimize(cost), constraints)
    return prob

def plot(u_traj, x_traj, t_traj):
    plt.figure(1)
    plt.plot(t_traj, x_traj[1,:])
    plt.plot(t_traj, [np.pi for _ in t_traj], linestyle=':')
    plt.xlabel("Time (s)")
    plt.ylabel("Theta (radians)")
    plt.title("Angle of Pendulum")

    plt.figure(2)
    plt.plot(t_traj[:-1], u_traj[0,:])
    plt.xlabel("Time (s)")
    plt.ylabel("Input Force (Newtons)")
    plt.title("Force on Cart")

    # TODO: Why's it so high
    # TODO: Find the actual limits analytically
    error = [np.zeros(4)]
    overall_error = [0]
    for i in range(n_sim):
        non_linear = f(x_traj[:,i], u_traj[:,i])
        state_error = np.abs(non_linear - x_traj[:,i+1])
        error.append(state_error)
        overall_error.append(np.sqrt(state_error.T@state_error))
    error = np.array(error).T
    cum_error = np.cumsum(error, axis=1)
    overall_error = np.array(overall_error)
    cum_overall_error = np.cumsum(overall_error)
    plt.figure(3)
    plt.plot(t_traj, cum_error[2,:])
    plt.xlabel("Time (s)")
    plt.ylabel("Theta Error (radians)")
    plt.title("Linearization Error")

if __name__ == "__main__":
    meshcat = StartMeshcat()
    for i in range(1, n_sim+1):
        prob = linear_MPC()
        x_init.value = x0
        if i == 1: start = time.time()
        prob.solve(solver=cp.OSQP, warm_start=True)
        if i == 1: print(f'Solve time: {time.time() - start}')
        x0 = A@(x0 - x_star) + B@u[:,0].value + x_star
        t_traj.append(i*dt)
        x_traj.append(x0)
        u_traj.append(u[:,0].value)
    u_traj = np.array(u_traj).T
    x_traj = np.array(x_traj).T
    t_traj = np.array(t_traj)

    x_traj_poly = PiecewisePolynomial.FirstOrderHold(t_traj, x_traj)
    play_back_trajectory_polynomial(meshcat, x_traj_poly)

    plot(u_traj, x_traj, t_traj)

    plt.show()
