import numpy as np
import os
import time
from utils import play_back_trajectory_polynomial, simulate_stabilization_FHLQR
from functools import partial
from pydrake.all import (
    RobotDiagramBuilder,
    Parser,
    eq,
    ExtractGradient,
    MathematicalProgram,
    AutoDiffXd,
    SnoptSolver,
    PiecewisePolynomial,
    StartMeshcat
)

# TODO: stabalize the trajectory with own algorithm

urdf = "file://"+os.getcwd()+"/cartpole.urdf"
solver = SnoptSolver()
meshcat = StartMeshcat()

builder = RobotDiagramBuilder(time_step=1e-4)
plant = builder.plant()
Parser(plant).AddModelsFromUrl(urdf)
plant.Finalize()
plant.set_name("cartpole")
ad_plant = plant.ToAutoDiffXd()

x0 = [0,0,0,0]
xf = [0,np.pi,0,0]
f_max = 10
N = 100
B = np.array([[1],[0]])

def autoDiffArrayEqual(a, b):
    return np.array_equal(a, b) and np.array_equal(ExtractGradient(a), ExtractGradient(b))

def dynamics_constraint(vars, context):
    q_qd, qdd, u = np.split(vars, [4,4+2])
    qd = q_qd[2:]
    if isinstance(vars[0], AutoDiffXd):
        if not autoDiffArrayEqual(q_qd, ad_plant.GetPositionsAndVelocities(context)):
            ad_plant.SetPositionsAndVelocities(context, q_qd)
        M = ad_plant.CalcMassMatrix(context)
        C = ad_plant.CalcBiasTerm(context)
        tau_g = ad_plant.CalcGravityGeneralizedForces(context)
    else:
        if not np.array_equal(q_qd, plant.GetPositionsAndVelocities(context)):
            plant.SetPositionsAndVelocities(context, q_qd)
        M = plant.CalcMassMatrix(context)
        C = plant.CalcBiasTerm(context)
        tau_g = plant.CalcGravityGeneralizedForces(context)
    return tau_g + B@u - M@qdd - C@qd

def run_opt():
    prog = MathematicalProgram()

    contexts = [ad_plant.CreateDefaultContext() for _ in range(N)]
    h = prog.NewContinuousVariables(N-1, 'h')
    u = prog.NewContinuousVariables(1, N-1, 'u')
    q = prog.NewContinuousVariables(2, N, 'q')
    qd = prog.NewContinuousVariables(2, N, 'qd')
    qdd = prog.NewContinuousVariables(2, N-1, 'qdd')

    Q = np.diag([10,10,1,1])    # costs
    R = np.eye(1)

    prog.AddBoundingBoxConstraint(0.01, 0.1, h) # time step

    prog.AddBoundingBoxConstraint(x0[:2], x0[:2], q[:, 0])  # initial conditions
    prog.AddBoundingBoxConstraint(x0[2:], x0[2:], qd[:, 0])
    prog.AddBoundingBoxConstraint(0, 0, qdd[:,0])

    prog.AddBoundingBoxConstraint(xf[:2], xf[:2], q[:, -1])  # final conditions
    prog.AddBoundingBoxConstraint(xf[2:], xf[2:], qd[:, -1])

    prog.AddBoundingBoxConstraint(-f_max, f_max, u) # force limit

    x = np.concatenate((q[:,N-1], qd[:,N-1]))   # final cost
    prog.AddCost(x.T@Q@x)

    for t in range(N-1):
        # satisfy manipulator equation
        prog.AddConstraint(
            partial(dynamics_constraint, context=contexts[t]),
            lb=[0,0],
            ub=[0,0],
            vars=np.concatenate((q[:,t], qd[:,t], qdd[:,t], u[:,t]))
        )
        # forward euler
        prog.AddConstraint(eq(qd[:, t+1], qd[:, t] + h[t]*qdd[:,t]))
        prog.AddConstraint(eq(q[:, t+1], q[:, t] + h[t]*qd[:,t]))

        x = np.concatenate((q[:,t], qd[:,t]))
        prog.AddCost(x.T@Q@x + u[:,t].T@R@u[:,t])

    # solve
    print("Solving")
    start = time.time()
    result = solver.Solve(prog)
    print(result.is_success())
    print("Time to solve:", time.time() - start)

    t_traj = np.cumsum(np.concatenate(([0], result.GetSolution(h))))
    x_traj = PiecewisePolynomial.FirstOrderHold(t_traj, np.vstack((result.GetSolution(q), result.GetSolution(qd))))
    u_traj = PiecewisePolynomial.FirstOrderHold(t_traj, result.GetSolution(q))
    return x_traj, u_traj

if __name__ == "__main__":
    x_traj, u_traj = run_opt()
    play_back_trajectory_polynomial(meshcat, x_traj)
    # simulate_stabilization_FHLQR(meshcat, x_traj, u_traj)
    while True:
        pass