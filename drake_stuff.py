import numpy as np
import os
import time
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    Parser,
    StartMeshcat,
    SnoptSolver,
    BasicVector,
    DirectTranscription,
)
from utils import play_back_trajectory_polynomial, simulate_stabilization_FHLQR

# State = [x, th, xd, thd]
urdf = "file://"+os.getcwd()+"/cartpole.urdf"

# only use for not built-in traj. opt
builder = DiagramBuilder()
plant, _ = AddMultibodyPlantSceneGraph(builder, time_step=1e-4)
Parser(plant).AddModelsFromUrl(urdf)
plant.Finalize()
plant.set_name("cartpole")

solver = SnoptSolver()
N = 100
x0 = [0,0,0,0]
xf = [0,np.pi,0,0]
f_max = 10
R = np.diag([1])
Q = np.diag([10, 10, 1, 1])


def swingup_ct(x0, xf):
    builder = DiagramBuilder()
    ct_plant, ct_scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0)
    Parser(ct_plant).AddModelsFromUrl(urdf)
    ct_plant.Finalize()
    ct_plant.set_name("ct_cartpole")
    dirtran = DirectTranscription(
        ct_plant,
        ct_plant.CreateDefaultContext(),
        num_time_samples=N,
        input_port_index=ct_plant.get_actuation_input_port().get_index(),
        fixed_time_step=DirectTranscription.TimeStep(0.05)
    )
    prog = dirtran.prog()

    u = dirtran.input()
    x = dirtran.state()
    dirtran.AddConstraintToAllKnotPoints(u[0] >= -f_max)
    dirtran.AddConstraintToAllKnotPoints(u[0] <= f_max)

    initial_state = BasicVector(x0)
    prog.AddBoundingBoxConstraint(initial_state.get_value(), initial_state.get_value(), dirtran.initial_state())
    final_state = BasicVector(xf)
    prog.AddBoundingBoxConstraint(final_state.get_value(), final_state.get_value(), dirtran.final_state())

    dirtran.AddRunningCost(x.T@Q@x + u.T@R@u)

    start = time.time()
    result = solver.Solve(prog)
    print(f"Solve time: {time.time()-start}")
    print(result.is_success())

    return dirtran.ReconstructStateTrajectory(result), dirtran.ReconstructInputTrajectory(result)

def swingup_dt(x0, xf):
    builder = DiagramBuilder()
    dt_plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.05)
    dt_plant.SetUseSampledOutputPorts(False)
    Parser(dt_plant).AddModelsFromUrl(urdf)
    dt_plant.Finalize()
    dt_plant.set_name("dt_cartpole")

    dirtran = DirectTranscription(
        dt_plant,
        dt_plant.CreateDefaultContext(),
        num_time_samples=N,
        input_port_index=dt_plant.get_actuation_input_port().get_index()
    )
    prog = dirtran.prog()

    u = dirtran.input()
    x = dirtran.state()
    dirtran.AddConstraintToAllKnotPoints(u[0] >= -f_max)
    dirtran.AddConstraintToAllKnotPoints(u[0] <= f_max)

    initial_state = BasicVector(x0)
    prog.AddBoundingBoxConstraint(initial_state.get_value(), initial_state.get_value(), dirtran.initial_state())
    final_state = BasicVector(xf)
    prog.AddBoundingBoxConstraint(final_state.get_value(), final_state.get_value(), dirtran.final_state())

    dirtran.AddRunningCost(x.T@Q@x + u.T@R@u)

    start = time.time()
    result = solver.Solve(prog)
    print(f"Solve time: {time.time()-start}")
    print(result.is_success())

    return dirtran.ReconstructStateTrajectory(result), dirtran.ReconstructInputTrajectory(result)

if __name__ == "__main__":
    meshcat = StartMeshcat()
    # x_traj, u_traj = swingup_dt(x0, xf)
    x_traj, u_traj = swingup_ct(x0, xf)
    # play_back_trajectory_polynomial(meshcat, x_traj)
    simulate_stabilization_FHLQR(meshcat, x_traj, u_traj)
    while True:
        pass