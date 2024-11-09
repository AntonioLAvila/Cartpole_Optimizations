import os
import numpy as np
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    MeshcatVisualizer,
    Parser,
    PiecewisePolynomial,
    FiniteHorizonLinearQuadraticRegulatorOptions,
    MakeFiniteHorizonLinearQuadraticRegulator,
    Simulator,
    AddDefaultVisualization,
)
Q = np.diag([10,10,1,1])
R = np.eye(1)
def simulate_stabilization_FHLQR(meshcat, x_traj: PiecewisePolynomial, u_traj: PiecewisePolynomial):
    options = FiniteHorizonLinearQuadraticRegulatorOptions()
    options.x0 = x_traj
    options.u0 = u_traj
    options.Qf = Q

    this_builder = DiagramBuilder()
    this_plant, this_scene_graph = AddMultibodyPlantSceneGraph(this_builder, time_step=0.0)
    this_plant.SetUseSampledOutputPorts(False)

    Parser(this_plant).AddModelsFromUrl("file://"+os.getcwd()+"/cartpole.urdf")
    this_plant.Finalize()
    options.input_port_index = this_plant.get_actuation_input_port().get_index()

    regulator = this_builder.AddSystem(
        MakeFiniteHorizonLinearQuadraticRegulator(
            this_plant,
            this_plant.CreateDefaultContext(),
            t0=options.u0.start_time(),
            tf=options.u0.end_time(),
            Q=Q,
            R=R,
            options=options
        )
    )

    this_builder.Connect(regulator.get_output_port(0), this_plant.get_actuation_input_port())
    this_builder.Connect(this_plant.get_state_output_port(), regulator.get_input_port(0))

    AddDefaultVisualization(this_builder, meshcat=meshcat)
    diagram = this_builder.Build()
    simulator = Simulator(diagram)
    simulator.set_target_realtime_rate(1)
    meshcat.StartRecording()
    simulator.AdvanceTo(10)
    meshcat.PublishRecording()

def play_back_trajectory(meshcat, x_traj, t_traj):
    # create diagram
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    Parser(plant).AddModelsFromUrl("file://"+os.getcwd()+"/cartpole.urdf")
    plant.Finalize()
    plant.set_name("cartpole")
    visualizer = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
    visualizer.set_name("visualizer")
    diagram = builder.Build()

    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)
    x_traj = PiecewisePolynomial.FirstOrderHold(t_traj, x_traj)
    visualizer.StartRecording()
    for t in t_traj:
        context.SetTime(t)
        plant.SetPositions(plant_context, x_traj.value(t)[:2])
        diagram.ForcedPublish(context)
    visualizer.StopRecording()
    visualizer.PublishRecording()

def play_back_trajectory_polynomial(meshcat, x_traj: PiecewisePolynomial):
    # create diagram
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    Parser(plant).AddModelsFromUrl("file://"+os.getcwd()+"/cartpole.urdf")
    plant.Finalize()
    plant.set_name("cartpole")
    visualizer = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
    visualizer.set_name("visualizer")
    diagram = builder.Build()

    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)
    visualizer.StartRecording()
    t_traj = np.linspace(x_traj.start_time(), x_traj.end_time(), 100)
    for t in t_traj:
        context.SetTime(t)
        plant.SetPositions(plant_context, x_traj.value(t)[:2])
        diagram.ForcedPublish(context)
    visualizer.StopRecording()
    visualizer.PublishRecording()