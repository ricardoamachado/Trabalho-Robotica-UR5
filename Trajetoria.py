from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from Cinematica import ur5_inverse_kinematics, set_joints_position, get_joints_position, get_joints_handles, change_grip_state, get_Tmatrix, set_joints_position_dynamics

client = RemoteAPIClient()
sim = client.require("sim")

def cubic_path_coefs(initial_position,initial_velocity,final_position,final_velocity,initial_time=0,final_time=1):
    matrix = np.array(
    [
        [1, initial_time, initial_time**2, initial_time**3],
        [0, 1, 2 * initial_time, 3 * initial_time**2],
        [1, final_time, final_time**2, final_time**3],
        [0, 1, 2 * final_time, 3 * final_time**2],
    ]
    )
    cubic_coefs = np.linalg.inv(matrix) @ np.array([initial_position,initial_velocity,final_position,final_velocity])
    return cubic_coefs

def quintic_path_coefs(initial_position,initial_velocity,initial_accel,final_position,final_velocity,final_accel,initial_time=0,final_time=1):
    matrix = np.array(
    [
        [1, initial_time, initial_time**2, initial_time**3, initial_time**4, initial_time**5],
        [0, 1, 2 * initial_time, 3 * initial_time**2, 4 * initial_time**3, 5 * initial_time**4],
        [0, 0, 2, 6 * initial_time, 12 * initial_time**2, 20 * initial_time**3],
        [1, final_time, final_time**2, final_time**3, final_time**4, final_time**5],
        [0, 1, 2 * final_time, 3 * final_time**2, 4 * final_time**3, 5 * final_time**4],
        [0, 0, 2, 6 * final_time, 12 * final_time**2, 20 * final_time**3],
    ]
    )
    quintic_coefs = np.linalg.inv(matrix) @ np.array([initial_position,initial_velocity,initial_accel,final_position,final_velocity,final_accel])
    return quintic_coefs

def quintic_joints_position_at_time(
    initial_joints_position,
    final_joints_position,
    current_time,
    initial_time=0,
    final_time=1,
    initial_velocity=0,
    final_velocity=0,
    initial_accel=0,
    final_accel=0,):

    initial_joints_position = np.asarray(initial_joints_position, dtype=float)
    final_joints_position = np.asarray(final_joints_position, dtype=float)
    # Velocidades iniciais e finais e acelarações iniciais e finais são iguais para todas as juntas.
    initial_velocity = np.broadcast_to(np.asarray(initial_velocity, dtype=float), (6,))
    final_velocity = np.broadcast_to(np.asarray(final_velocity, dtype=float), (6,))
    initial_accel = np.broadcast_to(np.asarray(initial_accel, dtype=float), (6,))
    final_accel = np.broadcast_to(np.asarray(final_accel, dtype=float), (6,))

    t = np.clip(current_time, initial_time, final_time)
    joints_position = np.zeros(6)

    for i in range(6):
        quintic_coefs = quintic_path_coefs(
            initial_joints_position[i],
            initial_velocity[i],
            initial_accel[i],
            final_joints_position[i],
            final_velocity[i],
            final_accel[i],
            initial_time,
            final_time,
        )
        joints_position[i] = (
            quintic_coefs[0]
            + quintic_coefs[1] * t
            + quintic_coefs[2] * t**2
            + quintic_coefs[3] * t**3
            + quintic_coefs[4] * t**4
            + quintic_coefs[5] * t**5
        )

    return joints_position

def move_to_joints_position(joints_handles,desired_position,delta_time,initial_velocity=0,final_velocity=0,initial_accel=0,final_accel=0,tol=1e-4):
    joints_initial_position = get_joints_position(joints_handles)
    initial_time = sim.getSimulationTime()
    final_time = initial_time + delta_time
    while not sim.getSimulationStopping():
        current_time = sim.getSimulationTime()
        current_joints_posision = quintic_joints_position_at_time(joints_initial_position,desired_position,current_time,initial_time,final_time)
        set_joints_position_dynamics(joints_handles,current_joints_posision)
        sim.step()
        if np.linalg.norm(current_joints_posision - desired_position) < tol:
            break

def move_to_pose(joints_handles,desired_matrix,delta_time,initial_velocity=0,final_velocity=0,initial_accel=0,final_accel=0,tol=1e-4):
    joints_initial_position = get_joints_position(joints_handles)
    # Itera sobre as 8 possíveis configurações do UR5.
    ik_configurations = [
        (shoulder, wrist, elbow)
        for shoulder in ("left", "right")
        for wrist in ("up", "down")
        for elbow in ("up", "down")
    ]
    delta_norm = np.inf
    for shoulder, wrist, elbow in ik_configurations:
        candidate_joints_params = ur5_inverse_kinematics(
            desired_matrix,
            shoulder=shoulder,
            wrist=wrist,
            elbow=elbow,
        )
        # Variação entre a posição inicial e final das juntas.
        delta_joints_position = np.abs(joints_initial_position - candidate_joints_params)
        if np.linalg.norm(delta_joints_position) < delta_norm:
            delta_norm =  np.linalg.norm(delta_joints_position)
            smallest_delta_position = candidate_joints_params
    print(smallest_delta_position)
    initial_time = sim.getSimulationTime()
    final_time = initial_time + delta_time
    while not sim.getSimulationStopping():
        current_time = sim.getSimulationTime()
        current_joints_posision = quintic_joints_position_at_time(joints_initial_position,smallest_delta_position,current_time,initial_time,final_time)
        set_joints_position_dynamics(joints_handles,current_joints_posision)
        sim.step()
        if np.linalg.norm(current_joints_posision - smallest_delta_position) < tol:
            break


def main():
    joints_paths: list[str] = [f"/UR5/joint{i}" for i in range(1,7)]
    joints_handles = get_joints_handles(joints_paths)
    object_handle = sim.getObject('/projector')
    base_handle = sim.getObject('/UR5/frame0')
    desired_matrix = get_Tmatrix(object_handle,base_handle)
    delta_time = 20
    initial_position = np.array([0,-np.pi/2,0,np.pi/2,0,0])
    sim.setStepping(True)
    sim.startSimulation()
    move_to_joints_position(joints_handles,initial_position,delta_time)
    move_to_pose(joints_handles,desired_matrix,delta_time)

def example_plot():
    initial_position = 0
    initial_velocity = 0
    final_velocity = 0
    final_position = 20
    initial_accel = 0
    final_accel = 0
    cubic_coefs = cubic_path_coefs(initial_position,initial_velocity,final_position,final_velocity)
    quintic_coefs = quintic_path_coefs(initial_position,initial_velocity,initial_accel,final_position,final_velocity,final_accel)
    t = np.linspace(0,1,100)
    cubic_position_eq = cubic_coefs[0] + cubic_coefs[1] * t + cubic_coefs[2] * t**2 + cubic_coefs[3] * t**3
    cubic_velocity_eq = cubic_coefs[1] + 2 * cubic_coefs[2] * t + 3 * cubic_coefs[3] * t**2
    quintic_position_eq = quintic_coefs[0] + quintic_coefs[1] * t + quintic_coefs[2] * t**2 + quintic_coefs[3] * t**3 + quintic_coefs[4] * t**4 + quintic_coefs[5] * t**5
    quintic_velocity_eq = quintic_coefs[1] * t + 2 * quintic_coefs[2] + 3 * quintic_coefs[3] * t**2 + 4 * quintic_coefs[4] * t**3 + 5 * quintic_coefs[5] * t**4
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.lineplot(x=t,y=cubic_position_eq,ax=axes[0])
    sns.lineplot(x=t,y=cubic_velocity_eq,ax=axes[1])
    plt.show()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.lineplot(x=t,y=quintic_position_eq,ax=axes[0])
    sns.lineplot(x=t,y=quintic_velocity_eq,ax=axes[1])
    plt.show()
if __name__ == '__main__':
    main()