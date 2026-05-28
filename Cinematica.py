from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import numpy as np
from dataclasses import dataclass

@dataclass
class DHModel:
    d:float
    a:float
    theta:float
    alpha:float

@dataclass
class UR5Params:
    d = (0.089159,0,0,0.10915,0.09465,0.0823)
    a = (0,-0.425,-0.39225,0,0,0)
    alpha = (np.pi/2,0,0,np.pi/2,-np.pi/2,0)

def build_dh_matrix(params:DHModel):
    c_t = np.cos(params.theta)
    s_t = np.sin(params.theta)
    c_a = np.cos(params.alpha)
    s_a = np.sin(params.alpha)

    matriz_dh = np.array(
        [
            [c_t, -s_t * c_a, s_t * s_a, params.a * c_t],
            [s_t, c_t * c_a, -c_t*s_a, params.a * s_t],
            [0, s_a, c_a, params.d],
            [0, 0, 0, 1],
        ]
    )
    return matriz_dh

def build_ur5_model(joints_params) -> list[DHModel]:
    dh_table = [
    DHModel(d=UR5Params.d[0],a=UR5Params.a[0],theta=joints_params[0], alpha=UR5Params.alpha[0]),
    DHModel(d=UR5Params.d[1],a=UR5Params.a[1],theta=joints_params[1], alpha=UR5Params.alpha[1]),
    DHModel(d=UR5Params.d[2],a=UR5Params.a[2],theta=joints_params[2], alpha=UR5Params.alpha[2]),
    DHModel(d=UR5Params.d[3],a=UR5Params.a[3],theta=joints_params[3], alpha=UR5Params.alpha[3]),
    DHModel(d=UR5Params.d[4],a=UR5Params.a[4],theta=joints_params[4], alpha=UR5Params.alpha[4]),
    DHModel(d=UR5Params.d[5],a=UR5Params.a[5],theta=joints_params[5], alpha=UR5Params.alpha[5])
]
    return dh_table

def build_forward_kinematics(tabela_dh:list[DHModel]):
    matriz_transform = np.eye(4)
    for linha in tabela_dh:
        matriz_transform = matriz_transform @ build_dh_matrix(linha)
    return matriz_transform

def ur5_forward_kinematics(joints_params) -> np.ndarray:
    model_T = np.eye(4)
    dh_table = build_ur5_model(joints_params)

    for row in dh_table:
        model_T = model_T @ build_dh_matrix(row)
    return model_T

def set_joints_position(joints_num:list[int],joints_params:list[float]):
    for joint, param in zip(joints_num,joints_params):
        sim.setJointPosition(joint, param)

def get_joints_position(joints_handles:list[int]):
    joints_params:list[float] = []
    for joint in joints_handles:
        current_joint_position = sim.getJointPosition(joint)
        joints_params.append(current_joint_position)
    return joints_params

def get_joints_handlers(joints_paths:list[str]) -> list[int]:
    joints_handlers:list[int] = []
    for path in joints_paths:
        current_handle = sim.getObject(path)
        joints_handlers.append(current_handle)
    return joints_handlers

def get_position_error(sim_matrix:np.ndarray,model_matrix:np.ndarray) -> np.ndarray:
    assert sim_matrix.shape == (4,4)
    assert model_matrix.shape == (4,4)
    error_vector = np.abs(sim_matrix - model_matrix)
    error_vector = error_vector [:3,3]
    return error_vector
 
def get_Tmatrix(joints_handles:list[int],ref_handle:int):
    sim_T = sim.getObjectMatrix(joints_handles[-1],ref_handle)
    sim_T = np.reshape(sim_T,(3,4))
    sim_T = np.concatenate((sim_T,[[0,0,0,1]]),axis=0)
    return sim_T


client = RemoteAPIClient()
sim = client.require("sim")

def main():


    sim.setStepping(True)
    sim.startSimulation()
    base_handle = sim.getObject('/UR5')
    joints_paths: list[str] = [f"/UR5/joint{i}" for i in range(1,7)]
    joints_handles = get_joints_handlers(joints_paths)
    joints_params = [0,0,0,0,0,0]

    set_joints_position(joints_handles,joints_params)
    sim_T = get_Tmatrix(joints_handles,base_handle)
    model_T = ur5_forward_kinematics(joints_params)
    print("Model Matrix 2")
    print(model_T.round(5))
    print("Simulation Matrix")
    print(sim_T.round(5))

    sim.stopSimulation()

if __name__ == '__main__':
    main()