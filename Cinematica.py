from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import numpy as np
from dataclasses import dataclass
import seaborn as sns
import matplotlib.pyplot as plt


@dataclass
class DHModel:
    d:float
    a:float
    theta:float
    alpha:float

@dataclass
class UR5Params:
    d = (0.089159,0,0,0.10915,0.09465,0.0823 + 0.088)
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

def build_test(joints_params):
    dh_table = [
        DHModel(d=UR5Params.d[1],a=UR5Params.a[1],theta=joints_params[1], alpha=UR5Params.alpha[1]),
        DHModel(d=UR5Params.d[2],a=UR5Params.a[2],theta=joints_params[2], alpha=UR5Params.alpha[2]),
        DHModel(d=UR5Params.d[3],a=UR5Params.a[3],theta=joints_params[3], alpha=UR5Params.alpha[3]),
    ]
    model_T = np.eye(4)
    for row in dh_table:
        model_T = model_T @ build_dh_matrix(row)

    return model_T

def ur5_forward_kinematics(joints_params) -> np.ndarray:
    model_T = np.eye(4)
    dh_table = build_ur5_model(joints_params)

    for row in dh_table:
        model_T = model_T @ build_dh_matrix(row)

    return model_T

def ur5_inverse_kinematics(desired_t_matrix:np.ndarray,shoulder="left",wrist="up",elbow="up"):
    assert desired_t_matrix.shape == (4,4)
    # Notação p_i_ref_j.
    # Representa a origem do frame de referência i em relação ao sistema j.

    # Determinação de theta_1.
    p_5_ref_0 = desired_t_matrix @ np.array([0,0,-UR5Params.d[5],1]).T
    phi_1 = np.atan2(p_5_ref_0[1],p_5_ref_0[0])
    np.hypot(p_5_ref_0[0],p_5_ref_0[1])
    phi_2 = np.acos(UR5Params.d[3]/(np.hypot(p_5_ref_0[0],p_5_ref_0[1])))
    if shoulder == "left":
        theta_1 = phi_1 + phi_2 + np.pi/2
    else:
        theta_1 = phi_1 - phi_2 + np.pi/2

    # Determinação de theta_5.
    p_6_ref_0 = desired_t_matrix @ np.array([0,0,0,1]).T
    theta_5 = np.acos((p_6_ref_0[0] * np.sin(theta_1) - p_6_ref_0[1] * np.cos(theta_1) - UR5Params.d[3])/UR5Params.d[5])
    if wrist == "down":
        theta_5 = - theta_5

    # Determinação de theta_6.
    # Calcula os vetores x e y do sistema 0 em relação ao sistema 6.
    # Inverte a matriz T de 6 para 0.
    x_0_ref_6 = np.linalg.inv(desired_t_matrix) @ np.array([1,0,0,0]).T
    y_0_ref_6 = np.linalg.inv(desired_t_matrix) @ np.array([0,1,0,0]).T
    #TODO: Verificar o que fazer caso sin(theta_5) = 0.
    theta_6 = np.atan2(
        (-x_0_ref_6[1] * np.sin(theta_1) + y_0_ref_6[1] * np.cos(theta_1))/np.sin(theta_5),
        (x_0_ref_6[0] * np.sin(theta_1) - y_0_ref_6[0] * np.cos(theta_1))/np.sin(theta_5)
        )

    # Utilizei o material de Ryan Keating para determinar os próximos ângulos.
    # O material de Andersen utiliza a convenção DH modificada.

    # Determinação de theta_3.
    # Determinação das matrizes de transformação.
    #t_m_to_n é o equivalente a T_{m}^{n}
    t_5_to_4 = build_dh_matrix(
        DHModel(d=UR5Params.d[4],a=UR5Params.a[4],theta=theta_5, alpha=UR5Params.alpha[4])
        )
    t_6_to_5 = build_dh_matrix(
        DHModel(d=UR5Params.d[5],a=UR5Params.a[5],theta=theta_6, alpha=UR5Params.alpha[5])
        )
    t_1_to_0 = build_dh_matrix(
    DHModel(d=UR5Params.d[0],a=UR5Params.a[0],theta=theta_1, alpha=UR5Params.alpha[0])
        )
    t_6_to_1 = (np.linalg.inv(t_1_to_0) @ desired_t_matrix)
    t_6_to_4 = t_5_to_4 @ t_6_to_5
    # Transforma de 4 para 6, de 6 para 0 e de 0 para 1.
    t_4_to_1 = t_6_to_1 @ np.linalg.inv(t_6_to_4)
    p_3_ref_1 = (t_4_to_1 @ np.array([0,-UR5Params.d[3],0,1])) - [0,0,0,1]
    p_3norm = np.linalg.norm(p_3_ref_1)
    cos_theta_3 = (p_3norm**2 - UR5Params.a[1]**2 - UR5Params.a[2]**2)/(2 * UR5Params.a[1] * UR5Params.a[2])
    theta_3 = np.acos(cos_theta_3)
    if elbow == "down":
        theta_3 = - theta_3

    # Determinação de theta_2.
    theta_2 = -np.atan2(p_3_ref_1[1],-p_3_ref_1[0]) + np.asin(UR5Params.a[2] * np.sin(theta_3)/(p_3norm))

    # Determinação de theta_4.
    t_3_to_2 = build_dh_matrix(
    DHModel(d=UR5Params.d[2],a=UR5Params.a[2],theta=theta_3, alpha=UR5Params.alpha[2])
        )
    t_2_to_1 = build_dh_matrix(
    DHModel(d=UR5Params.d[1],a=UR5Params.a[1],theta=theta_2, alpha=UR5Params.alpha[1])
        )
    t_3_to_1 = t_2_to_1 @ t_3_to_2
    t_4_to_3 = np.linalg.inv(t_3_to_1) @ t_4_to_1
    x_4_ref_3 = t_4_to_3 @ np.array([1,0,0,0])
    theta_4 = np.atan2(x_4_ref_3[1],x_4_ref_3[0])
    print(f"Soma {np.rad2deg(theta_2+theta_3+theta_4)}")
    return [theta_1,theta_2, theta_3, theta_4, theta_5, theta_6]


def tmatrix_to_angles(transform_matrix:np.ndarray):
    assert transform_matrix.shape == (4,4)
    rot_matrix = transform_matrix[:3, :3].copy()
    
    # Threshold para verificar singularidade
    tolerance = 1e-6
    
    if abs(rot_matrix[0, 2]) < (1.0 - tolerance):
        # Caso não singular (sin(beta) != 0)
        beta = np.asin(rot_matrix[0, 2])

        # alpha = atan2(-r23, r33)
        alpha = np.atan2(-rot_matrix[1, 2], rot_matrix[2, 2])
        # gamma = atan2(-r12, r11)
        gamma = np.atan2(-rot_matrix[0, 1], rot_matrix[0, 0])
    else:
        # Caso singular
        alpha = 0.0
        gamma = np.atan2(rot_matrix[1, 0], rot_matrix[1, 1])  # usando r21 e r22        
        # beta pode ser +/- pi/2
        beta = np.asin(rot_matrix[0, 2])
    
    return np.array([alpha,beta,gamma])

def set_joints_position(joints_num:list[int],joints_params):
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
 
def get_orientation_error(sim_matrix:np.ndarray,model_matrix:np.ndarray) -> np.ndarray:
    assert sim_matrix.shape == (4,4)
    assert model_matrix.shape == (4,4)
    sim_angles = tmatrix_to_angles(sim_matrix)
    model_angles = tmatrix_to_angles(model_matrix)
    orientation_error = np.abs(sim_angles-model_angles) 
    return orientation_error


def get_Tmatrix(target_handle:int,ref_handle:int):
    sim_T = sim.getObjectMatrix(target_handle,ref_handle)
    sim_T = np.reshape(sim_T,(3,4))
    sim_T = np.concatenate((sim_T,[[0,0,0,1]]),axis=0)
    return sim_T


def set_Tmatrix(desired_matrix,target_handle:int,ref_handle:int):
    sim_matrix = desired_matrix[:3, :4].copy()
    sim_matrix = sim_matrix.flatten().tolist()
    sim.setObjectMatrix(target_handle, sim_matrix, ref_handle)


def validate_fk(joints_handles:list[int],target_handle:int,base_handle:int,num_iter=200):
    position_error_list = []
    orientation_error_list = []
    joints_params_history:dict = {"q1": [], "q2": [], "q3": [], "q4": [], "q5": [], "q6": []}
    np.random.seed(15)
    for iter in range(num_iter):
        joints_params = np.random.uniform(-1,1,6) * np.pi
        deg_params = np.rad2deg(joints_params)
        joints_params_history["q1"].append(deg_params[0])
        joints_params_history["q2"].append(deg_params[1])
        joints_params_history["q3"].append(deg_params[2])
        joints_params_history["q4"].append(deg_params[3])
        joints_params_history["q5"].append(deg_params[4])
        joints_params_history["q6"].append(deg_params[5])
        set_joints_position(joints_handles,joints_params)
        sim_T = get_Tmatrix(target_handle,base_handle)
        model_T = ur5_forward_kinematics(joints_params)
        position_error_vec = get_position_error(sim_T,model_T)
        orientation_error_vec = get_orientation_error(sim_T,model_T)
        position_error_list.append(np.linalg.norm(position_error_vec))
        orientation_error_list.append(np.linalg.norm(orientation_error_vec))
    return position_error_list, orientation_error_list, joints_params_history

def validate_ik(num_iter=200):
    position_error_list = []
    orientation_error_list = []
    ref_params_history:dict = {"q1": [], "q2": [], "q3": [], "q4": [], "q5": [], "q6": []}
    found_params_history:dict = {"q1": [], "q2": [], "q3": [], "q4": [], "q5": [], "q6": []}
    np.random.seed(15)
    for iter in range(num_iter):
        ref_joints_params = np.random.uniform(-1,1,6) * np.pi
        desired_T = ur5_forward_kinematics(ref_joints_params)
        #TODO: Verificar as 8 possíveis configurações para o UR5.
        found_joints_params = ur5_inverse_kinematics(desired_T)

client = RemoteAPIClient()
sim = client.require("sim")

def main_model():
    joints_params = np.deg2rad([140, -40, 78, 20, 45, 40])
    model_T = ur5_forward_kinematics(joints_params)
    print("Model Matrix")
    print(model_T.round(5))
    desired_q_params = ur5_inverse_kinematics(model_T)
    ik_model_T = ur5_forward_kinematics(desired_q_params)
    print("Joints params values:")
    print(np.rad2deg(desired_q_params))
    print("Transformation Matrix with IK joints values.")
    print(ik_model_T.round(5))

def main():
    sim.setStepping(True)
    sim.startSimulation()
    base_handle = sim.getObject('/UR5/frame0')
    target_handle = sim.getObject('/UR5/ROBOTIQ85/attachPoint')
    joints_paths: list[str] = [f"/UR5/joint{i}" for i in range(1,7)]
    joints_handles = get_joints_handlers(joints_paths)
    joints_params = [0,0,np.pi/2,np.pi/2,0,np.pi/6]
    joints_params = np.random.uniform(-1,1,6) * np.pi
    print("Desired Joints Params.")
    print(joints_params)
    set_joints_position(joints_handles,joints_params)
    sim_T = get_Tmatrix(target_handle,base_handle)
    model_T = ur5_forward_kinematics(joints_params)
    sim_joints_params = get_joints_position(joints_handles)
    print("Sim Joints Params.")
    print(sim_joints_params)

    print("Model Matrix 2")
    print(model_T.round(5))
    print("Simulation Matrix")
    print(sim_T.round(5))
    sim_angles = tmatrix_to_angles(sim_T)
    print(f"Simulation angles: {sim_angles}")
    model_angles = tmatrix_to_angles(model_T)
    print(f"Simulation angles: {model_angles}")
    sim.stopSimulation()

def run_fk_validation():
    sim.setStepping(True)
    sim.startSimulation()
    base_handle = sim.getObject('/UR5/frame0')
    target_handle = sim.getObject('/UR5/ROBOTIQ85/attachPoint')
    joints_paths: list[str] = [f"/UR5/joint{i}" for i in range(1,7)]
    joints_handles = get_joints_handlers(joints_paths)
    position_error_list, orientation_error_list, joints_params_history = validate_fk(joints_handles,target_handle,base_handle)
    # Plotagem dos resultados.
    fig, axes = plt.subplots(2,1,figsize=(13, 9))
    mean_position_error = np.mean(position_error_list) * np.ones_like(position_error_list)
    mean_orientation_error = np.mean(orientation_error_list) * np.ones_like(orientation_error_list)
    sns.lineplot(position_error_list,label="Erro",color="black",ax=axes[0])
    sns.lineplot(mean_position_error,label=f"Média: {np.mean(position_error_list):.6f}",color="red",ax=axes[0])
    axes[0].set(xlabel="Iteração", ylabel="Erro de posição (m)",title="Histórico de Erro vs Iteração")
    sns.lineplot(orientation_error_list,label="Erro",color="black",ax=axes[1])
    sns.lineplot(mean_orientation_error,label=f"Média: {np.mean(orientation_error_list):.5e}",color="red",ax=axes[1])
    axes[1].set(xlabel="Iteração", ylabel="Erro de orientação (rad)")
    plt.tight_layout()
    fig.savefig("erro_fk.pdf", bbox_inches="tight")
    plt.show()
    # Plotagem das variáveis das juntas.
    fig, axes = plt.subplots(6, 1, figsize=(16, 16), sharex=True)
    for ax, joint_name in zip(axes, [f"q{i}" for i in range(1, 7)]):
        sns.lineplot(
            x=range(len(joints_params_history[joint_name])),
            y=joints_params_history[joint_name],
            label=joint_name,
            ax=ax,
            color="black"
        )
        ax.set_ylabel("Valor da junta (graus)")
        ax.set_title(f"Histórico de {joint_name}")
        ax.legend()
    axes[-1].set_xlabel("Iteração")
    plt.tight_layout()
    fig.savefig("juntas_fk.pdf", bbox_inches="tight")
    plt.show()
    sim.stopSimulation()



if __name__ == '__main__':
    run_fk_validation()