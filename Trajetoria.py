import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
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


def main():
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