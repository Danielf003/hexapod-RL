import numpy as np

from Traj_thetas import thetas_traj

def param_traj(T_f, T_b, L, alfa, delta_thetas):


    
    #-----------------------------------------------------------

    # T_b - время движения по параболе в фазе опоры (если T_b = 0, то энд-эффектор движется просто по прямой)

    step_length = L
    rotation_angle = alfa

    # Предварительные расчеты для фазы опоры и 1/2 фазы перемещения
    A = np.array([
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, T_f**2, T_f, 1],
        [T_b**2, T_b, 1, -T_b, -1, 0, 0, 0],
        [1/2*T_b, 1, 0, -1, 0, 0, 0, 0],
        [0, 0, 0, -(T_f-T_b), -1, (T_f-T_b)**2, (T_f-T_b), 1],
        [0, 0, 0, -1, 0, 1/2*(T_f-T_b), 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1/2*T_f, 1, 0]
    ])

    B_x = np.array([1.5 + step_length/2 * np.sin(rotation_angle), 1.5 - step_length/2 * np.sin(rotation_angle), 0, 0, 0, 0, 0, 0])

    y_cor = step_length/2 * np.cos(rotation_angle)
    B_y = np.array([-y_cor, y_cor, 0, 0, 0, 0, 0, 0])

    # координата z отсчитывается от СК, связанной с корпусом (0,56 - высота подъёма корпуса над землёй)
    # Думаю, можно добавить этот параметр в перечень входных параметров функции (добавить этот выход НС)
    B_z = np.array([-0.56, -0.56, 0, 0, 0, 0, 0, 0])

    # Решаем систему уравнений
    C_x = np.linalg.lstsq(A, B_x, rcond=None)[0]
    C_y = np.linalg.lstsq(A, B_y, rcond=None)[0]
    C_z = np.linalg.lstsq(A, B_z, rcond=None)[0]

    #-----------------------------------------------------------

    # Предварительные расчеты для 1/2 фазы перемещения

    # Решение обратной задачи кинематики
    # # Входные параметры
    # x = 1.5
    # y = 0
    # z = H

    # # # Решение ОЗК

    # th = IK(x, y, z, l1, l2, l3, l4)
    # th[2] = -th[2]
    # th = np.array([0, 0.25,-0.2,0])

    T = np.array([
        [T_f**4, T_f**3, T_f**2, T_f, 1],
        [(3/2*T_f)**4, (3/2*T_f)**3, (3/2*T_f)**2, (3/2*T_f), 1],
        [(2*T_f)**4, (2*T_f)**3, (2*T_f)**2, (2*T_f), 1],
        [4*T_f**3, 3*T_f**2, 2*T_f, 1, 0],
        [4*(2*T_f)**3, 3*(2*T_f)**2, 2*(2*T_f), 1, 0]
    ])

    a = np.zeros((4, 5))  # параметры уравнений для 4 обобщенных координат
    # Уравнения вида: (q = a_4*t^4 + a_3*t^3 + a_2*t^2 + a_1*t + a_0)
    for idx in range(4):
        delta_theta = delta_thetas[idx]
        D_theta = np.array([0, delta_theta, 0, 0, 0])
        a[idx, :] = np.linalg.inv(T).dot(D_theta)

    #-----------------------------------------------------------

    return C_x, C_y, C_z, a

# T_f, T_b, L, alfa, delta_thetas =  50, 2, 1, 0.44, [0.17, 0.25, 0, 0.1]
# C_x, C_y, C_z, a = param_traj(T_f, T_b, L, alfa, delta_thetas)
# t=10
# traject = thetas_traj(t, T_f, T_b, T_f, C_x, C_y, C_z, a, J_inv_func=None, dJ_dt_func=None)
# print(traject)