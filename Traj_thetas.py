import numpy as np
from IK import IK

def thetas_traj(t, T_f, T_b, L, alfa, H, delta_T):

    
    #-----------------------------------------------------------

    # T_b - время движения по параболе в фазе опоры (если T_b = 0, то энд-эффектор движется просто по прямой)

    l1 = 0.3  # m length of the first link
    l2 = 0.848   # m length of the second link
    l3 = 1.221   # m length of the third link
    l4 = 0.6  # m length of the fourth link

    step_length = L
    rotation_angle = alfa

    time = (t + delta_T) % (2 * T_f)

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
    # Входные параметры
    x = 1.5
    y = 0
    z = H

    # # Решение ОЗК

    th = IK(x, y, z, l1, l2, l3, l4)
    th[2] = -th[2]
    th = np.array([0, 0.25,-0.2,0])

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
        delta_theta = th[idx]
        D_theta = np.array([0, delta_theta, 0, 0, 0])
        a[idx, :] = np.linalg.inv(T).dot(D_theta)

    #-----------------------------------------------------------
    # Построение траектории

    if time <= T_f:  # Фаза опоры thet_S

        if 0 <= time <= T_b:
            p_x_s = np.array([time**2, time, 1, 0, 0, 0, 0, 0]).dot(C_x)
            p_y_s = np.array([time**2, time, 1, 0, 0, 0, 0, 0]).dot(C_y)
            p_z_s = np.array([time**2, time, 1, 0, 0, 0, 0, 0]).dot(C_z)

            v_x_s = np.array([2*time, 1, 0, 0, 0, 0, 0, 0]).dot(C_x)
            v_y_s = np.array([2*time, 1, 0, 0, 0, 0, 0, 0]).dot(C_y)
            v_z_s = np.array([2*time, 1, 0, 0, 0, 0, 0, 0]).dot(C_z)

            a_x_s = np.array([2, 0, 0, 0, 0, 0, 0, 0]).dot(C_x)
            a_y_s = np.array([2, 0, 0, 0, 0, 0, 0, 0]).dot(C_y)
            a_z_s = np.array([2, 0, 0, 0, 0, 0, 0, 0]).dot(C_z)

        elif T_b < time <= T_f - T_b:

            p_x_s = np.array([0, 0, 0, time, 1, 0, 0, 0]).dot(C_x)
            p_y_s = np.array([0, 0, 0, time, 1, 0, 0, 0]).dot(C_y)
            p_z_s = np.array([0, 0, 0, time, 1, 0, 0, 0]).dot(C_z)

            v_x_s = np.array([0, 0, 0, 1, 0, 0, 0, 0]).dot(C_x)
            v_y_s = np.array([0, 0, 0, 1, 0, 0, 0, 0]).dot(C_y)
            v_z_s = np.array([0, 0, 0, 1, 0, 0, 0, 0]).dot(C_z)

            a_x_s = np.array([0, 0, 0, 0, 0, 0, 0, 0]).dot(C_x)
            a_y_s = np.array([0, 0, 0, 0, 0, 0, 0, 0]).dot(C_y)
            a_z_s = np.array([0, 0, 0, 0, 0, 0, 0, 0]).dot(C_z)

        elif time > T_f - T_b:
            p_x_s = np.array([0, 0, 0, 0, 0, time**2, time, 1]).dot(C_x)
            p_y_s = np.array([0, 0, 0, 0, 0, time**2, time, 1]).dot(C_y)
            p_z_s = np.array([0, 0, 0, 0, 0, time**2, time, 1]).dot(C_z)

            v_x_s = np.array([0, 0, 0, 0, 0, 2*time, 1, 0]).dot(C_x)
            v_y_s = np.array([0, 0, 0, 0, 0, 2*time, 1, 0]).dot(C_y)
            v_z_s = np.array([0, 0, 0, 0, 0, 2*time, 1, 0]).dot(C_z)

            a_x_s = np.array([0, 0, 0, 0, 0, 2, 0, 0]).dot(C_x)
            a_y_s = np.array([0, 0, 0, 0, 0, 2, 0, 0]).dot(C_y)
            a_z_s = np.array([0, 0, 0, 0, 0, 2, 0, 0]).dot(C_z)

        # Решение обратной задачи кинематики

        q = IK(p_x_s, p_y_s, p_z_s, l1, l2, l3, l4)
        q1 = q[0]
        q2 = q[1]
        q3 = q[2]
        q4 = q[3]

        # Якобиан
        J_inv = np.array([
            [-np.sin(q1) / (l1 * np.sin(q1)**2 + l1 * np.cos(q1)**2 + l2 * np.cos(q1)**2 * np.cos(q2) + l2 * np.cos(q2) * np.sin(q1)**2 + l3 * np.cos(q1)**2 * np.cos(q2) * np.cos(q3) + l3 * np.cos(q2) * np.cos(q3) * np.sin(q1)**2 - l3 * np.cos(q1)**2 * np.sin(q2) * np.sin(q3) - l3 * np.sin(q1)**2 * np.sin(q2) * np.sin(q3) + l4 * np.cos(q1)**2 * np.cos(q2) * np.cos(q3) * np.cos(q4) + l4 * np.cos(q2) * np.cos(q3) * np.cos(q4) * np.sin(q1)**2 - l4 * np.cos(q1)**2 * np.cos(q2) * np.sin(q3) * np.sin(q4) - l4 * np.cos(q1)**2 * np.cos(q3) * np.sin(q2) * np.sin(q4) - l4 * np.cos(q1)**2 * np.cos(q4) * np.sin(q2) * np.sin(q3) - l4 * np.cos(q2) * np.sin(q1)**2 * np.sin(q3) * np.sin(q4) - l4 * np.cos(q3) * np.sin(q1)**2 * np.sin(q2) * np.sin(q4) - l4 * np.cos(q4) * np.sin(q1)**2 * np.sin(q2) * np.sin(q3)), np.cos(q1) / (l1 * np.sin(q1)**2 + l1 * np.cos(q1)**2 + l2 * np.cos(q1)**2 * np.cos(q2) + l2 * np.cos(q2) * np.sin(q1)**2 + l3 * np.cos(q1)**2 * np.cos(q2) * np.cos(q3) + l3 * np.cos(q2) * np.cos(q3) * np.sin(q1)**2 - l3 * np.cos(q1)**2 * np.sin(q2) * np.sin(q3) - l3 * np.sin(q1)**2 * np.sin(q2) * np.sin(q3) + l4 * np.cos(q1)**2 * np.cos(q2) * np.cos(q3) * np.cos(q4) + l4 * np.cos(q2) * np.cos(q3) * np.cos(q4) * np.sin(q1)**2 - l4 * np.cos(q1)**2 * np.cos(q2) * np.sin(q3) * np.sin(q4) - l4 * np.cos(q1)**2 * np.cos(q3) * np.sin(q2) * np.sin(q4) - l4 * np.cos(q1)**2 * np.cos(q4) * np.sin(q2) * np.sin(q3) - l4 * np.cos(q2) * np.sin(q1)**2 * np.sin(q3) * np.sin(q4) - l4 * np.cos(q3) * np.sin(q1)**2 * np.sin(q2) * np.sin(q4) - l4 * np.cos(q4) * np.sin(q1)**2 * np.sin(q2) * np.sin(q3)), 0, 0],
            [(np.cos(q1) * np.cos(q2) * np.cos(q3) - np.cos(q1) * np.sin(q2) * np.sin(q3)) / (l2 * np.sin(q3) * np.cos(q1)**2 * np.cos(q2)**2 + l2 * np.sin(q3) * np.cos(q1)**2 * np.sin(q2)**2 + l2 * np.sin(q3) * np.cos(q2)**2 * np.sin(q1)**2 + l2 * np.sin(q3) * np.sin(q1)**2 * np.sin(q2)**2), -(np.sin(q1) * np.sin(q2) * np.sin(q3) - np.cos(q2) * np.cos(q3) * np.sin(q1)) / (l2 * np.sin(q3) * np.cos(q1)**2 * np.cos(q2)**2 + l2 * np.sin(q3) * np.cos(q1)**2 * np.sin(q2)**2 + l2 * np.sin(q3) * np.cos(q2)**2 * np.sin(q1)**2 + l2 * np.sin(q3) * np.sin(q1)**2 * np.sin(q2)**2), (np.cos(q2) * np.sin(q3) + np.cos(q3) * np.sin(q2)) / (l2 * np.sin(q3) * np.cos(q2)**2 + l2 * np.sin(q3) * np.sin(q2)**2), (l4 * np.sin(q4) * np.cos(q3)**2 + l4 * np.sin(q4) * np.sin(q3)**2) / (l2 * np.sin(q1) * np.sin(q3))],
            [-(l2 * np.cos(q1) * np.cos(q2) + l3 * np.cos(q1) * np.cos(q2) * np.cos(q3) - l3 * np.cos(q1) * np.sin(q2) * np.sin(q3)) / (l2 * l3 * np.sin(q3) * np.cos(q1)**2 * np.cos(q2)**2 + l2 * l3 * np.sin(q3) * np.cos(q1)**2 * np.sin(q2)**2 + l2 * l3 * np.sin(q3) * np.cos(q2)**2 * np.sin(q1)**2 + l2 * l3 * np.sin(q3) * np.sin(q1)**2 * np.sin(q2)**2), -(l2 * np.cos(q2) * np.sin(q1) + l3 * np.cos(q2) * np.cos(q3) * np.sin(q1) - l3 * np.sin(q1) * np.sin(q2) * np.sin(q3)) / (l2 * l3 * np.sin(q3) * np.cos(q1)**2 * np.cos(q2)**2 + l2 * l3 * np.sin(q3) * np.cos(q1)**2 * np.sin(q2)**2 + l2 * l3 * np.sin(q3) * np.cos(q2)**2 * np.sin(q1)**2 + l2 * l3 * np.sin(q3) * np.sin(q1)**2 * np.sin(q2)**2), -(l2 * np.sin(q2) + l3 * np.cos(q2) * np.sin(q3) + l3 * np.cos(q3) * np.sin(q2)) / (l2 * l3 * np.sin(q3) * np.cos(q2)**2 + l2 * l3 * np.sin(q3) * np.sin(q2)**2), -(l3 * l4 * np.sin(q4) * np.cos(q3)**2 + l2 * l4 * np.sin(q4) * np.cos(q3) + l3 * l4 * np.sin(q4) * np.sin(q3)**2 + l2 * l4 * np.cos(q4) * np.sin(q3)) / (l2 * l3 * np.sin(q1) * np.sin(q3))],
            [(np.cos(q1) * np.cos(q2)) / (l3 * np.sin(q3) * np.cos(q1)**2 * np.cos(q2)**2 + l3 * np.sin(q3) * np.cos(q1)**2 * np.sin(q2)**2 + l3 * np.sin(q3) * np.cos(q2)**2 * np.sin(q1)**2 + l3 * np.sin(q3) * np.sin(q1)**2 * np.sin(q2)**2), (np.cos(q2) * np.sin(q1)) / (l3 * np.sin(q3) * np.cos(q1)**2 * np.cos(q2)**2 + l3 * np.sin(q3) * np.cos(q1)**2 * np.sin(q2)**2 + l3 * np.sin(q3) * np.cos(q2)**2 * np.sin(q1)**2 + l3 * np.sin(q3) * np.sin(q1)**2 * np.sin(q2)**2), np.sin(q2) / (l3 * np.sin(q3) * np.cos(q2)**2 + l3 * np.sin(q3) * np.sin(q2)**2), (l3 * np.sin(q3) + l4 * np.cos(q3) * np.sin(q4) + l4 * np.cos(q4) * np.sin(q3)) / (l3 * np.sin(q1) * np.sin(q3))]
        ])
        dq = J_inv.dot(np.array([v_x_s, v_y_s, v_z_s, 0]))
        dq1 = dq[0]
        dq2 = dq[1]
        dq3 = dq[2]
        dq4 = dq[3]
        dJ_dt = np.array([
            [l4 * np.cos(q4) * (np.cos(q1) * np.sin(q2) * np.sin(q3) * dq1 + np.cos(q2) * np.sin(q1) * np.sin(q3) * dq2 + np.cos(q3) * np.sin(q1) * np.sin(q2) * dq2 + np.cos(q2) * np.sin(q1) * np.sin(q3) * dq3 + np.cos(q3) * np.sin(q1) * np.sin(q2) * dq3 - np.cos(q1) * np.cos(q2) * np.cos(q3) * dq1) - l1 * np.cos(q1) * dq1 + l4 * np.sin(q4) * (np.cos(q1) * np.cos(q2) * np.sin(q3) * dq1 + np.cos(q1) * np.cos(q3) * np.sin(q2) * dq1 + np.cos(q2) * np.cos(q3) * np.sin(q1) * dq2 + np.cos(q2) * np.cos(q3) * np.sin(q1) * dq3 - np.sin(q1) * np.sin(q2) * np.sin(q3) * dq2 - np.sin(q1) * np.sin(q2) * np.sin(q3) * dq3) - l2 * np.cos(q1) * np.cos(q2) * dq1 + l2 * np.sin(q1) * np.sin(q2) * dq2 + l4 * np.cos(q4) * (np.cos(q2) * np.sin(q1) * np.sin(q3) + np.cos(q3) * np.sin(q1) * np.sin(q2)) * dq4 + l4 * np.sin(q4) * (np.cos(q2) * np.cos(q3) * np.sin(q1) - np.sin(q1) * np.sin(q2) * np.sin(q3)) * dq4 - l3 * np.cos(q1) * np.cos(q2) * np.cos(q3) * dq1 + l3 * np.cos(q1) * np.sin(q2) * np.sin(q3) * dq1 + l3 * np.cos(q2) * np.sin(q1) * np.sin(q3) * dq2 + l3 * np.cos(q3) * np.sin(q1) * np.sin(q2) * dq2 + l3 * np.cos(q2) * np.sin(q1) * np.sin(q3) * dq3 + l3 * np.cos(q3) * np.sin(q1) * np.sin(q2) * dq3, l4 * np.cos(q4) * (np.cos(q2) * np.sin(q1) * np.sin(q3) * dq1 + np.cos(q3) * np.sin(q1) * np.sin(q2) * dq1 + np.cos(q1) * np.sin(q2) * np.sin(q3) * dq2 + np.cos(q1) * np.sin(q2) * np.sin(q3) * dq3 - np.cos(q1) * np.cos(q2) * np.cos(q3) * dq2 - np.cos(q1) * np.cos(q2) * np.cos(q3) * dq3) + l4 * np.sin(q4) * (np.cos(q2) * np.cos(q3) * np.sin(q1) * dq1 + np.cos(q1) * np.cos(q2) * np.sin(q3) * dq2 + np.cos(q1) * np.cos(q3) * np.sin(q2) * dq2 + np.cos(q1) * np.cos(q2) * np.sin(q3) * dq3 + np.cos(q1) * np.cos(q3) * np.sin(q2) * dq3 - np.sin(q1) * np.sin(q2) * np.sin(q3) * dq1) - l2 * np.cos(q1) * np.cos(q2) * dq2 + l2 * np.sin(q1) * np.sin(q2) * dq1 - l4 * np.cos(q4) * (np.cos(q1) * np.cos(q2) * np.cos(q3) - np.cos(q1) * np.sin(q2) * np.sin(q3)) * dq4 + l4 * np.sin(q4) * (np.cos(q1) * np.cos(q2) * np.sin(q3) + np.cos(q1) * np.cos(q3) * np.sin(q2)) * dq4 - l3 * np.cos(q1) * np.cos(q2) * np.cos(q3) * dq2 - l3 * np.cos(q1) * np.cos(q2) * np.cos(q3) * dq3 + l3 * np.cos(q2) * np.sin(q1) * np.sin(q3) * dq1 + l3 * np.cos(q3) * np.sin(q1) * np.sin(q2) * dq1 + l3 * np.cos(q1) * np.sin(q2) * np.sin(q3) * dq2 + l3 * np.cos(q1) * np.sin(q2) * np.sin(q3) * dq3, l4 * np.cos(q4) * (np.cos(q2) * np.sin(q1) * np.sin(q3) * dq1 + np.cos(q3) * np.sin(q1) * np.sin(q2) * dq1 + np.cos(q1) * np.sin(q2) * np.sin(q3) * dq2 + np.cos(q1) * np.sin(q2) * np.sin(q3) * dq3 - np.cos(q1) * np.cos(q2) * np.cos(q3) * dq2 - np.cos(q1) * np.cos(q2) * np.cos(q3) * dq3) + l4 * np.sin(q4) * (np.cos(q2) * np.cos(q3) * np.sin(q1) * dq1 + np.cos(q1) * np.cos(q2) * np.sin(q3) * dq2 + np.cos(q1) * np.cos(q3) * np.sin(q2) * dq2 + np.cos(q1) * np.cos(q2) * np.sin(q3) * dq3 + np.cos(q1) * np.cos(q3) * np.sin(q2) * dq3 - np.sin(q1) * np.sin(q2) * np.sin(q3) * dq1) - l4 * np.cos(q4) * (np.cos(q1) * np.cos(q2) * np.cos(q3) - np.cos(q1) * np.sin(q2) * np.sin(q3)) * dq4 + l4 * np.sin(q4) * (np.cos(q1) * np.cos(q2) * np.sin(q3) + np.cos(q1) * np.cos(q3) * np.sin(q2)) * dq4 - l3 * np.cos(q1) * np.cos(q2) * np.cos(q3) * dq2 - l3 * np.cos(q1) * np.cos(q2) * np.cos(q3) * dq3 + l3 * np.cos(q2) * np.sin(q1) * np.sin(q3) * dq1 + l3 * np.cos(q3) * np.sin(q1) * np.sin(q2) * dq1 + l3 * np.cos(q1) * np.sin(q2) * np.sin(q3) * dq2 + l3 * np.cos(q1) * np.sin(q2) * np.sin(q3) * dq3, l4 * np.cos(q4) * (np.cos(q2) * np.sin(q1) * np.sin(q3) * dq1 + np.cos(q3) * np.sin(q1) * np.sin(q2) * dq1 + np.cos(q1) * np.sin(q2) * np.sin(q3) * dq2 + np.cos(q1) * np.sin(q2) * np.sin(q3) * dq3 - np.cos(q1) * np.cos(q2) * np.cos(q3) * dq2 - np.cos(q1) * np.cos(q2) * np.cos(q3) * dq3) + l4 * np.sin(q4) * (np.cos(q2) * np.cos(q3) * np.sin(q1) * dq1 + np.cos(q1) * np.cos(q2) * np.sin(q3) * dq2 + np.cos(q1) * np.cos(q3) * np.sin(q2) * dq2 + np.cos(q1) * np.cos(q2) * np.sin(q3) * dq3 + np.cos(q1) * np.cos(q3) * np.sin(q2) * dq3 - np.sin(q1) * np.sin(q2) * np.sin(q3) * dq1) - l4 * np.cos(q4) * (np.cos(q1) * np.cos(q2) * np.cos(q3) - np.cos(q1) * np.sin(q2) * np.sin(q3)) * dq4 + l4 * np.sin(q4) * (np.cos(q1) * np.cos(q2) * np.sin(q3) + np.cos(q1) * np.cos(q3) * np.sin(q2)) * dq4],
            [l4 * np.sin(q4) * (np.cos(q2) * np.sin(q1) * np.sin(q3) * dq1 + np.cos(q3) * np.sin(q1) * np.sin(q2) * dq1 + np.cos(q1) * np.sin(q2) * np.sin(q3) * dq2 + np.cos(q1) * np.sin(q2) * np.sin(q3) * dq3 - np.cos(q1) * np.cos(q2) * np.cos(q3) * dq2 - np.cos(q1) * np.cos(q2) * np.cos(q3) * dq3) - l4 * np.cos(q4) * (np.cos(q2) * np.cos(q3) * np.sin(q1) * dq1 + np.cos(q1) * np.cos(q2) * np.sin(q3) * dq2 + np.cos(q1) * np.cos(q3) * np.sin(q2) * dq2 + np.cos(q1) * np.cos(q2) * np.sin(q3) * dq3 + np.cos(q1) * np.cos(q3) * np.sin(q2) * dq3 - np.sin(q1) * np.sin(q2) * np.sin(q3) * dq1) - l1 * np.sin(q1) * dq1 - l2 * np.cos(q2) * np.sin(q1) * dq1 - l2 * np.cos(q1) * np.sin(q2) * dq2 - l4 * np.cos(q4) * (np.cos(q1) * np.cos(q2) * np.sin(q3) + np.cos(q1) * np.cos(q3) * np.sin(q2)) * dq4 - l4 * np.sin(q4) * (np.cos(q1) * np.cos(q2) * np.cos(q3) - np.cos(q1) * np.sin(q2) * np.sin(q3)) * dq4 - l3 * np.cos(q2) * np.cos(q3) * np.sin(q1) * dq1 - l3 * np.cos(q1) * np.cos(q2) * np.sin(q3) * dq2 - l3 * np.cos(q1) * np.cos(q3) * np.sin(q2) * dq2 - l3 * np.cos(q1) * np.cos(q2) * np.sin(q3) * dq3 - l3 * np.cos(q1) * np.cos(q3) * np.sin(q2) * dq3 + l3 * np.sin(q1) * np.sin(q2) * np.sin(q3) * dq1, l4 * np.sin(q4) * (np.cos(q1) * np.sin(q2) * np.sin(q3) * dq1 + np.cos(q2) * np.sin(q1) * np.sin(q3) * dq2 + np.cos(q3) * np.sin(q1) * np.sin(q2) * dq2 + np.cos(q2) * np.sin(q1) * np.sin(q3) * dq3 + np.cos(q3) * np.sin(q1) * np.sin(q2) * dq3 - np.cos(q1) * np.cos(q2) * np.cos(q3) * dq1) - l4 * np.cos(q4) * (np.cos(q1) * np.cos(q2) * np.sin(q3) * dq1 + np.cos(q1) * np.cos(q3) * np.sin(q2) * dq1 + np.cos(q2) * np.cos(q3) * np.sin(q1) * dq2 + np.cos(q2) * np.cos(q3) * np.sin(q1) * dq3 - np.sin(q1) * np.sin(q2) * np.sin(q3) * dq2 - np.sin(q1) * np.sin(q2) * np.sin(q3) * dq3) - l2 * np.cos(q1) * np.sin(q2) * dq1 - l2 * np.cos(q2) * np.sin(q1) * dq2 - l4 * np.cos(q4) * (np.cos(q2) * np.cos(q3) * np.sin(q1) - np.sin(q1) * np.sin(q2) * np.sin(q3)) * dq4 + l4 * np.sin(q4) * (np.cos(q2) * np.sin(q1) * np.sin(q3) + np.cos(q3) * np.sin(q1) * np.sin(q2)) * dq4 - l3 * np.cos(q1) * np.cos(q2) * np.sin(q3) * dq1 - l3 * np.cos(q1) * np.cos(q3) * np.sin(q2) * dq1 - l3 * np.cos(q2) * np.cos(q3) * np.sin(q1) * dq2 - l3 * np.cos(q2) * np.cos(q3) * np.sin(q1) * dq3 + l3 * np.sin(q1) * np.sin(q2) * np.sin(q3) * dq2 + l3 * np.sin(q1) * np.sin(q2) * np.sin(q3) * dq3, l4 * np.sin(q4) * (np.cos(q1) * np.sin(q2) * np.sin(q3) * dq1 + np.cos(q2) * np.sin(q1) * np.sin(q3) * dq2 + np.cos(q3) * np.sin(q1) * np.sin(q2) * dq2 + np.cos(q2) * np.sin(q1) * np.sin(q3) * dq3 + np.cos(q3) * np.sin(q1) * np.sin(q2) * dq3 - np.cos(q1) * np.cos(q2) * np.cos(q3) * dq1) - l4 * np.cos(q4) * (np.cos(q1) * np.cos(q2) * np.sin(q3) * dq1 + np.cos(q1) * np.cos(q3) * np.sin(q2) * dq1 + np.cos(q2) * np.cos(q3) * np.sin(q1) * dq2 + np.cos(q2) * np.cos(q3) * np.sin(q1) * dq3 - np.sin(q1) * np.sin(q2) * np.sin(q3) * dq2 - np.sin(q1) * np.sin(q2) * np.sin(q3) * dq3) - l4 * np.cos(q4) * (np.cos(q2) * np.cos(q3) * np.sin(q1) - np.sin(q1) * np.sin(q2) * np.sin(q3)) * dq4 + l4 * np.sin(q4) * (np.cos(q2) * np.sin(q1) * np.sin(q3) + np.cos(q3) * np.sin(q1) * np.sin(q2)) * dq4 - l3 * np.cos(q1) * np.cos(q2) * np.sin(q3) * dq1 - l3 * np.cos(q1) * np.cos(q3) * np.sin(q2) * dq1 - l3 * np.cos(q2) * np.cos(q3) * np.sin(q1) * dq2 - l3 * np.cos(q2) * np.cos(q3) * np.sin(q1) * dq3 + l3 * np.sin(q1) * np.sin(q2) * np.sin(q3) * dq2 + l3 * np.sin(q1) * np.sin(q2) * np.sin(q3) * dq3, l4 * np.sin(q4) * (np.cos(q1) * np.sin(q2) * np.sin(q3) * dq1 + np.cos(q2) * np.sin(q1) * np.sin(q3) * dq2 + np.cos(q3) * np.sin(q1) * np.sin(q2) * dq2 + np.cos(q2) * np.sin(q1) * np.sin(q3) * dq3 + np.cos(q3) * np.sin(q1) * np.sin(q2) * dq3 - np.cos(q1) * np.cos(q2) * np.cos(q3) * dq1) - l4 * np.cos(q4) * (np.cos(q1) * np.cos(q2) * np.sin(q3) * dq1 + np.cos(q1) * np.cos(q3) * np.sin(q2) * dq1 + np.cos(q2) * np.cos(q3) * np.sin(q1) * dq2 + np.cos(q2) * np.cos(q3) * np.sin(q1) * dq3 - np.sin(q1) * np.sin(q2) * np.sin(q3) * dq2 - np.sin(q1) * np.sin(q2) * np.sin(q3) * dq3) - l4 * np.cos(q4) * (np.cos(q2) * np.cos(q3) * np.sin(q1) - np.sin(q1) * np.sin(q2) * np.sin(q3)) * dq4 + l4 * np.sin(q4) * (np.cos(q2) * np.sin(q1) * np.sin(q3) + np.cos(q3) * np.sin(q1) * np.sin(q2)) * dq4],
            [0, l4 * np.sin(q4) * (np.sin(q2) * np.sin(q3) - np.cos(q2) * np.cos(q3)) * dq4 - l4 * np.sin(q4) * (np.cos(q2) * np.cos(q3) * dq2 + np.cos(q2) * np.cos(q3) * dq3 - np.sin(q2) * np.sin(q3) * dq2 - np.sin(q2) * np.sin(q3) * dq3) - l2 * np.sin(q2) * dq2 - l3 * np.cos(q2) * np.sin(q3) * dq2 - l3 * np.cos(q3) * np.sin(q2) * dq2 - l3 * np.cos(q2) * np.sin(q3) * dq3 - l3 * np.cos(q3) * np.sin(q2) * dq3 - l4 * np.cos(q4) * (np.cos(q2) * np.sin(q3) + np.cos(q3) * np.sin(q2)) * dq4 - l4 * np.cos(q4) * (np.cos(q2) * np.sin(q3) * dq2 + np.cos(q3) * np.sin(q2) * dq2 + np.cos(q2) * np.sin(q3) * dq3 + np.cos(q3) * np.sin(q2) * dq3), l4 * np.sin(q4) * (np.sin(q2) * np.sin(q3) - np.cos(q2) * np.cos(q3)) * dq4 - l4 * np.sin(q4) * (np.cos(q2) * np.cos(q3) * dq2 + np.cos(q2) * np.cos(q3) * dq3 - np.sin(q2) * np.sin(q3) * dq2 - np.sin(q2) * np.sin(q3) * dq3) - l3 * np.cos(q2) * np.sin(q3) * dq2 - l3 * np.cos(q3) * np.sin(q2) * dq2 - l3 * np.cos(q2) * np.sin(q3) * dq3 - l3 * np.cos(q3) * np.sin(q2) * dq3 - l4 * np.cos(q4) * (np.cos(q2) * np.sin(q3) + np.cos(q3) * np.sin(q2)) * dq4 - l4 * np.cos(q4) * (np.cos(q2) * np.sin(q3) * dq2 + np.cos(q3) * np.sin(q2) * dq2 + np.cos(q2) * np.sin(q3) * dq3 + np.cos(q3) * np.sin(q2) * dq3), l4 * np.sin(q4) * (np.sin(q2) * np.sin(q3) - np.cos(q2) * np.cos(q3)) * dq4 - l4 * np.sin(q4) * (np.cos(q2) * np.cos(q3) * dq2 + np.cos(q2) * np.cos(q3) * dq3 - np.sin(q2) * np.sin(q3) * dq2 - np.sin(q2) * np.sin(q3) * dq3) - l4 * np.cos(q4) * (np.cos(q2) * np.sin(q3) + np.cos(q3) * np.sin(q2)) * dq4 - l4 * np.cos(q4) * (np.cos(q2) * np.sin(q3) * dq2 + np.cos(q3) * np.sin(q2) * dq2 + np.cos(q2) * np.sin(q3) * dq3 + np.cos(q3) * np.sin(q2) * dq3)],
            [0, np.cos(q1) * dq1, np.cos(q1) * dq1, np.cos(q1) * dq1]
        ])

        
        ddq = J_inv.dot(np.array([a_x_s, a_y_s, a_z_s, 0]) - dJ_dt.dot(dq))

    else:
        # Фаза перемещения thet_B (обратная thet_S) и thet_P

        # Расчет thet_B
        time_2 = 2 * T_f - time

        if 0 <= time_2 <= T_b:
            p_x_b = np.dot([time_2**2, time_2, 1, 0, 0, 0, 0, 0], C_x)
            p_y_b = np.dot([time_2**2, time_2, 1, 0, 0, 0, 0, 0], C_y)
            p_z_b = np.dot([time_2**2, time_2, 1, 0, 0, 0, 0, 0], C_z)

            v_x_b = np.dot([2*time_2, 1, 0, 0, 0, 0, 0, 0], C_x)
            v_y_b = np.dot([2*time_2, 1, 0, 0, 0, 0, 0, 0], C_y)
            v_z_b = np.dot([2*time_2, 1, 0, 0, 0, 0, 0, 0], C_z)

            a_x_b = np.dot([2, 0, 0, 0, 0, 0, 0, 0], C_x)
            a_y_b = np.dot([2, 0, 0, 0, 0, 0, 0, 0], C_y)
            a_z_b = np.dot([2, 0, 0, 0, 0, 0, 0, 0], C_z)

        elif T_b < time_2 <= T_f - T_b:
            p_x_b = np.dot([0, 0, 0, time_2, 1, 0, 0, 0], C_x)
            p_y_b = np.dot([0, 0, 0, time_2, 1, 0, 0, 0], C_y)
            p_z_b = np.dot([0, 0, 0, time_2, 1, 0, 0, 0], C_z)

            v_x_b = np.dot([0, 0, 0, 1, 0, 0, 0, 0], C_x)
            v_y_b = np.dot([0, 0, 0, 1, 0, 0, 0, 0], C_y)
            v_z_b = np.dot([0, 0, 0, 1, 0, 0, 0, 0], C_z)

            a_x_b = np.dot([0, 0, 0, 0, 0, 0, 0, 0], C_x)
            a_y_b = np.dot([0, 0, 0, 0, 0, 0, 0, 0], C_y)
            a_z_b = np.dot([0, 0, 0, 0, 0, 0, 0, 0], C_z)

        elif T_f - T_b < time_2 <= T_f + 1:
            p_x_b = np.dot([0, 0, 0, 0, 0, time_2**2, time_2, 1], C_x)
            p_y_b = np.dot([0, 0, 0, 0, 0, time_2**2, time_2, 1], C_y)
            p_z_b = np.dot([0, 0, 0, 0, 0, time_2**2, time_2, 1], C_z)

            v_x_b = np.dot([0, 0, 0, 0, 0, 2*time_2, 1, 0], C_x)
            v_y_b = np.dot([0, 0, 0, 0, 0, 2*time_2, 1, 0], C_y)
            v_z_b = np.dot([0, 0, 0, 0, 0, 2*time_2, 1, 0], C_z)

            a_x_b = np.dot([0, 0, 0, 0, 0, 2, 0, 0], C_x)
            a_y_b = np.dot([0, 0, 0, 0, 0, 2, 0, 0], C_y)
            a_z_b = np.dot([0, 0, 0, 0, 0, 2, 0, 0], C_z)

        # Решение обратной задачи кинематики

        q = IK(p_x_b, p_y_b, p_z_b, l1, l2, l3, l4)
        q1 = q[0]
        q2 = q[1]
        q3 = q[2]
        q4 = q[3]
        q_b = q
        # Якобиан
        J_inv = np.array([
            [-np.sin(q1) / (l1 * np.sin(q1)**2 + l1 * np.cos(q1)**2 + l2 * np.cos(q1)**2 * np.cos(q2) + l2 * np.cos(q2) * np.sin(q1)**2 + l3 * np.cos(q1)**2 * np.cos(q2) * np.cos(q3) + l3 * np.cos(q2) * np.cos(q3) * np.sin(q1)**2 - l3 * np.cos(q1)**2 * np.sin(q2) * np.sin(q3) - l3 * np.sin(q1)**2 * np.sin(q2) * np.sin(q3) + l4 * np.cos(q1)**2 * np.cos(q2) * np.cos(q3) * np.cos(q4) + l4 * np.cos(q2) * np.cos(q3) * np.cos(q4) * np.sin(q1)**2 - l4 * np.cos(q1)**2 * np.cos(q2) * np.sin(q3) * np.sin(q4) - l4 * np.cos(q1)**2 * np.cos(q3) * np.sin(q2) * np.sin(q4) - l4 * np.cos(q1)**2 * np.cos(q4) * np.sin(q2) * np.sin(q3) - l4 * np.cos(q2) * np.sin(q1)**2 * np.sin(q3) * np.sin(q4) - l4 * np.cos(q3) * np.sin(q1)**2 * np.sin(q2) * np.sin(q4) - l4 * np.cos(q4) * np.sin(q1)**2 * np.sin(q2) * np.sin(q3)), np.cos(q1) / (l1 * np.sin(q1)**2 + l1 * np.cos(q1)**2 + l2 * np.cos(q1)**2 * np.cos(q2) + l2 * np.cos(q2) * np.sin(q1)**2 + l3 * np.cos(q1)**2 * np.cos(q2) * np.cos(q3) + l3 * np.cos(q2) * np.cos(q3) * np.sin(q1)**2 - l3 * np.cos(q1)**2 * np.sin(q2) * np.sin(q3) - l3 * np.sin(q1)**2 * np.sin(q2) * np.sin(q3) + l4 * np.cos(q1)**2 * np.cos(q2) * np.cos(q3) * np.cos(q4) + l4 * np.cos(q2) * np.cos(q3) * np.cos(q4) * np.sin(q1)**2 - l4 * np.cos(q1)**2 * np.cos(q2) * np.sin(q3) * np.sin(q4) - l4 * np.cos(q1)**2 * np.cos(q3) * np.sin(q2) * np.sin(q4) - l4 * np.cos(q1)**2 * np.cos(q4) * np.sin(q2) * np.sin(q3) - l4 * np.cos(q2) * np.sin(q1)**2 * np.sin(q3) * np.sin(q4) - l4 * np.cos(q3) * np.sin(q1)**2 * np.sin(q2) * np.sin(q4) - l4 * np.cos(q4) * np.sin(q1)**2 * np.sin(q2) * np.sin(q3)), 0, 0],
            [(np.cos(q1) * np.cos(q2) * np.cos(q3) - np.cos(q1) * np.sin(q2) * np.sin(q3)) / (l2 * np.sin(q3) * np.cos(q1)**2 * np.cos(q2)**2 + l2 * np.sin(q3) * np.cos(q1)**2 * np.sin(q2)**2 + l2 * np.sin(q3) * np.cos(q2)**2 * np.sin(q1)**2 + l2 * np.sin(q3) * np.sin(q1)**2 * np.sin(q2)**2), -(np.sin(q1) * np.sin(q2) * np.sin(q3) - np.cos(q2) * np.cos(q3) * np.sin(q1)) / (l2 * np.sin(q3) * np.cos(q1)**2 * np.cos(q2)**2 + l2 * np.sin(q3) * np.cos(q1)**2 * np.sin(q2)**2 + l2 * np.sin(q3) * np.cos(q2)**2 * np.sin(q1)**2 + l2 * np.sin(q3) * np.sin(q1)**2 * np.sin(q2)**2), (np.cos(q2) * np.sin(q3) + np.cos(q3) * np.sin(q2)) / (l2 * np.sin(q3) * np.cos(q2)**2 + l2 * np.sin(q3) * np.sin(q2)**2), (l4 * np.sin(q4) * np.cos(q3)**2 + l4 * np.sin(q4) * np.sin(q3)**2) / (l2 * np.sin(q1) * np.sin(q3))],
            [-(l2 * np.cos(q1) * np.cos(q2) + l3 * np.cos(q1) * np.cos(q2) * np.cos(q3) - l3 * np.cos(q1) * np.sin(q2) * np.sin(q3)) / (l2 * l3 * np.sin(q3) * np.cos(q1)**2 * np.cos(q2)**2 + l2 * l3 * np.sin(q3) * np.cos(q1)**2 * np.sin(q2)**2 + l2 * l3 * np.sin(q3) * np.cos(q2)**2 * np.sin(q1)**2 + l2 * l3 * np.sin(q3) * np.sin(q1)**2 * np.sin(q2)**2), -(l2 * np.cos(q2) * np.sin(q1) + l3 * np.cos(q2) * np.cos(q3) * np.sin(q1) - l3 * np.sin(q1) * np.sin(q2) * np.sin(q3)) / (l2 * l3 * np.sin(q3) * np.cos(q1)**2 * np.cos(q2)**2 + l2 * l3 * np.sin(q3) * np.cos(q1)**2 * np.sin(q2)**2 + l2 * l3 * np.sin(q3) * np.cos(q2)**2 * np.sin(q1)**2 + l2 * l3 * np.sin(q3) * np.sin(q1)**2 * np.sin(q2)**2), -(l2 * np.sin(q2) + l3 * np.cos(q2) * np.sin(q3) + l3 * np.cos(q3) * np.sin(q2)) / (l2 * l3 * np.sin(q3) * np.cos(q2)**2 + l2 * l3 * np.sin(q3) * np.sin(q2)**2), -(l3 * l4 * np.sin(q4) * np.cos(q3)**2 + l2 * l4 * np.sin(q4) * np.cos(q3) + l3 * l4 * np.sin(q4) * np.sin(q3)**2 + l2 * l4 * np.cos(q4) * np.sin(q3)) / (l2 * l3 * np.sin(q1) * np.sin(q3))],
            [(np.cos(q1) * np.cos(q2)) / (l3 * np.sin(q3) * np.cos(q1)**2 * np.cos(q2)**2 + l3 * np.sin(q3) * np.cos(q1)**2 * np.sin(q2)**2 + l3 * np.sin(q3) * np.cos(q2)**2 * np.sin(q1)**2 + l3 * np.sin(q3) * np.sin(q1)**2 * np.sin(q2)**2), (np.cos(q2) * np.sin(q1)) / (l3 * np.sin(q3) * np.cos(q1)**2 * np.cos(q2)**2 + l3 * np.sin(q3) * np.cos(q1)**2 * np.sin(q2)**2 + l3 * np.sin(q3) * np.cos(q2)**2 * np.sin(q1)**2 + l3 * np.sin(q3) * np.sin(q1)**2 * np.sin(q2)**2), np.sin(q2) / (l3 * np.sin(q3) * np.cos(q2)**2 + l3 * np.sin(q3) * np.sin(q2)**2), (l3 * np.sin(q3) + l4 * np.cos(q3) * np.sin(q4) + l4 * np.cos(q4) * np.sin(q3)) / (l3 * np.sin(q1) * np.sin(q3))]
        ])
        dq = J_inv.dot(np.array([v_x_b, v_y_b, v_z_b, 0]))
        dq1 = dq[0]
        dq2 = dq[1]
        dq3 = dq[2]
        dq4 = dq[3]
        dq_b = dq
        dJ_dt = np.array([
            [l4 * np.cos(q4) * (np.cos(q1) * np.sin(q2) * np.sin(q3) * dq1 + np.cos(q2) * np.sin(q1) * np.sin(q3) * dq2 + np.cos(q3) * np.sin(q1) * np.sin(q2) * dq2 + np.cos(q2) * np.sin(q1) * np.sin(q3) * dq3 + np.cos(q3) * np.sin(q1) * np.sin(q2) * dq3 - np.cos(q1) * np.cos(q2) * np.cos(q3) * dq1) - l1 * np.cos(q1) * dq1 + l4 * np.sin(q4) * (np.cos(q1) * np.cos(q2) * np.sin(q3) * dq1 + np.cos(q1) * np.cos(q3) * np.sin(q2) * dq1 + np.cos(q2) * np.cos(q3) * np.sin(q1) * dq2 + np.cos(q2) * np.cos(q3) * np.sin(q1) * dq3 - np.sin(q1) * np.sin(q2) * np.sin(q3) * dq2 - np.sin(q1) * np.sin(q2) * np.sin(q3) * dq3) - l2 * np.cos(q1) * np.cos(q2) * dq1 + l2 * np.sin(q1) * np.sin(q2) * dq2 + l4 * np.cos(q4) * (np.cos(q2) * np.sin(q1) * np.sin(q3) + np.cos(q3) * np.sin(q1) * np.sin(q2)) * dq4 + l4 * np.sin(q4) * (np.cos(q2) * np.cos(q3) * np.sin(q1) - np.sin(q1) * np.sin(q2) * np.sin(q3)) * dq4 - l3 * np.cos(q1) * np.cos(q2) * np.cos(q3) * dq1 + l3 * np.cos(q1) * np.sin(q2) * np.sin(q3) * dq1 + l3 * np.cos(q2) * np.sin(q1) * np.sin(q3) * dq2 + l3 * np.cos(q3) * np.sin(q1) * np.sin(q2) * dq2 + l3 * np.cos(q2) * np.sin(q1) * np.sin(q3) * dq3 + l3 * np.cos(q3) * np.sin(q1) * np.sin(q2) * dq3, l4 * np.cos(q4) * (np.cos(q2) * np.sin(q1) * np.sin(q3) * dq1 + np.cos(q3) * np.sin(q1) * np.sin(q2) * dq1 + np.cos(q1) * np.sin(q2) * np.sin(q3) * dq2 + np.cos(q1) * np.sin(q2) * np.sin(q3) * dq3 - np.cos(q1) * np.cos(q2) * np.cos(q3) * dq2 - np.cos(q1) * np.cos(q2) * np.cos(q3) * dq3) + l4 * np.sin(q4) * (np.cos(q2) * np.cos(q3) * np.sin(q1) * dq1 + np.cos(q1) * np.cos(q2) * np.sin(q3) * dq2 + np.cos(q1) * np.cos(q3) * np.sin(q2) * dq2 + np.cos(q1) * np.cos(q2) * np.sin(q3) * dq3 + np.cos(q1) * np.cos(q3) * np.sin(q2) * dq3 - np.sin(q1) * np.sin(q2) * np.sin(q3) * dq1) - l2 * np.cos(q1) * np.cos(q2) * dq2 + l2 * np.sin(q1) * np.sin(q2) * dq1 - l4 * np.cos(q4) * (np.cos(q1) * np.cos(q2) * np.cos(q3) - np.cos(q1) * np.sin(q2) * np.sin(q3)) * dq4 + l4 * np.sin(q4) * (np.cos(q1) * np.cos(q2) * np.sin(q3) + np.cos(q1) * np.cos(q3) * np.sin(q2)) * dq4 - l3 * np.cos(q1) * np.cos(q2) * np.cos(q3) * dq2 - l3 * np.cos(q1) * np.cos(q2) * np.cos(q3) * dq3 + l3 * np.cos(q2) * np.sin(q1) * np.sin(q3) * dq1 + l3 * np.cos(q3) * np.sin(q1) * np.sin(q2) * dq1 + l3 * np.cos(q1) * np.sin(q2) * np.sin(q3) * dq2 + l3 * np.cos(q1) * np.sin(q2) * np.sin(q3) * dq3, l4 * np.cos(q4) * (np.cos(q2) * np.sin(q1) * np.sin(q3) * dq1 + np.cos(q3) * np.sin(q1) * np.sin(q2) * dq1 + np.cos(q1) * np.sin(q2) * np.sin(q3) * dq2 + np.cos(q1) * np.sin(q2) * np.sin(q3) * dq3 - np.cos(q1) * np.cos(q2) * np.cos(q3) * dq2 - np.cos(q1) * np.cos(q2) * np.cos(q3) * dq3) + l4 * np.sin(q4) * (np.cos(q2) * np.cos(q3) * np.sin(q1) * dq1 + np.cos(q1) * np.cos(q2) * np.sin(q3) * dq2 + np.cos(q1) * np.cos(q3) * np.sin(q2) * dq2 + np.cos(q1) * np.cos(q2) * np.sin(q3) * dq3 + np.cos(q1) * np.cos(q3) * np.sin(q2) * dq3 - np.sin(q1) * np.sin(q2) * np.sin(q3) * dq1) - l4 * np.cos(q4) * (np.cos(q1) * np.cos(q2) * np.cos(q3) - np.cos(q1) * np.sin(q2) * np.sin(q3)) * dq4 + l4 * np.sin(q4) * (np.cos(q1) * np.cos(q2) * np.sin(q3) + np.cos(q1) * np.cos(q3) * np.sin(q2)) * dq4 - l3 * np.cos(q1) * np.cos(q2) * np.cos(q3) * dq2 - l3 * np.cos(q1) * np.cos(q2) * np.cos(q3) * dq3 + l3 * np.cos(q2) * np.sin(q1) * np.sin(q3) * dq1 + l3 * np.cos(q3) * np.sin(q1) * np.sin(q2) * dq1 + l3 * np.cos(q1) * np.sin(q2) * np.sin(q3) * dq2 + l3 * np.cos(q1) * np.sin(q2) * np.sin(q3) * dq3, l4 * np.cos(q4) * (np.cos(q2) * np.sin(q1) * np.sin(q3) * dq1 + np.cos(q3) * np.sin(q1) * np.sin(q2) * dq1 + np.cos(q1) * np.sin(q2) * np.sin(q3) * dq2 + np.cos(q1) * np.sin(q2) * np.sin(q3) * dq3 - np.cos(q1) * np.cos(q2) * np.cos(q3) * dq2 - np.cos(q1) * np.cos(q2) * np.cos(q3) * dq3) + l4 * np.sin(q4) * (np.cos(q2) * np.cos(q3) * np.sin(q1) * dq1 + np.cos(q1) * np.cos(q2) * np.sin(q3) * dq2 + np.cos(q1) * np.cos(q3) * np.sin(q2) * dq2 + np.cos(q1) * np.cos(q2) * np.sin(q3) * dq3 + np.cos(q1) * np.cos(q3) * np.sin(q2) * dq3 - np.sin(q1) * np.sin(q2) * np.sin(q3) * dq1) - l4 * np.cos(q4) * (np.cos(q1) * np.cos(q2) * np.cos(q3) - np.cos(q1) * np.sin(q2) * np.sin(q3)) * dq4 + l4 * np.sin(q4) * (np.cos(q1) * np.cos(q2) * np.sin(q3) + np.cos(q1) * np.cos(q3) * np.sin(q2)) * dq4],
            [l4 * np.sin(q4) * (np.cos(q2) * np.sin(q1) * np.sin(q3) * dq1 + np.cos(q3) * np.sin(q1) * np.sin(q2) * dq1 + np.cos(q1) * np.sin(q2) * np.sin(q3) * dq2 + np.cos(q1) * np.sin(q2) * np.sin(q3) * dq3 - np.cos(q1) * np.cos(q2) * np.cos(q3) * dq2 - np.cos(q1) * np.cos(q2) * np.cos(q3) * dq3) - l4 * np.cos(q4) * (np.cos(q2) * np.cos(q3) * np.sin(q1) * dq1 + np.cos(q1) * np.cos(q2) * np.sin(q3) * dq2 + np.cos(q1) * np.cos(q3) * np.sin(q2) * dq2 + np.cos(q1) * np.cos(q2) * np.sin(q3) * dq3 + np.cos(q1) * np.cos(q3) * np.sin(q2) * dq3 - np.sin(q1) * np.sin(q2) * np.sin(q3) * dq1) - l1 * np.sin(q1) * dq1 - l2 * np.cos(q2) * np.sin(q1) * dq1 - l2 * np.cos(q1) * np.sin(q2) * dq2 - l4 * np.cos(q4) * (np.cos(q1) * np.cos(q2) * np.sin(q3) + np.cos(q1) * np.cos(q3) * np.sin(q2)) * dq4 - l4 * np.sin(q4) * (np.cos(q1) * np.cos(q2) * np.cos(q3) - np.cos(q1) * np.sin(q2) * np.sin(q3)) * dq4 - l3 * np.cos(q2) * np.cos(q3) * np.sin(q1) * dq1 - l3 * np.cos(q1) * np.cos(q2) * np.sin(q3) * dq2 - l3 * np.cos(q1) * np.cos(q3) * np.sin(q2) * dq2 - l3 * np.cos(q1) * np.cos(q2) * np.sin(q3) * dq3 - l3 * np.cos(q1) * np.cos(q3) * np.sin(q2) * dq3 + l3 * np.sin(q1) * np.sin(q2) * np.sin(q3) * dq1, l4 * np.sin(q4) * (np.cos(q1) * np.sin(q2) * np.sin(q3) * dq1 + np.cos(q2) * np.sin(q1) * np.sin(q3) * dq2 + np.cos(q3) * np.sin(q1) * np.sin(q2) * dq2 + np.cos(q2) * np.sin(q1) * np.sin(q3) * dq3 + np.cos(q3) * np.sin(q1) * np.sin(q2) * dq3 - np.cos(q1) * np.cos(q2) * np.cos(q3) * dq1) - l4 * np.cos(q4) * (np.cos(q1) * np.cos(q2) * np.sin(q3) * dq1 + np.cos(q1) * np.cos(q3) * np.sin(q2) * dq1 + np.cos(q2) * np.cos(q3) * np.sin(q1) * dq2 + np.cos(q2) * np.cos(q3) * np.sin(q1) * dq3 - np.sin(q1) * np.sin(q2) * np.sin(q3) * dq2 - np.sin(q1) * np.sin(q2) * np.sin(q3) * dq3) - l2 * np.cos(q1) * np.sin(q2) * dq1 - l2 * np.cos(q2) * np.sin(q1) * dq2 - l4 * np.cos(q4) * (np.cos(q2) * np.cos(q3) * np.sin(q1) - np.sin(q1) * np.sin(q2) * np.sin(q3)) * dq4 + l4 * np.sin(q4) * (np.cos(q2) * np.sin(q1) * np.sin(q3) + np.cos(q3) * np.sin(q1) * np.sin(q2)) * dq4 - l3 * np.cos(q1) * np.cos(q2) * np.sin(q3) * dq1 - l3 * np.cos(q1) * np.cos(q3) * np.sin(q2) * dq1 - l3 * np.cos(q2) * np.cos(q3) * np.sin(q1) * dq2 - l3 * np.cos(q2) * np.cos(q3) * np.sin(q1) * dq3 + l3 * np.sin(q1) * np.sin(q2) * np.sin(q3) * dq2 + l3 * np.sin(q1) * np.sin(q2) * np.sin(q3) * dq3, l4 * np.sin(q4) * (np.cos(q1) * np.sin(q2) * np.sin(q3) * dq1 + np.cos(q2) * np.sin(q1) * np.sin(q3) * dq2 + np.cos(q3) * np.sin(q1) * np.sin(q2) * dq2 + np.cos(q2) * np.sin(q1) * np.sin(q3) * dq3 + np.cos(q3) * np.sin(q1) * np.sin(q2) * dq3 - np.cos(q1) * np.cos(q2) * np.cos(q3) * dq1) - l4 * np.cos(q4) * (np.cos(q1) * np.cos(q2) * np.sin(q3) * dq1 + np.cos(q1) * np.cos(q3) * np.sin(q2) * dq1 + np.cos(q2) * np.cos(q3) * np.sin(q1) * dq2 + np.cos(q2) * np.cos(q3) * np.sin(q1) * dq3 - np.sin(q1) * np.sin(q2) * np.sin(q3) * dq2 - np.sin(q1) * np.sin(q2) * np.sin(q3) * dq3) - l4 * np.cos(q4) * (np.cos(q2) * np.cos(q3) * np.sin(q1) - np.sin(q1) * np.sin(q2) * np.sin(q3)) * dq4 + l4 * np.sin(q4) * (np.cos(q2) * np.sin(q1) * np.sin(q3) + np.cos(q3) * np.sin(q1) * np.sin(q2)) * dq4 - l3 * np.cos(q1) * np.cos(q2) * np.sin(q3) * dq1 - l3 * np.cos(q1) * np.cos(q3) * np.sin(q2) * dq1 - l3 * np.cos(q2) * np.cos(q3) * np.sin(q1) * dq2 - l3 * np.cos(q2) * np.cos(q3) * np.sin(q1) * dq3 + l3 * np.sin(q1) * np.sin(q2) * np.sin(q3) * dq2 + l3 * np.sin(q1) * np.sin(q2) * np.sin(q3) * dq3, l4 * np.sin(q4) * (np.cos(q1) * np.sin(q2) * np.sin(q3) * dq1 + np.cos(q2) * np.sin(q1) * np.sin(q3) * dq2 + np.cos(q3) * np.sin(q1) * np.sin(q2) * dq2 + np.cos(q2) * np.sin(q1) * np.sin(q3) * dq3 + np.cos(q3) * np.sin(q1) * np.sin(q2) * dq3 - np.cos(q1) * np.cos(q2) * np.cos(q3) * dq1) - l4 * np.cos(q4) * (np.cos(q1) * np.cos(q2) * np.sin(q3) * dq1 + np.cos(q1) * np.cos(q3) * np.sin(q2) * dq1 + np.cos(q2) * np.cos(q3) * np.sin(q1) * dq2 + np.cos(q2) * np.cos(q3) * np.sin(q1) * dq3 - np.sin(q1) * np.sin(q2) * np.sin(q3) * dq2 - np.sin(q1) * np.sin(q2) * np.sin(q3) * dq3) - l4 * np.cos(q4) * (np.cos(q2) * np.cos(q3) * np.sin(q1) - np.sin(q1) * np.sin(q2) * np.sin(q3)) * dq4 + l4 * np.sin(q4) * (np.cos(q2) * np.sin(q1) * np.sin(q3) + np.cos(q3) * np.sin(q1) * np.sin(q2)) * dq4],
            [0, l4 * np.sin(q4) * (np.sin(q2) * np.sin(q3) - np.cos(q2) * np.cos(q3)) * dq4 - l4 * np.sin(q4) * (np.cos(q2) * np.cos(q3) * dq2 + np.cos(q2) * np.cos(q3) * dq3 - np.sin(q2) * np.sin(q3) * dq2 - np.sin(q2) * np.sin(q3) * dq3) - l2 * np.sin(q2) * dq2 - l3 * np.cos(q2) * np.sin(q3) * dq2 - l3 * np.cos(q3) * np.sin(q2) * dq2 - l3 * np.cos(q2) * np.sin(q3) * dq3 - l3 * np.cos(q3) * np.sin(q2) * dq3 - l4 * np.cos(q4) * (np.cos(q2) * np.sin(q3) + np.cos(q3) * np.sin(q2)) * dq4 - l4 * np.cos(q4) * (np.cos(q2) * np.sin(q3) * dq2 + np.cos(q3) * np.sin(q2) * dq2 + np.cos(q2) * np.sin(q3) * dq3 + np.cos(q3) * np.sin(q2) * dq3), l4 * np.sin(q4) * (np.sin(q2) * np.sin(q3) - np.cos(q2) * np.cos(q3)) * dq4 - l4 * np.sin(q4) * (np.cos(q2) * np.cos(q3) * dq2 + np.cos(q2) * np.cos(q3) * dq3 - np.sin(q2) * np.sin(q3) * dq2 - np.sin(q2) * np.sin(q3) * dq3) - l3 * np.cos(q2) * np.sin(q3) * dq2 - l3 * np.cos(q3) * np.sin(q2) * dq2 - l3 * np.cos(q2) * np.sin(q3) * dq3 - l3 * np.cos(q3) * np.sin(q2) * dq3 - l4 * np.cos(q4) * (np.cos(q2) * np.sin(q3) + np.cos(q3) * np.sin(q2)) * dq4 - l4 * np.cos(q4) * (np.cos(q2) * np.sin(q3) * dq2 + np.cos(q3) * np.sin(q2) * dq2 + np.cos(q2) * np.sin(q3) * dq3 + np.cos(q3) * np.sin(q2) * dq3), l4 * np.sin(q4) * (np.sin(q2) * np.sin(q3) - np.cos(q2) * np.cos(q3)) * dq4 - l4 * np.sin(q4) * (np.cos(q2) * np.cos(q3) * dq2 + np.cos(q2) * np.cos(q3) * dq3 - np.sin(q2) * np.sin(q3) * dq2 - np.sin(q2) * np.sin(q3) * dq3) - l4 * np.cos(q4) * (np.cos(q2) * np.sin(q3) + np.cos(q3) * np.sin(q2)) * dq4 - l4 * np.cos(q4) * (np.cos(q2) * np.sin(q3) * dq2 + np.cos(q3) * np.sin(q2) * dq2 + np.cos(q2) * np.sin(q3) * dq3 + np.cos(q3) * np.sin(q2) * dq3)],
            [0, np.cos(q1) * dq1, np.cos(q1) * dq1, np.cos(q1) * dq1]
        ])

        
        ddq = J_inv.dot(np.array([a_x_b, a_y_b, a_z_b, 0]) - dJ_dt.dot(dq))
        ddq_b = ddq

        # Расчет theta_P
        q1 = np.dot(a[0, :], [time**4, time**3, time**2, time, 1])
        q2 = np.dot(a[1, :], [time**4, time**3, time**2, time, 1])
        q3 = np.dot(a[2, :], [time**4, time**3, time**2, time, 1])
        q4 = np.dot(a[3, :], [time**4, time**3, time**2, time, 1])

        dq1 = np.dot(a[0, :], [4*time**3, 3*time**2, 2*time, 1, 0])
        dq2 = np.dot(a[1, :], [4*time**3, 3*time**2, 2*time, 1, 0])
        dq3 = np.dot(a[2, :], [4*time**3, 3*time**2, 2*time, 1, 0])
        dq4 = np.dot(a[3, :], [4*time**3, 3*time**2, 2*time, 1, 0])

        ddq1 = np.dot(a[0, :], [12*time**2, 6*time, 2, 0, 0])
        ddq2 = np.dot(a[1, :], [12*time**2, 6*time, 2, 0, 0])
        ddq3 = np.dot(a[2, :], [12*time**2, 6*time, 2, 0, 0])
        ddq4 = np.dot(a[3, :], [12*time**2, 6*time, 2, 0, 0])
        # print (f'q4_s = {q4}')
        # print (f'q_b = {q_b}')
        q = q_b + np.array([q1, q2, q3, q4])
        dq = dq_b + np.array([dq1, dq2, dq3, dq4])
        ddq = ddq_b + np.array([ddq1, ddq2, ddq3, ddq4])

    return q, dq, ddq