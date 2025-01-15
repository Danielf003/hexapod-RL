import numpy as np

def calc_Jinv(s1,s2,s3,s4,c1,c2,c3,c4,l1,l2,l3,l4):
    J_inv = np.array([
                [-s1 / (l1 * s1**2 + l1 * c1**2 + l2 * c1**2 * c2 + l2 * c2 * s1**2 + l3 * c1**2 * c2 * c3 + l3 * c2 * c3 * s1**2 - l3 * c1**2 * s2 * s3 - l3 * s1**2 * s2 * s3 + l4 * c1**2 * c2 * c3 * c4 + l4 * c2 * c3 * c4 * s1**2 - l4 * c1**2 * c2 * s3 * s4 - l4 * c1**2 * c3 * s2 * s4 - l4 * c1**2 * c4 * s2 * s3 - l4 * c2 * s1**2 * s3 * s4 - l4 * c3 * s1**2 * s2 * s4 - l4 * c4 * s1**2 * s2 * s3), c1 / (l1 * s1**2 + l1 * c1**2 + l2 * c1**2 * c2 + l2 * c2 * s1**2 + l3 * c1**2 * c2 * c3 + l3 * c2 * c3 * s1**2 - l3 * c1**2 * s2 * s3 - l3 * s1**2 * s2 * s3 + l4 * c1**2 * c2 * c3 * c4 + l4 * c2 * c3 * c4 * s1**2 - l4 * c1**2 * c2 * s3 * s4 - l4 * c1**2 * c3 * s2 * s4 - l4 * c1**2 * c4 * s2 * s3 - l4 * c2 * s1**2 * s3 * s4 - l4 * c3 * s1**2 * s2 * s4 - l4 * c4 * s1**2 * s2 * s3), 0, 0],
                [(c1 * c2 * c3 - c1 * s2 * s3) / (l2 * s3 * c1**2 * c2**2 + l2 * s3 * c1**2 * s2**2 + l2 * s3 * c2**2 * s1**2 + l2 * s3 * s1**2 * s2**2), -(s1 * s2 * s3 - c2 * c3 * s1) / (l2 * s3 * c1**2 * c2**2 + l2 * s3 * c1**2 * s2**2 + l2 * s3 * c2**2 * s1**2 + l2 * s3 * s1**2 * s2**2), (c2 * s3 + c3 * s2) / (l2 * s3 * c2**2 + l2 * s3 * s2**2), (l4 * s4 * c3**2 + l4 * s4 * s3**2) / (l2 * s1 * s3)],
                [-(l2 * c1 * c2 + l3 * c1 * c2 * c3 - l3 * c1 * s2 * s3) / (l2 * l3 * s3 * c1**2 * c2**2 + l2 * l3 * s3 * c1**2 * s2**2 + l2 * l3 * s3 * c2**2 * s1**2 + l2 * l3 * s3 * s1**2 * s2**2), -(l2 * c2 * s1 + l3 * c2 * c3 * s1 - l3 * s1 * s2 * s3) / (l2 * l3 * s3 * c1**2 * c2**2 + l2 * l3 * s3 * c1**2 * s2**2 + l2 * l3 * s3 * c2**2 * s1**2 + l2 * l3 * s3 * s1**2 * s2**2), -(l2 * s2 + l3 * c2 * s3 + l3 * c3 * s2) / (l2 * l3 * s3 * c2**2 + l2 * l3 * s3 * s2**2), -(l3 * l4 * s4 * c3**2 + l2 * l4 * s4 * c3 + l3 * l4 * s4 * s3**2 + l2 * l4 * c4 * s3) / (l2 * l3 * s1 * s3)],
                [(c1 * c2) / (l3 * s3 * c1**2 * c2**2 + l3 * s3 * c1**2 * s2**2 + l3 * s3 * c2**2 * s1**2 + l3 * s3 * s1**2 * s2**2), (c2 * s1) / (l3 * s3 * c1**2 * c2**2 + l3 * s3 * c1**2 * s2**2 + l3 * s3 * c2**2 * s1**2 + l3 * s3 * s1**2 * s2**2), s2 / (l3 * s3 * c2**2 + l3 * s3 * s2**2), (l3 * s3 + l4 * c3 * s4 + l4 * c4 * s3) / (l3 * s1 * s3)]
            ])
    return J_inv

def calc_Jdot(s1,s2,s3,s4,c1,c2,c3,c4,l1,l2,l3,l4,dq1,dq2,dq3,dq4):
    dJ_dt = np.array([
                [l4 * c4 * (c1 * s2 * s3 * dq1 + c2 * s1 * s3 * dq2 + c3 * s1 * s2 * dq2 + c2 * s1 * s3 * dq3 + c3 * s1 * s2 * dq3 - c1 * c2 * c3 * dq1) - l1 * c1 * dq1 + l4 * s4 * (c1 * c2 * s3 * dq1 + c1 * c3 * s2 * dq1 + c2 * c3 * s1 * dq2 + c2 * c3 * s1 * dq3 - s1 * s2 * s3 * dq2 - s1 * s2 * s3 * dq3) - l2 * c1 * c2 * dq1 + l2 * s1 * s2 * dq2 + l4 * c4 * (c2 * s1 * s3 + c3 * s1 * s2) * dq4 + l4 * s4 * (c2 * c3 * s1 - s1 * s2 * s3) * dq4 - l3 * c1 * c2 * c3 * dq1 + l3 * c1 * s2 * s3 * dq1 + l3 * c2 * s1 * s3 * dq2 + l3 * c3 * s1 * s2 * dq2 + l3 * c2 * s1 * s3 * dq3 + l3 * c3 * s1 * s2 * dq3, l4 * c4 * (c2 * s1 * s3 * dq1 + c3 * s1 * s2 * dq1 + c1 * s2 * s3 * dq2 + c1 * s2 * s3 * dq3 - c1 * c2 * c3 * dq2 - c1 * c2 * c3 * dq3) + l4 * s4 * (c2 * c3 * s1 * dq1 + c1 * c2 * s3 * dq2 + c1 * c3 * s2 * dq2 + c1 * c2 * s3 * dq3 + c1 * c3 * s2 * dq3 - s1 * s2 * s3 * dq1) - l2 * c1 * c2 * dq2 + l2 * s1 * s2 * dq1 - l4 * c4 * (c1 * c2 * c3 - c1 * s2 * s3) * dq4 + l4 * s4 * (c1 * c2 * s3 + c1 * c3 * s2) * dq4 - l3 * c1 * c2 * c3 * dq2 - l3 * c1 * c2 * c3 * dq3 + l3 * c2 * s1 * s3 * dq1 + l3 * c3 * s1 * s2 * dq1 + l3 * c1 * s2 * s3 * dq2 + l3 * c1 * s2 * s3 * dq3, l4 * c4 * (c2 * s1 * s3 * dq1 + c3 * s1 * s2 * dq1 + c1 * s2 * s3 * dq2 + c1 * s2 * s3 * dq3 - c1 * c2 * c3 * dq2 - c1 * c2 * c3 * dq3) + l4 * s4 * (c2 * c3 * s1 * dq1 + c1 * c2 * s3 * dq2 + c1 * c3 * s2 * dq2 + c1 * c2 * s3 * dq3 + c1 * c3 * s2 * dq3 - s1 * s2 * s3 * dq1) - l4 * c4 * (c1 * c2 * c3 - c1 * s2 * s3) * dq4 + l4 * s4 * (c1 * c2 * s3 + c1 * c3 * s2) * dq4 - l3 * c1 * c2 * c3 * dq2 - l3 * c1 * c2 * c3 * dq3 + l3 * c2 * s1 * s3 * dq1 + l3 * c3 * s1 * s2 * dq1 + l3 * c1 * s2 * s3 * dq2 + l3 * c1 * s2 * s3 * dq3, l4 * c4 * (c2 * s1 * s3 * dq1 + c3 * s1 * s2 * dq1 + c1 * s2 * s3 * dq2 + c1 * s2 * s3 * dq3 - c1 * c2 * c3 * dq2 - c1 * c2 * c3 * dq3) + l4 * s4 * (c2 * c3 * s1 * dq1 + c1 * c2 * s3 * dq2 + c1 * c3 * s2 * dq2 + c1 * c2 * s3 * dq3 + c1 * c3 * s2 * dq3 - s1 * s2 * s3 * dq1) - l4 * c4 * (c1 * c2 * c3 - c1 * s2 * s3) * dq4 + l4 * s4 * (c1 * c2 * s3 + c1 * c3 * s2) * dq4],
                [l4 * s4 * (c2 * s1 * s3 * dq1 + c3 * s1 * s2 * dq1 + c1 * s2 * s3 * dq2 + c1 * s2 * s3 * dq3 - c1 * c2 * c3 * dq2 - c1 * c2 * c3 * dq3) - l4 * c4 * (c2 * c3 * s1 * dq1 + c1 * c2 * s3 * dq2 + c1 * c3 * s2 * dq2 + c1 * c2 * s3 * dq3 + c1 * c3 * s2 * dq3 - s1 * s2 * s3 * dq1) - l1 * s1 * dq1 - l2 * c2 * s1 * dq1 - l2 * c1 * s2 * dq2 - l4 * c4 * (c1 * c2 * s3 + c1 * c3 * s2) * dq4 - l4 * s4 * (c1 * c2 * c3 - c1 * s2 * s3) * dq4 - l3 * c2 * c3 * s1 * dq1 - l3 * c1 * c2 * s3 * dq2 - l3 * c1 * c3 * s2 * dq2 - l3 * c1 * c2 * s3 * dq3 - l3 * c1 * c3 * s2 * dq3 + l3 * s1 * s2 * s3 * dq1, l4 * s4 * (c1 * s2 * s3 * dq1 + c2 * s1 * s3 * dq2 + c3 * s1 * s2 * dq2 + c2 * s1 * s3 * dq3 + c3 * s1 * s2 * dq3 - c1 * c2 * c3 * dq1) - l4 * c4 * (c1 * c2 * s3 * dq1 + c1 * c3 * s2 * dq1 + c2 * c3 * s1 * dq2 + c2 * c3 * s1 * dq3 - s1 * s2 * s3 * dq2 - s1 * s2 * s3 * dq3) - l2 * c1 * s2 * dq1 - l2 * c2 * s1 * dq2 - l4 * c4 * (c2 * c3 * s1 - s1 * s2 * s3) * dq4 + l4 * s4 * (c2 * s1 * s3 + c3 * s1 * s2) * dq4 - l3 * c1 * c2 * s3 * dq1 - l3 * c1 * c3 * s2 * dq1 - l3 * c2 * c3 * s1 * dq2 - l3 * c2 * c3 * s1 * dq3 + l3 * s1 * s2 * s3 * dq2 + l3 * s1 * s2 * s3 * dq3, l4 * s4 * (c1 * s2 * s3 * dq1 + c2 * s1 * s3 * dq2 + c3 * s1 * s2 * dq2 + c2 * s1 * s3 * dq3 + c3 * s1 * s2 * dq3 - c1 * c2 * c3 * dq1) - l4 * c4 * (c1 * c2 * s3 * dq1 + c1 * c3 * s2 * dq1 + c2 * c3 * s1 * dq2 + c2 * c3 * s1 * dq3 - s1 * s2 * s3 * dq2 - s1 * s2 * s3 * dq3) - l4 * c4 * (c2 * c3 * s1 - s1 * s2 * s3) * dq4 + l4 * s4 * (c2 * s1 * s3 + c3 * s1 * s2) * dq4 - l3 * c1 * c2 * s3 * dq1 - l3 * c1 * c3 * s2 * dq1 - l3 * c2 * c3 * s1 * dq2 - l3 * c2 * c3 * s1 * dq3 + l3 * s1 * s2 * s3 * dq2 + l3 * s1 * s2 * s3 * dq3, l4 * s4 * (c1 * s2 * s3 * dq1 + c2 * s1 * s3 * dq2 + c3 * s1 * s2 * dq2 + c2 * s1 * s3 * dq3 + c3 * s1 * s2 * dq3 - c1 * c2 * c3 * dq1) - l4 * c4 * (c1 * c2 * s3 * dq1 + c1 * c3 * s2 * dq1 + c2 * c3 * s1 * dq2 + c2 * c3 * s1 * dq3 - s1 * s2 * s3 * dq2 - s1 * s2 * s3 * dq3) - l4 * c4 * (c2 * c3 * s1 - s1 * s2 * s3) * dq4 + l4 * s4 * (c2 * s1 * s3 + c3 * s1 * s2) * dq4],
                [0, l4 * s4 * (s2 * s3 - c2 * c3) * dq4 - l4 * s4 * (c2 * c3 * dq2 + c2 * c3 * dq3 - s2 * s3 * dq2 - s2 * s3 * dq3) - l2 * s2 * dq2 - l3 * c2 * s3 * dq2 - l3 * c3 * s2 * dq2 - l3 * c2 * s3 * dq3 - l3 * c3 * s2 * dq3 - l4 * c4 * (c2 * s3 + c3 * s2) * dq4 - l4 * c4 * (c2 * s3 * dq2 + c3 * s2 * dq2 + c2 * s3 * dq3 + c3 * s2 * dq3), l4 * s4 * (s2 * s3 - c2 * c3) * dq4 - l4 * s4 * (c2 * c3 * dq2 + c2 * c3 * dq3 - s2 * s3 * dq2 - s2 * s3 * dq3) - l3 * c2 * s3 * dq2 - l3 * c3 * s2 * dq2 - l3 * c2 * s3 * dq3 - l3 * c3 * s2 * dq3 - l4 * c4 * (c2 * s3 + c3 * s2) * dq4 - l4 * c4 * (c2 * s3 * dq2 + c3 * s2 * dq2 + c2 * s3 * dq3 + c3 * s2 * dq3), l4 * s4 * (s2 * s3 - c2 * c3) * dq4 - l4 * s4 * (c2 * c3 * dq2 + c2 * c3 * dq3 - s2 * s3 * dq2 - s2 * s3 * dq3) - l4 * c4 * (c2 * s3 + c3 * s2) * dq4 - l4 * c4 * (c2 * s3 * dq2 + c3 * s2 * dq2 + c2 * s3 * dq3 + c3 * s2 * dq3)],
                [0, c1 * dq1, c1 * dq1, c1 * dq1]
            ])
    return dJ_dt

def fk(q1, q2, q3, q4, L1, L2, L3, L4):
    c2 = np.cos(q2)
    c23 = np.cos(q2+q3)
    c234 = np.cos(q2+q3+q4)
    Lc = (L1 + L2*c2 + L3*c23 + L4*c234)
    x = np.cos(q1)*Lc
    y = np.sin(q1)*Lc
    z = L2*np.sin(q2) + L3*np.sin(q2+q3) + L4*np.sin(q2+q3+q4)

    pos = np.array([x, y, z])

    return pos

def ik(x, y, z, l1, l2, l3, l4):
    # Решение обратной задачи кинематики

    import numpy as np

    z = z + l4
    if x > 0:
        theta_1 = np.arctan(y/x)
    elif x == 0:
        theta_1 = np.pi/2
    elif x < 0:
        theta_1 = np.pi - np.arctan(y/-x)

    d = np.sqrt((np.sqrt(x**2 + y**2) - l1)**2 + z**2)
    cos_b = (d**2 + l2**2 - l3**2)/(2*l2*d)
    cos_gamma = (l2**2 + l3**2 - d**2)/(2*l2*l3)
    sin_a = -z/d

    theta_2 = np.arccos(cos_b) - np.arcsin(sin_a)
    theta_3 = np.pi - np.arccos(cos_gamma)
    theta_4 = 2*np.pi - (np.pi - (np.pi/2 - theta_2) - (2*np.pi - (2*np.pi - theta_3)))

    return np.array([theta_1, theta_2, theta_3, theta_4])

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

def thetas_traj(t, T_f, T_b, delta_T, C_x, C_y, C_z, a, J_inv_func=None, dJ_dt_func=None):
    #-----------------------------------------------------------

    # T_b - время движения по параболе в фазе опоры (если T_b = 0, то энд-эффектор движется просто по прямой)

    l1 = 0.3  # m length of the first link
    l2 = 0.848   # m length of the second link
    l3 = 1.221   # m length of the third link
    l4 = 0.6  # m length of the fourth link

    time = (t + delta_T) % (2 * T_f)

    #-----------------------------------------------------------
    # Построение траектории

    if time <= T_f:  # Фаза опоры thet_S
        p_s = np.array([time**2, time, 1, 0, 0, 0, 0, 0])
        v_s = np.array([2*time, 1, 0, 0, 0, 0, 0, 0])
        if 0 <= time <= T_b:
            p_s = np.array([time**2, time, 1, 0, 0, 0, 0, 0])
            v_s = np.array([2*time, 1, 0, 0, 0, 0, 0, 0])
            # a_s = np.array([2, 0, 0, 0, 0, 0, 0, 0])
        elif T_b < time <= T_f - T_b:
            p_s = np.array([0, 0, 0, time, 1, 0, 0, 0])
            v_s = np.array([0, 0, 0, 1, 0, 0, 0, 0])
            # a_s = np.zeros(8)
        elif time > T_f - T_b:
            p_s = np.array([0, 0, 0, 0, 0, time**2, time, 1])
            v_s = np.array([0, 0, 0, 0, 0, 2*time, 1, 0])
            # a_s = np.array([0, 0, 0, 0, 0, 2, 0, 0])

        p_x_s = p_s.dot(C_x)
        p_y_s = p_s.dot(C_y)
        p_z_s = p_s.dot(C_z)
        v_x_s = v_s.dot(C_x)
        v_y_s = v_s.dot(C_y)
        v_z_s = v_s.dot(C_z)
        # a_x_s = a_s.dot(C_x)
        # a_y_s = a_s.dot(C_y)
        # a_z_s = a_s.dot(C_z)
        # a_z_s = a_s.dot(C_z)

        # Решение обратной задачи кинематики

        q = ik(p_x_s, p_y_s, p_z_s, l1, l2, l3, l4)

        s1, s2, s3, s4 = np.sin(q)
        c1, c2, c3, c4 = np.cos(q)

        # Якобиан
        if J_inv_func is None:
            J_inv = calc_Jinv(s1,s2,s3,s4,c1,c2,c3,c4,l1,l2,l3,l4)
        else:
            J_inv = J_inv_func(q)
        dq = J_inv.dot(np.array([v_x_s, v_y_s, v_z_s, 0]))
        # dq1, dq2, dq3, dq4 = dq
        # if dJ_dt_func is None:
        #     dJ_dt = calc_Jdot(s1,s2,s3,s4,c1,c2,c3,c4,l1,l2,l3,l4,dq1,dq2,dq3,dq4)
        # else:
        #     dJ_dt = dJ_dt_func(q, dq)

        
        # ddq = J_inv.dot(np.array([a_x_s, a_y_s, a_z_s, 0]) - dJ_dt.dot(dq))
        ddq = np.zeros(4)

    else:
        # Фаза перемещения thet_B (обратная thet_S) и thet_P

        # Расчет thet_B
        time_2 = 2 * T_f - time
        p_b = np.array([time_2**2, time_2, 1, 0, 0, 0, 0, 0])
        v_b = np.array([2*time_2, 1, 0, 0, 0, 0, 0, 0])
        if 0 <= time_2 <= T_b:
            p_b = np.array([time_2**2, time_2, 1, 0, 0, 0, 0, 0])
            v_b = np.array([2*time_2, 1, 0, 0, 0, 0, 0, 0])
            # a_b = np.array([2, 0, 0, 0, 0, 0, 0, 0])

        elif T_b < time_2 <= T_f - T_b:
            p_b = np.array([0, 0, 0, time_2, 1, 0, 0, 0])
            v_b = np.array([0, 0, 0, 1, 0, 0, 0, 0])
            # a_b = np.zeros(8)

        elif T_f - T_b < time_2 <= T_f + 1:
            p_b = np.array([0, 0, 0, 0, 0, time_2**2, time_2, 1])
            v_b = np.array([0, 0, 0, 0, 0, 2*time_2, 1, 0])
            # a_b = np.array([0, 0, 0, 0, 0, 2, 0, 0])

        p_x_b = p_b.dot(C_x)
        p_y_b = p_b.dot(C_y)
        p_z_b = p_b.dot(C_z)
        v_x_b = v_b.dot(C_x)
        v_y_b = v_b.dot(C_y)
        v_z_b = v_b.dot(C_z)
        # a_x_b = a_b.dot(C_x)
        # a_y_b = a_b.dot(C_y)
        # a_z_b = a_b.dot(C_z)
        # a_z_b = a_b.dot(C_z)

        # Решение обратной задачи кинематики

        q = ik(p_x_b, p_y_b, p_z_b, l1, l2, l3, l4)
        q_b = q

        s1, s2, s3, s4 = np.sin(q)
        c1, c2, c3, c4 = np.cos(q)

        # Якобиан
        if J_inv_func is None:
            J_inv = calc_Jinv(s1,s2,s3,s4,c1,c2,c3,c4,l1,l2,l3,l4)
        else:
            J_inv = J_inv_func(q)
        dq = J_inv.dot(np.array([v_x_b, v_y_b, v_z_b, 0]))
        dq1, dq2, dq3, dq4 = dq
        dq_b = dq
        # if dJ_dt_func is None:
        #     dJ_dt = calc_Jdot(s1,s2,s3,s4,c1,c2,c3,c4,l1,l2,l3,l4,dq1,dq2,dq3,dq4)
        # else:
        #     dJ_dt = dJ_dt_func(q, dq)
        
        # ddq = J_inv.dot(np.array([a_x_b, a_y_b, a_z_b, 0]) - dJ_dt.dot(dq))
        # ddq_b = ddq

        # Расчет theta_P
        qt = [time**4, time**3, time**2, time, 1]
        q1 = np.dot(a[0, :], qt)
        q2 = np.dot(a[1, :], qt)
        q3 = np.dot(a[2, :], qt)
        q4 = np.dot(a[3, :], qt)

        dqt = [4*time**3, 3*time**2, 2*time, 1, 0]
        dq1 = np.dot(a[0, :], dqt)
        dq2 = np.dot(a[1, :], dqt)
        dq3 = np.dot(a[2, :], dqt)
        dq4 = np.dot(a[3, :], dqt)

        # ddqt = [12*time**2, 6*time, 2, 0, 0]
        # ddq1 = np.dot(a[0, :], ddqt)
        # ddq2 = np.dot(a[1, :], ddqt)
        # ddq3 = np.dot(a[2, :], ddqt)
        # ddq4 = np.dot(a[3, :], ddqt)
        # print (f'q4_s = {q4}')
        # print (f'q_b = {q_b}')
        q = q_b + np.array([q1, q2, q3, q4])
        dq = dq_b + np.array([dq1, dq2, dq3, dq4])
        # ddq = ddq_b + np.array([ddq1, ddq2, ddq3, ddq4])
        ddq = np.zeros(4)

    return q, dq, ddq
