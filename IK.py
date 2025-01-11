def IK(x, y, z, l1, l2, l3, l4):
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

