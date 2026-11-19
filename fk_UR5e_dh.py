
import numpy as np

# DH parameters (meters)
a = [0, -0.425, -0.39225, 0, 0, 0]
d = [0.089159, 0, 0, 0.10915, 0.09465, 0.0823]
alpha = [np.pi/2, 0, 0, np.pi/2, -np.pi/2, 0]


def DH(theta, d, a, alpha):

    c = np.cos(theta)
    s = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)

    A = np.array([
        [c, -s*ca,  s*sa, a*c],
        [s,  c*ca, -c*sa, a*s],
        [0,    sa,    ca,   d],
        [0,     0,     0,   1]
    ])

    return A


def FK(q):

    T = np.eye(4)

    for i in range(6):
        A = DH(q[i], d[i], a[i], alpha[i])
        T = T @ A

    return T


# Joint angles (radians)
q = [0, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0]

T = FK(q)

print("DH end-effector transformation:")
print(np.round(T, 4))

print("DH end-effector position:")
print(np.round(T[:3, 3], 4))

# EE Pose is [-0.492 -0.134  0.488] from MujoCO and 
# [-0.4869 -0.1091  0.4319] from analytical
