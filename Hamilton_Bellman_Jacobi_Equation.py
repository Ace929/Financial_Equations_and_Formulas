import numpy as np

def hamilton_bellman_jacobi(A, B, Q, R, x0, T, N):
    # Discretize the time horizon
    delta_t = T/N
    t = np.linspace(0, T, N+1)
    
    # Initialize the value function and control inputs
    V = np.zeros((N+1, 1))
    u = np.zeros((N, 1))
    
    # Backward pass to calculate the value function and control inputs
    for k in range(N-1, -1, -1):
        Q_k = Q + np.dot(A.T, np.dot(V[k+1], A))
        R_k = R + np.dot(B.T, np.dot(V[k+1], B)) * delta_t
        P = np.linalg.inv(R_k + np.dot(B.T, np.dot(np.linalg.inv(Q_k), B)))
        K = np.dot(np.linalg.inv(R_k), np.dot(B.T, np.linalg.inv(Q_k)))
        V[k] = np.dot(A.T, np.dot(V[k+1], A)) + np.dot(K.T, np.dot(R_k, K)) + Q_k
        u[k] = -np.dot(P, np.dot(B.T, np.linalg.inv(Q_k)))
    
    # Forward pass to simulate the system dynamics
    x = np.zeros((N+1, 1))
    x[0] = x0
    for k in range(N):
        x[k+1] = np.dot(A, x[k]) + np.dot(B, u[k])
    
    return x, u, V
