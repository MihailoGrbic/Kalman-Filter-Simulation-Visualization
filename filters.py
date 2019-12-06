import numpy as np

class KalmanFilter:
    def __init__(self, A, B, H, Q, R, x_start, P_start):
        self.A = A
        self.B = B
        self.H = H
        self.Q = Q
        self.R = R
        self.x_hat_k = x_start
        self.P_k = P_start

    def step(self, z_k, u_k):
        x_hat_k_minus_1 = self.x_hat_k
        P_k_minus_1 = self.P_k

        x_hat_k = self.A @ x_hat_k_minus_1 + self.B @ u_k
        P_k = self.A @ P_k_minus_1 @ np.transpose(self.A) + self.Q 

        K_k = P_k @ np.transpose(self.H) @ np.linalg.inv(self.H @ P_k @ np.transpose(self.H) + self.R)
        x_hat_k = x_hat_k + K_k @ (z_k - self.H @ x_hat_k)
        P_k = (P_k - K_k @ self.H @ P_k)

        self.x_hat_k = x_hat_k
        self.P_k = P_k
        return x_hat_k, P_k

class Simple2DModel(KalmanFilter):
    def __init__(self, Q_scalar = 1, R_scalar = 1, x_start = np.zeros((2, 1)), P_start = np.eye(2) * 10):
        self.A = np.eye(2)
        self.B = np.eye(2)
        self.H = np.eye(2)
        self.Q = np.eye(2) * Q_scalar
        self.R = np.eye(2) * R_scalar
        self.x_hat_k = x_start
        self.P_k = P_start