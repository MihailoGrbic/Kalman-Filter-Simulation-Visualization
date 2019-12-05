import numpy as np
import pygame
from matplotlib import pyplot as plt

WINDOW_DIM = (1280, 720)

class KalmanFilter:
    def __init__(self, A, B, H, Q, R, x_init, P_init = 1, name = ""):
        self.A = A
        self.B = B
        self.H = H
        self.Q = Q
        self.R = R
        self.x_hat_k = x_init
        self.P_k = P_init
        self.name = name
        
    def reset(self, x_init, P_init):
        self.x_hat_k = x_init
        self.P_k = P_init

    def step(self, u_k, z_k):
        x_hat_k_minus_1 = self.x_hat_k
        P_k_minus_1 = self.P_k

        x_hat_k = self.A @ x_hat_k_minus_1 #+ self.B @ u_k
        P_k = self.A @ P_k_minus_1 @ np.transpose(self.A) + self.Q 
        
        K_k = P_k @ np.transpose(self.H) @ np.linalg.inv(self.H @ P_k @ np.transpose(self.H) + self.R)
        x_hat_k = x_hat_k + K_k @ (z_k - self.H @ x_hat_k)
        P_k = (P_k - K_k @ self.H @ P_k)

        self.x_hat_k = x_hat_k
        self.P_k = P_k
        return x_hat_k, P_k

def simulation_step(x_prev):
    pass


pygame.init()
win = pygame.display.set_mode(WINDOW_DIM)
pygame.display.set_caption("Constant velocity model")

pos = np.matrix([[0.0, 0.0]]).T
velocity = np.matrix([[200.0, 100.0]]).T

gt_crosses = []
meausure_crosses = []
P_through_time = []
error_through_time = []
measurement_error_through_time = []

dt = 0.001 # 1ms measurement delay


# represents egomotion of model (e.g. car with constant velocity)
# x_new = x_old + vx*dt ; y_new = y_new + vy*dt
A = np.matrix([[1.0, 0.0, dt, 0.0],
              [0.0, 1.0, 0.0, dt],
              [0.0, 0.0, 1.0, 0.0],
              [0.0, 0.0, 0.0, 1.0]]) 

# No control signal
B = np.zeros((4, 4))

# Observation represent velocity of object, (vx, vy)
H = np.matrix([[0.0, 0.0, 1.0, 0.0],
              [0.0, 0.0, 0.0, 1.0]])

# Process noise can be modeled as artifcial acceleration
# (due to external phenomenoms for example)
# as such we can represent process noise covariance as:
# Q = G * Gt * sigma_v
# where acceleration process noise, which can be assumed for vehicle to be 8.8m/s2
#       and G is matrix representing egomotion of constant acceleration for model
G = np.matrix([[0.5*dt**2],
               [0.5*dt**2],
               [dt],
               [dt]])
sigma_v = 8.8
Q = G * np.transpose(G) * sigma_v**2
R = np.eye(2) * 10

x_init = np.matrix([[0.0, 0.0, 0.0, 0.0]]).T # x, y, vx, vy
kalman_filter = KalmanFilter(A, B, H, Q, R, x_init, np.eye(4) * 1000)

run = True
step_index = 0
while run:
    pygame.time.delay(1)

    # Listen for interrupts
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

    # Update the simulation
    pos += velocity*dt

    # Take the measurement
    measured_velocity = velocity + np.random.normal(0, 50.0, velocity.shape)

    # Update the filter
    kalman_state, P = kalman_filter.step(np.zeros(4), measured_velocity)
    kalman_position = np.matrix([[kalman_state[0, 0], kalman_state[1, 0]]]).T

    P_through_time.append(np.average(P))
    error_through_time.append(np.linalg.norm(kalman_position - pos))
    # measurement_error_through_time.append(np.linalg.norm(measured_pos - pos))
    bounds = (pos > WINDOW_DIM).flatten().tolist()[0]
    if any(bounds):
        run = False
        plt.plot(P_through_time)
        plt.show()
        plt.plot(measurement_error_through_time)
        plt.show()
        plt.plot(error_through_time)
        plt.show()


    # Draw
    win.fill((255,255,255))
    pygame.draw.circle(win, (255,0,0), pos.astype(int), 10)   
    # pygame.draw.circle(win, (0,255,0), measured_pos.astype(int), 10)
    pygame.draw.circle(win, (0,0,255), kalman_position.astype(int), 10) 

    if step_index % 25 == 0: gt_crosses.append(pos.astype(int))
    # if step_index % 5 == 0: meausure_crosses.append(measured_pos.astype(int).tolist())

    for i in range(len(gt_crosses)): pygame.draw.circle(win, (255,0,0), gt_crosses[i], 5)
    for i in range(len(meausure_crosses)): pygame.draw.circle(win, (0,255,0), meausure_crosses[i], 5)  

    pygame.display.update() 
    step_index += 1

pygame.quit()
