import numpy as np
import pygame


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

        x_hat_k = self.A @ x_hat_k_minus_1 + self.B @ u_k
        P_k = self.A @ P_k_minus_1 @ np.transpose(self.A) + self.Q 
        
        K_k = P_k @ np.transpose(self.H) @ np.linalg.inv(self.H @ P_k @ np.transpose(self.H) + self.R)
        x_hat_k = x_hat_k + K_k @ (z_k - self.H @ x_hat_k)
        P_k = (P_k - P_k @ K_k @ self.H)

        print(x_hat_k)
        print(P_k) 
        print()
        self.x_hat_k = x_hat_k
        self.P_k = P_k
        return x_hat_k, P_k

def simulation_step(x_prev):
    pass


pygame.init()
win = pygame.display.set_mode(WINDOW_DIM)
pygame.display.set_caption("Straight Line")

pos = np.array([0, WINDOW_DIM[1] / 2])
velocity = np.array([1, 0])

gt_crosses = []
meausure_crosses = []

A = np.eye(2)
B = np.eye(2)
H = np.eye(2)
Q = np.zeros((2, 2))
R = np.eye(2) * 10
x_init = pos
kalman_filter = KalmanFilter(A, B, H, Q, R, x_init, np.eye(2) * 1000) 

run = True
step_index = 0
while run:
    pygame.time.delay(10)

    # Listen for interrupts
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

    # Update the simulation
    pos += velocity

    # Take the measurement
    measured_pos = pos + np.random.normal(0, 50, pos.size)

    # Update the filter
    kalman_position = kalman_filter.step(velocity, measured_pos)[0]

    # Draw
    win.fill((255,255,255))
    pygame.draw.circle(win, (255,0,0), pos.astype(int), 10)   
    pygame.draw.circle(win, (0,255,0), measured_pos.astype(int), 10)  
    pygame.draw.circle(win, (0,0,255), kalman_position.astype(int), 10) 

    if step_index % 25 == 0: gt_crosses.append(pos.astype(int).tolist())
    if step_index % 5 == 0: meausure_crosses.append(measured_pos.astype(int).tolist())

    for i in range(len(gt_crosses)): pygame.draw.circle(win, (255,0,0), gt_crosses[i], 5)  
    for i in range(len(meausure_crosses)): pygame.draw.circle(win, (0,255,0), meausure_crosses[i], 5)  

    pygame.display.update() 
    step_index += 1

pygame.quit()