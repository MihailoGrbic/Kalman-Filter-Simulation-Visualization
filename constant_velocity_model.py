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
        return x_hat_k, P_k, K_k

def simulation_step(x_prev):
    pass


def draw_graph(data, labels, title, xlabel, ylabel, change_ticks = False):
    for i in range(len(data)):
        plt.plot(data[i], label = labels[i])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if change_ticks:
        plt.xticks(np.arange(0, len(data), 50.0))
        plt.yticks(np.arange(0, max(data), 20.0))
    plt.grid()
    plt.legend(loc='best', prop={'size':20})
    plt.show()


pygame.init()
win = pygame.display.set_mode(WINDOW_DIM)
pygame.display.set_caption("Constant velocity model")

pos = np.matrix([[0.0, 0.0]]).T
velocity = np.matrix([[20.0, 10.0]]).T

gt_crosses = []
meausure_crosses = []
P_through_time = []
error_through_time = []
measurement_error_through_time = []

dt = 0.1 # 1ms measurement delay


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
R = np.eye(2) * 100

x_init = np.matrix([[0.0, 0.0, 0.0, 0.0]]).T # x, y, vx, vy
kalman_filter = KalmanFilter(A, B, H, Q, R, x_init, np.eye(4) * 1000)

run = True
step_index = 0


xt = np.array([0])
yt = np.array([0])
vxt = np.array([0])
vyt = np.array([0])
mvxt = np.array([])
mvyt = np.array([])
Kx = np.array([])
Ky = np.array([])
Kvx = np.array([])
Kvy = np.array([])
Px = np.array([1000.0])
Py = np.array([1000.0])
Pvx = np.array([1000.0])
Pvy = np.array([1000.0])

from sympy import Symbol, Matrix
from sympy.interactive import printing
printing.init_printing()
dts = Symbol('dt')
Qs = Matrix([[0.5*dts**2],[0.5*dts**2],[dts],[dts]])
Qs*Qs.T


fig = plt.figure(figsize=(6, 6))
im = plt.imshow(Q, interpolation="none", cmap=plt.get_cmap('binary'))
plt.title('Process Noise Covariance Matrix $P$')
ylocs, ylabels = plt.yticks()
# set the locations of the yticks
plt.yticks(np.arange(7))
# set the locations and labels of the yticks
plt.yticks(np.arange(6),('$x$', '$y$', '$\dot x$', '$\dot y$'), fontsize=22)

xlocs, xlabels = plt.xticks()
# set the locations of the yticks
plt.xticks(np.arange(7))
# set the locations and labels of the yticks
plt.xticks(np.arange(6),('$x$', '$y$', '$\dot x$', '$\dot y$'), fontsize=22)

plt.xlim([-0.5,3.5])
plt.ylim([3.5, -0.5])

from mpl_toolkits.axes_grid1 import make_axes_locatable
divider = make_axes_locatable(plt.gca())
cax = divider.append_axes("right", "5%", pad="3%")
plt.colorbar(im, cax=cax);
plt.show()

cnt = 0
while run:
    pygame.time.delay(1)
    cnt = cnt + 1
    # Listen for interrupts
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

    # Update the simulation
    pos += velocity*dt

    # Take the measurement
    # measured_velocity = velocity + np.random.normal(0, 1.0, velocity.shape)
    measured_velocity = velocity + np.random.randn(len(velocity))
    # Update the filter
    kalman_state, P, K = kalman_filter.step(np.zeros(4), measured_velocity)
    kalman_position = np.matrix([[kalman_state[0, 0], kalman_state[1, 0]]]).T

    # save for plotting
    xt = np.append(xt, float(kalman_state[0, 0]))
    yt = np.append(yt, float(kalman_state[1, 0]))
    vxt = np.append(vxt, float(kalman_state[2, 0]))
    vyt = np.append(vyt, float(kalman_state[3, 0]))
    Px = np.append(Px, float(P[0,0]))
    Py = np.append(Py, float(P[1,1]))
    Pvx = np.append(Pvx, float(P[2,2]))
    Pvy = np.append(Pvy, float(P[3,3]))
    Kx = np.append(Kx, [float(K[0,0])])
    Ky = np.append(Ky, float(K[1,0]))
    Kvx = np.append(Kvx, float(K[2,0]))
    Kvy = np.append(Kvy, float(K[3,0]))
    mvxt = np.append(mvxt, measured_velocity[0,0])
    mvyt = np.append(mvyt, measured_velocity[1,0])

    P_through_time.append(np.average(P))
    error_through_time.append(np.linalg.norm(kalman_position - pos))
    measurement_error_through_time.append(np.linalg.norm(measured_velocity - velocity))
    bounds = (pos > WINDOW_DIM).flatten().tolist()[0]
    if cnt > 800:
        run = False
        # plt.plot(measurement_error_through_time)
        # plt.show()
        # plt.plot(error_through_time)
        # plt.show()
        print(xt.shape, Px.shape, Kx.shape, kalman_state.shape)
        gtvx = np.full([len(vxt)],20.0)
        gtvy = np.full([len(vyt)], 10.0)
        draw_graph([vxt, vyt, gtvx, gtvy], ["Vx estimated", "Vy estimated", "Vx true", "Vy true"], "Ground truth vs estimated velocity", "Step", "Velocity")
        draw_graph([Kx, Ky, Kvx, Kvy], ["Kalman gain for x", "Kalman gain for y", "Kalman gain for Vx", "Kalman gain for Vy"], "Kalman gain trough steps", "Step", "Gain")


    # Draw
    win.fill((255,255,255))
    pygame.draw.circle(win, (255,0,0), pos.astype(int), 10)   
    # pygame.draw.circle(win, (0,255,0), measured_pos.astype(int), 10)
    pygame.draw.circle(win, (0,0,255), kalman_position.astype(int), 10) 

    if step_index % 5 == 0: gt_crosses.append(pos.astype(int))
    # if step_index % 5 == 0: meausure_crosses.append(measured_pos.astype(int).tolist())

    for i in range(len(gt_crosses)): pygame.draw.circle(win, (255,0,0), gt_crosses[i], 5)
    for i in range(len(meausure_crosses)): pygame.draw.circle(win, (0,255,0), meausure_crosses[i], 5)  

    pygame.display.update() 
    step_index += 1

pygame.quit()
