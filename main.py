import numpy as np
from matplotlib import pyplot as plt
import pygame
pygame.init()


from filters import KalmanFilter, Simple2DModel
import simulations

WINDOW_DIM = (1280, 720)
CIRCLE_RADIUS = 300
TIMESTEP = 1
MEASUREMENT_NOISE_VAR = 50
P_START = np.eye(2) * 10000
R_model = 100
GT_CROSS_PERIOD = 25
MEASURE_CROSS_PERIOD = 3
KALMAN_CROSS_PERIOD = 20

def reset(only_crosses = False):
    ''' Delete all crosses, reset saved data for graphs'''
    global gt_crosses, meausure_crosses, kalman_crosses, P_through_time, measurement_error_through_time, kalman_error_through_time
    gt_crosses = []
    meausure_crosses = []
    kalman_crosses = []
    if not only_crosses:
        P_through_time = []
        measurement_error_through_time = []
        kalman_error_through_time = []

def draw_cross(pos, color, size = 8):
    pygame.draw.line(win, color, [pos[0] - size/2, pos[1] - size/2], [pos[0] + size/2, pos[1] + size/2], int(size/3))
    pygame.draw.line(win, color, [pos[0] - size/2, pos[1] + size/2], [pos[0] + size/2, pos[1] - size/2], int(size/3))

def draw_graph(data, title, xlabel, ylabel, change_ticks = False):
    plt.plot(data)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if change_ticks:
        plt.xticks(np.arange(0, len(data), 50))
        plt.yticks(np.arange(0, max(data), 20))
    plt.grid()
    plt.show()

def show_results():
    global P_through_time, measurement_error_through_time, kalman_error_through_time

    draw_graph(P_through_time, "Kalman P", "Time[ms]", "")
    draw_graph(measurement_error_through_time, "Measurement error", "Time[ms]", "Pixels", True)
    draw_graph(kalman_error_through_time, "Kalman estimation error", "Time[ms]", "Pixels", True)

def write_text(text, color, position, font = pygame.font.Font('freesansbold.ttf', 36)):
    global win
    text = font.render(text, True, color)
    textRect = text.get_rect()   
    textRect.center = position 
    win.blit(text, textRect)


win = pygame.display.set_mode(WINDOW_DIM)
pygame.display.set_caption("Kalman Filter Simulations")

# Save past positions, measurements, and Kalman filter predictions to lists, to be drawn as crosses
gt_crosses = []
meausure_crosses = []
kalman_crosses = []

# Save P, measurement error, and kalman estimation error through time, to be presented in a graph
P_through_time = []
measurement_error_through_time = []
kalman_error_through_time = []

run = True
state = "Main Menu"
step_index = 0
while run:
    pygame.time.delay(TIMESTEP)

    # Listen for interrupts
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

    # Choosing a state, will be changed in the future
    keys = pygame.key.get_pressed()
    if keys[pygame.K_a]:
        state = "LineGT Simulation"
        reset()
        simulation = simulations.LineGT(TIMESTEP, MEASUREMENT_NOISE_VAR)
        kalman_filter = Simple2DModel(0, R_model, P_start=P_START) 

    elif keys[pygame.K_s]:
        state = "CircleGT Simulation"
        reset()
        simulation = simulations.CircleGT(TIMESTEP, MEASUREMENT_NOISE_VAR)
        kalman_filter = Simple2DModel(0, R_model, P_start=P_START) 
        
    elif keys[pygame.K_d]:
        state = "LineWControl Simulation"
        reset()
        simulation = simulations.LineWControl(TIMESTEP, MEASUREMENT_NOISE_VAR)
        kalman_filter = Simple2DModel(0, R_model, P_start=P_START)
        kalman_position = np.zeros((2, 1))

    elif keys[pygame.K_f]:
        state = "CircleWControl Simulation"
        reset()
        simulation = simulations.CircleWControl(TIMESTEP, MEASUREMENT_NOISE_VAR)
        kalman_filter = Simple2DModel(0, R_model, P_start=P_START) 
        kalman_position = np.zeros((2, 1))

    elif keys[pygame.K_c]:
        state = "Credits"
    elif keys[pygame.K_r]:  # Clear screen key
        reset(only_crosses=True)
    elif keys[pygame.K_b]:  # Back to Main Menu key
        if "Simulation" in state:
            show_results()
            reset()
        state = "Main Menu"


    # Drawing
    win.fill((255,255,255))
    if state == "Main Menu":
        write_text('Kalman Filter Simulation and Visualization', (0,0,0), (WINDOW_DIM[0] // 2, WINDOW_DIM[1] // 2), pygame.font.Font('freesansbold.ttf', 45))
        write_text('Start simulation: a,s,d,f', (0,0,0), (WINDOW_DIM[0] // 2, WINDOW_DIM[1] // 2 + 70))
        write_text('During simulation: r, b', (0,0,0), (WINDOW_DIM[0] // 2, WINDOW_DIM[1] // 2 + 120))
        write_text('Credits: c', (0,0,0), (WINDOW_DIM[0] // 2, WINDOW_DIM[1] // 2 + 170))
 
    elif state == "Credits":
        write_text('Jelena Ristic', (255,0,0), (WINDOW_DIM[0] // 2, WINDOW_DIM[1] // 2 - 50))
        write_text('Stefan Stepanovic', (0,255,0), (WINDOW_DIM[0] // 2, WINDOW_DIM[1] // 2))
        write_text('Mihailo Grbic', (0,0,255), (WINDOW_DIM[0] // 2, WINDOW_DIM[1] // 2 + 50))

    else:
        # Perform one step of the simulation
        if "Control" in state:
            gt_pos, measured_pos, velocity = simulation.step(kalman_position)
        else:
            gt_pos, measured_pos, velocity = simulation.step()

        if state == "CircleWControl Simulation":
            pygame.draw.circle(win, (0,0,0), simulation.tangent_point.astype(int), 10) 

        # Perform one Kalman filter step
        kalman_position, P = kalman_filter.step(measured_pos, velocity)

        # Draw current GT, measured, and Kalman filter estimated positions
        pygame.draw.circle(win, (255,0,0), gt_pos.astype(int), 10)   
        pygame.draw.circle(win, (0,255,0), measured_pos.astype(int), 10)  
        pygame.draw.circle(win, (0,0,255), kalman_position.astype(int), 10)

        # Draw paths
        if "Line" in state:
            pygame.draw.line(win, (200,200,200), [0, simulation.line_y], [WINDOW_DIM[0], simulation.line_y], 5)
        if "Circle" in state:
            pygame.draw.circle(win, (200,200,200), (simulation.circle_center), simulation.circle_radius, 5) 

        print(P)
        # Save current P, measurement error, and kalman estimation error through time, to be presented in a graph
        P_through_time.append(np.average(P))
        measurement_error_through_time.append(np.linalg.norm(measured_pos - gt_pos))
        kalman_error_through_time.append(np.linalg.norm(kalman_position - gt_pos))
        
        # Save current GT, measured, and Kalman filter estimated positions as crosses
        if step_index % GT_CROSS_PERIOD == 0: gt_crosses.append(gt_pos.astype(int).flatten().tolist())
        if step_index % MEASURE_CROSS_PERIOD == 0: meausure_crosses.append(measured_pos.astype(int).flatten().tolist())
        if step_index % KALMAN_CROSS_PERIOD == 0: kalman_crosses.append(kalman_position.astype(int).flatten().tolist())

        # Draw crosses
        for i in range(len(gt_crosses)): draw_cross(gt_crosses[i], (255,0,0))
        for i in range(len(meausure_crosses)): draw_cross(meausure_crosses[i], (0,255,0)) 
        for i in range(len(kalman_crosses)): draw_cross(kalman_crosses[i], (0,0,255))

        if state == "LineGT Simulation" or state == "LineWControl Simulation":
            if (gt_pos > np.array([WINDOW_DIM]).T).any():
                state = "Main Menu"
                show_results()
                reset()

    pygame.display.update() 
    step_index += 1

