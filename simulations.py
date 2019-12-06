import math

import numpy as np
import pygame

def find_tangent_point(outside_point, circle_center, circle_radius):
    distance_from_circle = np.linalg.norm(outside_point - circle_center)
    point_angle_to_circle = math.atan2(outside_point[1] - circle_center[1], outside_point[0] - circle_center[0])
    point_angle_to_circle = -point_angle_to_circle
    print(outside_point[1] - circle_center[1], outside_point[0] - circle_center[0])
    if point_angle_to_circle < 0: point_angle_to_circle += 2 * math.pi
    if distance_from_circle > circle_radius: effective_distance = distance_from_circle
    else: effective_distance = circle_radius * 2 - distance_from_circle

    tangent_angle = point_angle_to_circle + math.acos(circle_radius / effective_distance)
    tangent_point = circle_radius * np.array([math.cos(tangent_angle), math.sin(tangent_angle)]).reshape(2,1) + circle_center
    tangent_point[1] = 720-tangent_point[1]
    velocity = (tangent_point - outside_point) / np.linalg.norm(tangent_point - outside_point)
    
    #velocity[1] = -velocity[1]
    print(point_angle_to_circle)
    # print(tangent_point - outside_point)
    # print(tangent_point)
    # print(outside_point)
    
    return velocity, tangent_point



class CircleGT:
    name = "Agent moving in a circle"
    def __init__(self, dt, measurement_noise_var, start_pos = np.array([[940, 360]]).T, 
                start_v = np.array([[0, -1]]).T, circle_radius = 300):
        '''
        To be added
        dt in ms
        start_v in pixel/ms
        '''
        self.circle_center = start_pos + circle_radius * (np.array([[0, 1], [-1, 0]]) @ start_v)
        self.pos = start_pos.astype(float) 
        self.v = start_v.astype(float)
        self.dt = dt
        self.measurement_noise_var = measurement_noise_var
        self.circle_radius = circle_radius

    def step(self):
        '''
        To be added
        '''
        # Update the simulation
        self.pos += self.v * self.dt
        acceleration = self.v * np.linalg.norm(self.v) / self.circle_radius # Set acceleration intensity to v^2/R 
        acceleration =  np.array([[0, 1], [-1, 0]]) @ acceleration # Set acceleration direction by rotating v by Pi/2
        self.v += acceleration * self.dt

        # Take the measurement
        measured_pos = self.pos + np.random.normal(0, self.measurement_noise_var, self.pos.shape)
        
        return self.pos, measured_pos, self.v

class CircleWControl:
    name = "Agent trying to move in a circle, using Kalman filter estimated position"
    def __init__(self, dt, measurement_noise_var, start_pos = np.array([[940, 360]]).T, 
                start_v = np.array([[0, -1]]).T, circle_radius = 300):
        '''
        To be added
        dt in ms
        start_v in pixel/ms
        '''
        self.circle_center = (start_pos + circle_radius * (np.array([[0, 1], [-1, 0]]) @ start_v)).astype(int)
        self.pos = start_pos.astype(float) 
        self.v = start_v.astype(float)
        self.dt = dt
        self.measurement_noise_var = measurement_noise_var
        self.circle_radius = circle_radius

    def step(self, estimated_pos):
        '''
        To be added
        '''
        # Update the simulation
        self.v, self.tangent_point = find_tangent_point(estimated_pos, self.circle_center, self.circle_radius)
        
        self.v = np.reshape(self.v, (2,1))
        self.pos += self.v * self.dt

        # Take the measurement
        measured_pos = self.pos + np.random.normal(0, self.measurement_noise_var, self.pos.shape)
        
        return self.pos, measured_pos, self.v


class LineGT:
    name = "Agent moving in a straight line"

    def __init__(self, dt, measurement_noise_var, start_pos = np.array([[0, 360]]).T, start_v = np.array([[1, 0]]).T):
        '''
        To be added
        dt in ms
        start_v in pixel/ms
        '''
        self.line_y = start_pos[1].astype(float) 
        self.pos = start_pos.astype(float) 
        self.v = start_v.astype(float)
        self.dt = dt
        self.measurement_noise_var = measurement_noise_var

    def step(self):
        '''
        To be added
        '''
        # Update the simulation
        self.pos += self.v * self.dt

        # Take the measurement
        measured_pos = self.pos + np.random.normal(0, self.measurement_noise_var, self.pos.shape)

        return self.pos, measured_pos, self.v

class LineWControl:
    name = "Agent trying to move in a straight line, using Kalman filter estimated position"

    def __init__(self, dt, measurement_noise_var, start_pos = np.array([[0, 360]]).T, start_v = np.array([[1, 0]]).T):
        '''
        To be added
        dt in ms
        start_v in pixel/ms
        '''
        self.line_y = start_pos[1].astype(float) 
        self.pos = start_pos.astype(float) 
        self.v = start_v.astype(float)
        self.dt = dt
        self.measurement_noise_var = measurement_noise_var

    def step(self, estimated_pos):
        '''
        To be added
        '''
        # Update the simulation
        distance_from_line = self.line_y - estimated_pos[1]
        self.v[1] = min([1, distance_from_line / 360])
        self.v[0] = (1 - self.v[1] ** 2) ** .5
        self.pos += self.v * self.dt

        # Take the measurement
        measured_pos = self.pos + np.random.normal(0, self.measurement_noise_var, self.pos.shape)

        return self.pos, measured_pos, self.v