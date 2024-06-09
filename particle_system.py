import numpy as np
import cv2

NUM_ITERATIONS = 1

class ParticleSystem:
    defaultPosA = np.array([0, 0, 0], dtype=np.float64)
    defaultPosB = np.array([500, 0, 0], dtype=np.float64)

    def __init__(self, num_particles, gravity, time_step):
        self.x = np.zeros((num_particles, 3), dtype=np.float64)
        self.oldx = np.zeros((num_particles, 3), dtype=np.float64)
        #use gravity to create aceleration array
        self.vGravity = gravity
        self.a = np.ones((num_particles, 3), dtype=np.float64) * gravity
        self.fTimeStep_2 = time_step * time_step
        self.constraints = []

    def setConstraint(self, constraints):
        self.constraints = constraints

    def addParticles(self, particles):
        self.x = particles
        self.oldx = particles

    def TimeStep(self, dt):
        # self.fTimeStep = dt * dt
        # self.AccumulateForces()
        self.Verlet()
        self.SatisfyConstraints()

    def Verlet(self):
        temp = self.x.copy()
        self.x = self.x + (self.x - self.oldx + self.a * self.fTimeStep_2)
        self.oldx = temp
        
    def SatisfyConstraints(self):
        for _ in range(NUM_ITERATIONS):
            for i, constraint in enumerate(self.constraints):
                delta = self.x[constraint[1]] - self.x[constraint[0]]
                deltaLength = np.linalg.norm(delta)
                correction = delta * (1 - constraint[2] / deltaLength)
                self.x[constraint[0]], self.x[constraint[1]] = self.x[constraint[0]] + correction * 0.5, self.x[constraint[1]] - correction * 0.5
            self.x[0], self.x[9] = self.defaultPosA, self.defaultPosB

    def AccumulateForces(self):
        self.a[:] = self.vGravity

    def mouseEvent(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.defaultPosA = np.array([x, y, 0], dtype=np.float64)
        if event == cv2.EVENT_RBUTTONDOWN:
            self.defaultPosB = np.array([x, y, 0], dtype=np.float64)
            
    def ball_collision(self, ball_radius, ball_pos):
        for i in range(len(self.x)):
            if np.linalg.norm(self.x[i] - ball_pos) < ball_radius:
                normal = (self.x[i] - ball_pos) / np.linalg.norm(self.x[i] - ball_pos)
                self.x[i] = ball_pos + normal * ball_radius
                self.oldx[i] = self.x[i]
