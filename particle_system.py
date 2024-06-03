import numpy as np
import cv2

NUM_ITERATIONS = 3

class Constraint:
    def __init__(self, particleA, particleB, restlength):
        self.particleA = particleA
        self.particleB = particleB
        self.restlength = restlength

class ParticleSystem:
    defaultPosA = np.array([0, 0], dtype=np.float64)
    defaultPosB = np.array([500, 0], dtype=np.float64)

    def __init__(self, num_particles, gravity, time_step):
        self.m_x = np.zeros((num_particles, 2), dtype=np.float64)
        self.m_oldx = np.zeros((num_particles, 2), dtype=np.float64)
        self.m_a = np.zeros((num_particles, 2), dtype=np.float64)
        self.m_vGravity = gravity
        self.m_fTimeStep = time_step
        self.m_constraints = []

    def addConstraint(self, constraints):
        self.m_constraints = constraints

    def addParticles(self, particles):
        self.m_x = particles
        self.m_oldx = particles

    def TimeStep(self, dt):
        self.m_fTimeStep = dt
        self.AccumulateForces()
        self.Verlet()
        self.SatisfyConstraints()

    def Verlet(self):
        self.m_x, self.m_oldx = self.m_x + (self.m_x - self.m_oldx + self.m_a * self.m_fTimeStep ** 2), self.m_x.copy()

    def SatisfyConstraints(self):
        for _ in range(NUM_ITERATIONS):
            for constraint in self.m_constraints:
                delta = self.m_x[constraint.particleB] - self.m_x[constraint.particleA]
                deltaLength = np.linalg.norm(delta)
                diff = (deltaLength - constraint.restlength) / deltaLength if deltaLength > 0 else 0
                self.m_x[constraint.particleA] += delta * 0.5 * diff
                self.m_x[constraint.particleB] -= delta * 0.5 * diff
            self.m_x[0], self.m_x[19] = self.defaultPosA, self.defaultPosB

    def AccumulateForces(self):
        self.m_a[:] = self.m_vGravity

    def mouseEvent(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.defaultPosA = np.array([x, y], dtype=np.float64)
        if event == cv2.EVENT_RBUTTONDOWN:
            self.defaultPosB = np.array([x, y], dtype=np.float64)
