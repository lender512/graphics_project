


import numpy as np
import cv2

NUM_ITERATIONS = 10

class Constraint:
    
    
    def __init__(self, particleA, particleB, restlength):
        self.particleA = particleA
        self.particleB = particleB
        self.restlength = restlength

class ParticleSystem:
    
    defaultPosA = [0, 0]
    defaultPosB = [500, 0]
    
    
    def __init__(self, num_particles, gravity, time_step):
        self.m_x = np.zeros((num_particles, 2), dtype=np.float64)
        self.m_oldx = np.zeros((num_particles, 2), dtype=np.float64)
        self.m_a = np.zeros((num_particles, 2), dtype=np.float64)
        self.m_vGravity = gravity
        self.m_fTimeStep = time_step
        self.m_constraints = []
        
        
        
    def TimeStep(self, dt):
        self.m_fTimeStep = dt
        self.AccumulateForces()
        self.Verlet()
        self.SatisfyConstraints()
        
    def Verlet(self):
        for i in range(len(self.m_x)):
            x = self.m_x[i]
            temp = x
            oldx = self.m_oldx[i]
            a = self.m_a[i]
            x += x - oldx + a * self.m_fTimeStep * self.m_fTimeStep
            oldx = temp
            self.m_oldx[i] = oldx
            
    def SatisfyConstraints(self):
        for j in range(NUM_ITERATIONS):
            
            for i in range(len(self.m_constraints)):
                c = self.m_constraints[i]
                x1 = self.m_x[c.particleA].copy()
                x2 = self.m_x[c.particleB].copy()
                delta = x2 - x1
                delta *= c.restlength * c.restlength / (delta.dot(delta) + c.restlength * c.restlength) - 0.5
                self.m_x[c.particleA] -= delta
                self.m_x[c.particleB] += delta
            
            self.m_x[0] = np.array(self.defaultPosA, dtype=np.float64)
                
    
            
                
                
    def AccumulateForces(self):
        # Accumulate forces
        for i in range(len(self.m_x)):

            self.m_a[i] = self.m_vGravity
            
    def mouseEvent(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.defaultPosA = [x, y]
        if event == cv2.EVENT_RBUTTONDOWN:
            self.defaultPosB = [x, y]