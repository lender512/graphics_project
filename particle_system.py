import numpy as np
import cv2

NUM_ITERATIONS = 5

class ParticleSystem:

    def __init__(self, cloth, gravity, time_step):
        particles = cloth.vertices
        self.x = np.array(particles, dtype=np.float64)
        self.oldx = np.array(particles, dtype=np.float64)
        #use gravity to create aceleration array
        self.vGravity = gravity
        self.a = np.ones((len(particles), 3), dtype=np.float64) * gravity
        self.fTimeStep_2 = time_step * time_step
        self.constraints = []
        # self.is_rigid = is_rigid
        self.is_rigid = np.array(cloth.is_rigid, dtype=np.bool_)

    def setConstraint(self, constraints, lengths):
        self.constraints = np.array(constraints, dtype=np.int32)
        self.lengths = np.array(lengths, dtype=np.float64)
        
        
    def addParticles(self, particles):
        self.x = particles
        self.oldx = particles

    def TimeStep(self, dt):
        # self.fTimeStep = dt * dt
        self.AccumulateForces()
        self.Verlet()
        self.SatisfyConstraints()

    def Verlet(self):
        #just update where not is rigid
        # for i in range(len(self.x)):
        #     if not self.is_rigid[i]:
        #         temp = self.x[i].copy()
        #         self.x[i] = self.x[i] + (self.x[i] - self.oldx[i] + self.a[i] * self.fTimeStep_2)
        #         self.oldx[i] = temp
        #use vectorized operations
        temp = self.x[~self.is_rigid].copy()
        self.x[~self.is_rigid] = self.x[~self.is_rigid] + (self.x[~self.is_rigid] - self.oldx[~self.is_rigid] + self.a[~self.is_rigid] * self.fTimeStep_2)
        self.oldx[~self.is_rigid] = temp
        
         
        
    def SatisfyConstraints(self):
        for _ in range(NUM_ITERATIONS):
            for i, constraint in enumerate(self.constraints):
                delta = self.x[constraint[1]] - self.x[constraint[0]]
                deltaLength = np.linalg.norm(delta)
                correction = delta * (1 - self.lengths[i] / deltaLength)
                if not self.is_rigid[constraint[0]]:
                    self.x[constraint[0]] += correction * 0.5
                if not self.is_rigid[constraint[1]]:
                    self.x[constraint[1]] -= correction * 0.5
            # self.x[0], self.x[14] = self.defaultPosA, self.defaultPosB

    def AccumulateForces(self):
        self.a[:] = self.vGravity
        #set gravity where not is rigid
        

    def mouseEvent(self, event, x, y, flags, param):
        SHIFT = param
        #rigid indexes 
        indexes = self.is_rigid.nonzero()[0]
        if event == cv2.EVENT_LBUTTONDOWN:
            self.x[indexes[0]][:2] = np.array([x - SHIFT[0], y - SHIFT[1]], dtype=np.float64)
        if event == cv2.EVENT_RBUTTONDOWN:
            self.x[indexes[1]][:2] = np.array([x - SHIFT[0], y - SHIFT[1]], dtype=np.float64)
            
    def spacebarEvent(self):
        self.is_rigid = np.zeros(len(self.x), dtype=np.bool_)
            
    def ball_collision(self, ball_radius, ball_pos):
        # ball_radius = ball_radius * 1.2
        for i in range(len(self.x)):
            v = self.x[i] - ball_pos
            l = np.linalg.norm(v)
            if l <= ball_radius:
                self.x[i] += v / l * (ball_radius*1.2 - l)
                self.oldx[i] = self.x[i]
                
                