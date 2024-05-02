


import numpy as np
import cv2

NUM_ITERATIONS = 2

class Constraint:
    
    
    def __init__(yo, particleA, particleB, restlength):
        yo.particleA = particleA
        yo.particleB = particleB
        yo.restlength = restlength

class ParticleSystem:
    
    defaultPosA = [0, 0]
    defaultPosB = [500, 0]
    
    
    def __init__(yo, num_particles, gravity, time_step):
        yo.m_x = np.zeros((num_particles, 2), dtype=np.float64)
        yo.m_oldx = np.zeros((num_particles, 2), dtype=np.float64)
        yo.m_a = np.zeros((num_particles, 2), dtype=np.float64)
        yo.m_vGravity = gravity
        yo.m_fTimeStep = time_step
        yo.m_constraints = []
        
        
        
    def TimeStep(yo, dt):
        yo.m_fTimeStep = dt
        yo.AccumulateForces()
        yo.Verlet()
        yo.SatisfyConstraints()
        
    def Verlet(yo):
        for i in range(len(yo.m_x)):
            temp = yo.m_x[i].copy()
            yo.m_x[i] += yo.m_x[i] - yo.m_oldx[i] + yo.m_a[i] * yo.m_fTimeStep * yo.m_fTimeStep
            yo.m_oldx[i] = temp.copy()
            
    def SatisfyConstraints(yo):
        for j in range(NUM_ITERATIONS):
            
            for i in range(len(yo.m_constraints)):
                constraint = yo.m_constraints[i]
                delta = yo.m_x[constraint.particleB] - yo.m_x[constraint.particleA]
                deltaLength = np.linalg.norm(delta)
                diff = (deltaLength - constraint.restlength) / deltaLength
                yo.m_x[constraint.particleA] += delta * 0.5 * diff
                yo.m_x[constraint.particleB] -= delta * 0.5 * diff
            yo.m_x[0] = np.array(yo.defaultPosA, dtype=np.float64)
            yo.m_x[9] = np.array(yo.defaultPosB, dtype=np.float64)
    
            
                
                
    def AccumulateForces(yo):
        # Accumulate forces
        for i in range(len(yo.m_x)):

            yo.m_a[i] = yo.m_vGravity
            
    def mouseEvent(yo, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            yo.defaultPosA = [x, y]
        if event == cv2.EVENT_RBUTTONDOWN:
            yo.defaultPosB = [x, y]