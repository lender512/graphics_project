import numpy as np
import cv2

class Particle:
    def __init__(self, x, y, isRigid=False):
        self.acceleration = np.array([0, 0], dtype=float)
        self.velocity = np.array([0, 0], dtype=float)
        self.position = np.array([x, y], dtype=float)
        self.mass = 1
        self.isRigid = isRigid
    
    def applyForce(self, force):
        f = force.copy()
        f /= self.mass
        self.acceleration += f
        
    def update(self):
        self.velocity *= 0.99
        self.velocity += self.acceleration
        self.position += self.velocity
        self.acceleration *= 0
        
    def show(self, img):

        cv2.circle(img, np.rint(self.position).astype(int), 5, (0, 255, 0), -1)