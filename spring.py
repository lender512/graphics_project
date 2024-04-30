import numpy as np
import cv2

class Spring:
    def __init__(self, k, restLength, a, b):
        self.k = k
        self.restLength = restLength
        self.a = a
        self.b = b
        
    def update(self):
        force = self.b.position - self.a.position
        x = np.linalg.norm(force) - self.restLength
        force = force / np.linalg.norm(force)
        force *= self.k * x
        if not self.a.isRigid:
            self.a.applyForce(force)
        force *= -1
        if not self.b.isRigid:
            self.b.applyForce(force)
        
    def show(self, img):
        cv2.line(img, np.rint(self.a.position).astype(int), np.rint(self.b.position).astype(int), (255, 255, 255), 4)