import numpy as np
import cv2

class Spring:
    def __init__(self, k, restLength, a, b):
        self.k = k
        self.restLength = restLength
        self.a = a
        self.b = b
    
    # delta = x2-x1;
    # deltalength = sqrt(delta*delta);
    # diff = (deltalength-restlength)/deltalength;
    # x1 -= delta*0.5*diff;
    # x2 += delta*0.5*diff;

    def update(self):
        delta = self.b.position - self.a.position
        deltaLength = np.linalg.norm(delta)
        diff = (deltaLength - self.restLength) / deltaLength
        self.a.position -= delta * 0.5 * diff
        self.b.position += delta * 0.5 * diff
        
        
        
        
        
    def show(self, img):
        cv2.line(img, np.rint(self.a.position).astype(int), np.rint(self.b.position).astype(int), (255, 255, 255), 4)