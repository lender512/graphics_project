import numpy as np
import cv2

class Particle:
    def __init__(self, x, y, isRigid=False):
        self.acceleration = np.array([0, 0], dtype=float)
        self.velocity = np.array([0, 0], dtype=float)
        self.position = np.array([x, y], dtype=float)
        self.mass = 1
        self.isRigid = isRigid
    
    # Vec3d apply_forces() const
    # {
    #     Vec3d grav_acc = Vec3d{0.0, 0.0, -9.81 }; // 9.81 m/s² down in the z-axis
    #     Vec3d drag_force = 0.5 * drag * (vel * vel); // D = 0.5 * (rho * C * Area * vel^2)
    #     Vec3d drag_acc = drag_force / mass; // a = F/m
    #     return grav_acc - drag_acc;
    # }
    def applyForce(self, force):
        self.acceleration += force / self.mass
        
    
    def update(self, dt):
        new_pos = self.position + self.velocity * dt + self.acceleration * (dt * dt * 0.5)
        # new_acc = self.apply_forces()
        new_vel = self.velocity + (self.acceleration) * (dt * 0.5)
        self.position = new_pos
        self.velocity = new_vel
        # self.acceleration = new_acc
        
        
        
    def show(self, img):
        cv2.circle(img, np.rint(self.position).astype(int), 5, (0, 255, 0), -1)