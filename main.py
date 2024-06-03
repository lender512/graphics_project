import cv2
import numpy as np
import random
import time
from particle_system import ParticleSystem, Constraint

WIDTH = 800
HEIGHT = 600

def generate_test(width, height, num_particles):
    test = []
    for i in range(num_particles):
        row = []
        for j in range(num_particles):
            row.append([j * width / num_particles, i * height / num_particles])
        test.append(row)
    return test

test = generate_test(500, 500, 20)

ps = ParticleSystem(len(test[0]) * len(test), np.array([0, 1000], dtype=np.float64), 0.2)

for i in range(len(test)):
    for j in range(len(test[i])):
        ps.m_x[i * len(test[i]) + j] = test[i][j]
        ps.m_oldx[i * len(test[i]) + j] = test[i][j]

l = 20    
constraints = []
for i in range(len(test)):
    for j in range(len(test[i])):
        if i > 0:
            constraints.append(Constraint(i * len(test[i]) + j, (i - 1) * len(test[i]) + j, l))
        if i < len(test) - 1:
            constraints.append(Constraint(i * len(test[i]) + j, (i + 1) * len(test[i]) + j, l))
        if j > 0:
            constraints.append(Constraint(i * len(test[i]) + j, i * len(test[i]) + j - 1, l))
        if j < len(test[i]) - 1:
            constraints.append(Constraint(i * len(test[i]) + j, i * len(test[i]) + j + 1, l))

ps.addConstraint(constraints)

t = time.time()
start_time = time.time()
frame_count = 0
while True:
    img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    for i in range(len(ps.m_x)):
        cv2.circle(img, np.rint(ps.m_x[i][:2]).astype(int), 2, (255, 255, 255), -1)
    for c in ps.m_constraints:
        cv2.line(img, np.rint(ps.m_x[c.particleA][:2]).astype(int), np.rint(ps.m_x[c.particleB][:2]).astype(int), (255, 255, 255), 1)
        
    cv2.putText(img, f"FPS: {frame_count / (time.time() - start_time):.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.imshow('img', img)
    cv2.setMouseCallback('img', ps.mouseEvent)
    
    ps.TimeStep(time.time() - t)
    t = time.time()
    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
