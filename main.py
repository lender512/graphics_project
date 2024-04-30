from particle import Particle
from spring import Spring
import cv2
import numpy as np
import imageio
import noise
import random

WIDTH = 800
HEIGHT = 600


def generate_1d_perlin_noise(num_points, scale=100.0, octaves=6, persistence=0.5, lacunarity=2.0, seed=None):
    perlin_noise = []
    for i in range(num_points):
        x = i / scale
        value = noise.pnoise1(x,
                              octaves=octaves,
                              persistence=persistence,
                              lacunarity=lacunarity,
                              repeat=1024,
                              base=seed)
        perlin_noise.append(value)
    return perlin_noise

def generateSineWave(freq, amp, phase, n):
    wave_noise = []
    for i in range(n):
        value = amp * np.sin(2 * np.pi * freq * i / n + phase)
        wave_noise.append(value)
    return wave_noise

def generateSquereCloth(n, m, size):
    k = .2
    particles = []
    springs = []
    for i in range(n):
        for j in range(m):
            particles.append(Particle(i * size, j * size, (i == 0 and j == 0) or (i == n - 1 and j == 0)))
    for i in range(n):
        for j in range(m):
            if i < n - 1:
                springs.append(Spring(k, size, particles[i * m + j], particles[(i + 1) * m + j]))
            if j < m - 1:
                springs.append(Spring(k, size, particles[i * m + j], particles[i * m + j + 1]))
    return particles, springs

particles, springs = generateSquereCloth(10, 10, 30)
gravity = np.array([0, 0.1], dtype=float)
# perlin_wind = generate_1d_perlin_noise(WIDTH, scale=10.0, octaves=60, persistence=0.01, lacunarity=2.0, seed=random.randint(0, 1000))
perlin_wind = generateSineWave(0.2, 0.1, 0, WIDTH)
# print(perlin_wind)
# exit()
perlin_counter = 0
mouse_particle = Particle(0, 0, True)

imgs = []

while True:
    img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    
    for particle in particles:
        particle.update()
        particle.show(img)
        if not particle.isRigid:
            particle.applyForce(gravity)
            particle.applyForce(np.array([perlin_wind[perlin_counter], 0], dtype=float))
            print(perlin_wind[perlin_counter])
    perlin_counter += 1
    if perlin_counter >= WIDTH:
        perlin_counter = 0
            
    for spring in springs:
        spring.update()
        spring.show(img)
    cv2.imshow('img', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    imgs.append(img)
    
    if len(imgs) > 1000:
        break
    
imageio.mimsave('cloth.gif', imgs, fps=60)
    
    
