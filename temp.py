import numpy as np
import matplotlib.pyplot as plt
import noise
import random


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



perlin_wind = generate_1d_perlin_noise(100, scale=10.0, octaves=60, persistence=0.01, lacunarity=2.0, seed=random.randint(0, 1000))

plt.plot(perlin_wind)
plt.show()