import cv2
import numpy as np
import time
from particle_system import ParticleSystem

WIDTH = 400
HEIGHT = 300

def generate_mesh(width, height, num_particles):
    points = []
    for i in range(num_particles):
        row = []
        for j in range(num_particles):
            # slightly offset the particles on the z axis to prevent them from being in the same plane
            z = np.random.rand() * 0.1
            row.append([j * width / num_particles, i * height / num_particles, z])
        points.append(row)
    return points

import numpy as np

def perspective_projection(points, fov, aspect_ratio, near, far):
    # Ensure points is a 2D array
    points = np.atleast_2d(points)

    # Calculate the perspective projection matrix
    f = 1 / np.tan(np.radians(fov) / 2)
    z_range = near - far
    
    projection_matrix = np.array([
        [f / aspect_ratio, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (far + near) / z_range, (2 * far * near) / z_range],
        [0, 0, -1, 0]
    ])
    
    # Add a fourth dimension to the points for matrix multiplication
    points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
    
    # Apply the perspective projection matrix
    projected_points = points_homogeneous @ projection_matrix.T
    
    # Normalize the points (perspective divide)
    projected_points /= projected_points[:, 3][:, np.newaxis]
    
    # Return the 2D coordinates (discard the z and w components)
    return projected_points[:, :2]

import numpy as np

def normalize(v):
    norm = np.linalg.norm(v, axis=1)
    return v / norm[:, np.newaxis]

def generate_icosahedron():
    t = (1.0 + np.sqrt(5.0)) / 2.0
    
    vertices = np.array([
        [-1,  t,  0],
        [ 1,  t,  0],
        [-1, -t,  0],
        [ 1, -t,  0],
        [ 0, -1,  t],
        [ 0,  1,  t],
        [ 0, -1, -t],
        [ 0,  1, -t],
        [ t,  0, -1],
        [ t,  0,  1],
        [-t,  0, -1],
        [-t,  0,  1]
    ])
    
    faces = np.array([
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1]
    ])
    
    return vertices, faces

def midpoint(p1, p2):
    return (p1 + p2) / 2.0

def subdivide(vertices, faces):
    vertex_map = {}
    new_faces = []

    def get_vertex(v1, v2):
        key = tuple(sorted((v1, v2)))
        if key not in vertex_map:
            new_vertex = midpoint(vertices[v1], vertices[v2])
            vertex_map[key] = len(vertices)
            vertices.append(new_vertex)
        return vertex_map[key]

    vertices = vertices.tolist()
    
    for tri in faces:
        v1, v2, v3 = tri
        a = get_vertex(v1, v2)
        b = get_vertex(v2, v3)
        c = get_vertex(v3, v1)
        
        new_faces.append([v1, a, c])
        new_faces.append([v2, b, a])
        new_faces.append([v3, c, b])
        new_faces.append([a, b, c])
    
    return np.array(vertices), np.array(new_faces)

def generate_sphere(radius, iterations, center=np.array([0, 0, 0], dtype=np.float64)):
    vertices, faces = generate_icosahedron()
    
    for _ in range(iterations):
        vertices, faces = subdivide(vertices, faces)
    
    vertices = np.array(vertices)
    vertices = normalize(vertices)
    vertices = vertices * radius + center
    
    return vertices, faces


FOCAL_LENGTH = 100

def draw_sphere(img, points, faces, pos=np.array([0, 0, 0], dtype=np.float64)):
    #apply perspective projection to the particles
    points = np.array(points)
    pts = perspective_projection(points, FOCAL_LENGTH, WIDTH / HEIGHT, 0.1, 100)
    for face in faces:
        #use pos to move the sphere
        #apply perspective projection to the particles
        
        # pts = np.array([points[face[0][0]][face[0][1]][:2], points[face[1][0]][face[1][1]][:2], points[face[2][0]][face[2][1]][:2]], np.int32)
        pts = np.array([pts[face[0][0]], pts[face[1][0]], pts[face[2][0]]], np.int32)
        #sum radius and pos to move the sphere
        pts += pos[:2]
        
        cv2.fillPoly(img, [pts], (255, 255, 255))
    
mesh_pointstest = generate_mesh(150, 150, 10)

ps = ParticleSystem(len(mesh_pointstest[0]) * len(mesh_pointstest), np.array([0, .91, 0], dtype=np.float64), 0.5)

for i in range(len(mesh_pointstest)):
    for j in range(len(mesh_pointstest[i])):
        ps.x[i * len(mesh_pointstest[i]) + j] = mesh_pointstest[i][j]
        ps.oldx[i * len(mesh_pointstest[i]) + j] = mesh_pointstest[i][j]

constraints = np.ndarray(shape=(0, 3), dtype=np.int32)

for i in range(len(mesh_pointstest)):
    for j in range(len(mesh_pointstest[i])):
        if i > 0:
            constraints = np.append(constraints, [[i * len(mesh_pointstest[i]) + j, (i - 1) * len(mesh_pointstest[i]) + j, int(np.sqrt((mesh_pointstest[i][j][0] - mesh_pointstest[i - 1][j][0]) ** 2 + (mesh_pointstest[i][j][1] - mesh_pointstest[i - 1][j][1]) ** 2))]], axis=0)
        if i < len(mesh_pointstest) - 1:
            constraints = np.append(constraints, [[i * len(mesh_pointstest[i]) + j, (i + 1) * len(mesh_pointstest[i]) + j, int(np.sqrt((mesh_pointstest[i][j][0] - mesh_pointstest[i + 1][j][0]) ** 2 + (mesh_pointstest[i][j][1] - mesh_pointstest[i + 1][j][1]) ** 2))]], axis=0)
        if j > 0:
            constraints = np.append(constraints, [[i * len(mesh_pointstest[i]) + j, i * len(mesh_pointstest[i]) + j - 1, int(np.sqrt((mesh_pointstest[i][j][0] - mesh_pointstest[i][j - 1][0]) ** 2 + (mesh_pointstest[i][j][1] - mesh_pointstest[i][j - 1][1]) ** 2))]], axis=0)
        if j < len(mesh_pointstest[i]) - 1:
            constraints = np.append(constraints, [[i * len(mesh_pointstest[i]) + j, i * len(mesh_pointstest[i]) + j + 1, int(np.sqrt((mesh_pointstest[i][j][0] - mesh_pointstest[i][j + 1][0]) ** 2 + (mesh_pointstest[i][j][1] - mesh_pointstest[i][j + 1][1]) ** 2))]], axis=0)
        
        # Only one diagonal constraint to form triangles
        if i > 0 and j > 0:
            constraints = np.append(constraints, [[i * len(mesh_pointstest[i]) + j, (i - 1) * len(mesh_pointstest[i]) + (j - 1), int(np.sqrt((mesh_pointstest[i][j][0] - mesh_pointstest[i - 1][j - 1][0]) ** 2 + (mesh_pointstest[i][j][1] - mesh_pointstest[i - 1][j - 1][1]) ** 2))]], axis=0)

ps.setConstraint(constraints)

# Generate triangles. 3 points per triangle where each point is an index of a particle
triangles = np.ndarray(shape=(0, 3), dtype=np.int32)

for i in range(len(mesh_pointstest) - 1):
    for j in range(len(mesh_pointstest[i]) - 1):
        triangles = np.append(triangles, [[i * len(mesh_pointstest[i]) + j, (i + 1) * len(mesh_pointstest[i]) + j, i * len(mesh_pointstest[i]) + j + 1]], axis=0)
        triangles = np.append(triangles, [[(i + 1) * len(mesh_pointstest[i]) + j, (i + 1) * len(mesh_pointstest[i]) + j + 1, i * len(mesh_pointstest[i]) + j + 1]], axis=0)

#sort the triangles clockwise
for i in range(len(triangles)):
    a = ps.x[triangles[i][0]][:2]
    b = ps.x[triangles[i][1]][:2]
    c = ps.x[triangles[i][2]][:2]
    center = (a + b + c) / 3
    angle = np.arctan2(center[1] - a[1], center[0] - a[0])
    angles = [np.arctan2(p[1] - a[1], p[0] - a[0]) for p in [b, c]]
    if angles[0] < angle:
        triangles[i][1], triangles[i][2] = triangles[i][2], triangles[i][1]

ps.triangles = triangles

# Define the light source direction
light_dir = np.array([0, -0.5, -0.5])

t = time.time()
start_time = time.time()
frame_count = 0

radius = 30

sphere = generate_sphere(radius, 3)

ball_posi = np.array([WIDTH // 2, HEIGHT // 2, 0])

while True:
    t = time.time()
    
    img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    
    normals = np.zeros((len(ps.x), 3), dtype=np.float64)
    for t in ps.triangles:
        a = ps.x[t[0]]
        b = ps.x[t[1]]
        c = ps.x[t[2]]
        
        ab = b - a
        ac = c - a
        n = np.cross(ab, ac)
        normals[t[0]] += n
        normals[t[1]] += n
        normals[t[2]] += n

    for i in range(len(normals)):
        normals[i] = normals[i] / np.linalg.norm(normals[i]) if np.linalg.norm(normals[i]) > 0 else normals[i]
    
    
    #sort the triangles by z value
    indices = np.argsort([np.mean([ps.x[t[0]][2], ps.x[t[1]][2], ps.x[t[2]][2]]) for t in ps.triangles])[::-1]
    
    for i in indices:
        t = ps.triangles[i]
        # pts = np.array([ps.x[t[0]][:2], ps.x[t[1]][:2], ps.x[t[2]][:2]], np.int32)
        #apply perspective projection to the particles

        pts = np.array([ps.x[t[0]][:2], ps.x[t[1]][:2], ps.x[t[2]][:2]], np.int32)
        
        
        normal = (normals[t[0]] + normals[t[1]] + normals[t[2]]) / 3
        intensity = np.dot(normal, light_dir)
        shade = np.clip((intensity + 1) / 2, 0, 1)
        color = (200, 200, 200)
        color = tuple([int(c * shade) for c in color])
        
        cv2.fillPoly(img, [pts], color)
    
    
    draw_sphere(img, sphere[0], sphere[1], ball_posi)
    ps.ball_collision(radius, ball_posi)
    
    cv2.putText(img, f"FPS: {frame_count / (time.time() - start_time):.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.imshow('img', img)
    cv2.setMouseCallback('img', ps.mouseEvent)
    
    frame_count += 1
    ps.TimeStep((time.time() - t))

    
    key = cv2.waitKey(1)
    if key == ord('w'):
        ball_posi[2] -= 2
    elif key == ord('s'):
        ball_posi[2] += 2
    elif key == ord('a'):
        ball_posi[0] -= 2
    elif key == ord('d'):
        ball_posi[0] += 2
    elif key == ord('q'):
        break
        
    
           
    
