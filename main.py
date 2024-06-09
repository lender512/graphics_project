import cv2
import numpy as np
import time
from particle_system import ParticleSystem

WIDTH = 400
HEIGHT = 300
FOCAL_LENGTH = 100

def generate_mesh(width, height, num_particles):
    vertices = []
    faces = []
    for i in range(num_particles):
        row = []
        for j in range(num_particles):
            # slightly offset the particles on the z axis to prevent them from being in the same plane
            z = np.random.rand() * 0.1
            row.append([j * width / num_particles, i * height / num_particles, z])
        vertices.append(row)
    
    #flatten the vertices
    vertices = np.array(vertices).reshape(-1, 3)

    #create trianglar faces
    for i in range(num_particles - 1):
        for j in range(num_particles - 1):
            faces.append([i * num_particles + j, i * num_particles + j + 1, (i + 1) * num_particles + j])
            faces.append([i * num_particles + j + 1, (i + 1) * num_particles + j + 1, (i + 1) * num_particles + j])
            
    for i in range(len(faces)):
        a = vertices[faces[i][0]]
        b = vertices[faces[i][1]]
        c = vertices[faces[i][2]]
        ab = b - a
        ac = c - a
        n = np.cross(ab, ac)
        if n[2] < 0:
            faces[i] = [faces[i][0], faces[i][2], faces[i][1]]
        
        #reverse
        faces[i] = faces[i][::-1]
        
    faces = np.array(faces)
    return vertices, faces

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


def generate_sphere(radius, iterations, center=np.array([0, 0, 0], dtype=np.float64)):
    vertices = []
    faces = []
    
    return vertices, faces

def draw_sphere(img, points, faces, pos=np.array([0, 0, 0], dtype=np.float64)):
    points = np.array(points)
    for face in faces:
        pts = np.array([pts[face[0][0]], pts[face[1][0]], pts[face[2][0]]], np.int32)
        pts += pos[:2]
        
        cv2.fillPoly(img, [pts], (255, 255, 255))
        
def calculate_normals(vertices, mesh_triangles):
    # Calculate vectors from points a to b and a to c
    a = vertices[mesh_triangles[:, 0]]
    b = vertices[mesh_triangles[:, 1]]
    c = vertices[mesh_triangles[:, 2]]

    ab = b - a
    ac = c - a

    # Compute the normals for each face
    n = np.cross(ab, ac)

    # Accumulate the normals for each vertex
    normals = np.zeros_like(vertices)
    np.add.at(normals, mesh_triangles[:, 0], n)
    np.add.at(normals, mesh_triangles[:, 1], n)
    np.add.at(normals, mesh_triangles[:, 2], n)

    # Normalize the normals
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    #make them all negative
    normals = -np.abs(normals)
    nonzero_mask = norms > 0
    normals[nonzero_mask[:, 0]] /= norms[nonzero_mask].reshape(-1, 1)

    return normals

def draw_mesh(img, vertices, mesh_triangles, normals, light_dir):
    indices = np.argsort(np.array([np.mean(vertices[triangle], axis=0)[2] for triangle in mesh_triangles]))[::-1]

    for i in indices:
        triangle = mesh_triangles[i]
        pts = np.array([vertices[triangle[0]][:2], vertices[triangle[1]][:2], vertices[triangle[2]][:2]], np.int32)
        
        normal = (normals[triangle[0]] + normals[triangle[1]] + normals[triangle[2]]) / 3
        intensity = np.dot(normal, light_dir)
        shade = np.clip((intensity + 1) / 2, 0, 1)
        color = (200, 200, 200)
        color = tuple([int(c * shade) for c in color])
        
        cv2.fillPoly(img, [pts], color)
    
mesh_vertices, mesh_triangles = generate_mesh(100, 100, 15)
ps = ParticleSystem(mesh_vertices, np.array([0, .91, 0], dtype=np.float64), 0.5)

constraints = []
lengths = []

for i in range(len(mesh_triangles)):
    for j in range(3):
        for k in range(j + 1, 3):
            edge = [mesh_triangles[i][j], mesh_triangles[i][k]]
            if edge not in constraints and edge[::-1] not in constraints:
                constraints.append(edge)
                lengths.append(np.linalg.norm(ps.x[edge[0]] - ps.x[edge[1]]))
ps.setConstraint(constraints, lengths)


# Define the light source direction
light_dir = np.array([0, -0.5, -0.5])


start_time = t = time.time()
frame_count = 0

radius = 20

sphere = generate_sphere(radius, 3)

ball_posi = np.array([WIDTH // 2, HEIGHT // 2, 0])

while True:
    t = time.time()
    
    img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    
    normals = calculate_normals(ps.x, mesh_triangles)
    draw_mesh(img, ps.x, mesh_triangles, normals, light_dir)
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
        
    
           
    
