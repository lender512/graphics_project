import cv2
import numpy as np
import time
from particle_system import ParticleSystem
import mediapipe as mp

HEIGHT = 300
WIDTH = 300*16//9
FOCAL_LENGTH = 100


def generate_mesh(width, height, num_particles):
    vertices = []
    faces = []
    for i in range(num_particles):
        row = []
        for j in range(num_particles):
            # slightly offset the particles on the z axis to prevent them from being in the same plane
            # z = np.random.rand() * 0.1
            z = -90
            row.append([j * width / num_particles,
                       i * height / num_particles, z])
        vertices.append(row)

    # flatten the vertices
    vertices = np.array(vertices).reshape(-1, 3)
    is_rigid = np.zeros(len(vertices))

    # create trianglar faces
    for i in range(num_particles - 1):
        for j in range(num_particles - 1):
            faces.append([i * num_particles + j, i *
                         num_particles + j + 1, (i + 1) * num_particles + j])
            faces.append([i * num_particles + j + 1, (i + 1) *
                         num_particles + j + 1, (i + 1) * num_particles + j])

    # make top corners   rigid
    # is_rigid[0] = is_rigid[num_particles - 1] = True
    is_rigid[0] = True
    # is_rigid[num_particles * (num_particles - 1)] = is_rigid[num_particles * num_particles - 1] = True

    for i in range(len(faces)):
        a = vertices[faces[i][0]]
        b = vertices[faces[i][1]]
        c = vertices[faces[i][2]]
        ab = b - a
        ac = c - a
        n = np.cross(ab, ac)
        if n[2] < 0:
            faces[i] = [faces[i][0], faces[i][2], faces[i][1]]

        # reverse
        faces[i] = faces[i][::-1]

    faces = np.array(faces)

    vertices[:, :2] -= np.mean(vertices[:, :2], axis=0)

    return vertices, faces, is_rigid


def perspective_projection(verteces, f):
    x, y, z = verteces[:, 0], verteces[:, 1], verteces[:, 2]
    z = np.where(z == 0, 1, z)
    x_p = f * x / z
    y_p = f * y / z
    return np.column_stack((-x_p, -y_p))


def create_sphere(radius, iterations=2, origin=(0, 0, 0)):
    vertices = []
    faces = []

    # create icosahedron
    t = (1.0 + np.sqrt(5.0)) / 2.0
    vertices = np.array([
        [-1, t, 0],
        [1, t, 0],
        [-1, -t, 0],
        [1, -t, 0],
        [0, -1, t],
        [0, 1, t],
        [0, -1, -t],
        [0, 1, -t],
        [t, 0, -1],
        [t, 0, 1],
        [-t, 0, -1],
        [-t, 0, 1]
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

    # normalize vertices
    vertices = vertices / np.linalg.norm(vertices, axis=1)[:, None]

    # refine the sphere
    for _ in range(iterations):
        new_faces = []
        for face in faces:
            v1 = vertices[face[0]]
            v2 = vertices[face[1]]
            v3 = vertices[face[2]]
            v12 = (v1 + v2) / 2
            v23 = (v2 + v3) / 2
            v31 = (v3 + v1) / 2
            v12 = v12 / np.linalg.norm(v12)
            v23 = v23 / np.linalg.norm(v23)
            v31 = v31 / np.linalg.norm(v31)
            i1 = len(vertices)
            i2 = i1 + 1
            i3 = i1 + 2
            vertices = np.append(vertices, [v12, v23, v31], axis=0)
            new_faces.append([face[0], i1, i3])
            new_faces.append([i1, face[1], i2])
            new_faces.append([i2, face[2], i3])
            new_faces.append([i1, i2, i3])
        faces = new_faces

    # scale the sphere
    vertices = vertices * radius + origin
    faces = np.array(faces)

    return vertices, faces


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
    # make them all negative
    normals = -np.abs(normals)
    nonzero_mask = norms > 0
    normals[nonzero_mask[:, 0]] /= norms[nonzero_mask].reshape(-1, 1)

    return normals


def draw_mesh(img, vertices, mesh_triangles, normals, light_dir, shift=(0, 0), colors = None):
    indices = np.argsort(np.array([np.mean(vertices[triangle], axis=0)[
                         2] for triangle in mesh_triangles]))[::-1]
    indices = indices[::-1]

    for i in indices:
        triangle = mesh_triangles[i]
        normal = (normals[triangle[0]] +
                  normals[triangle[1]] + normals[triangle[2]]) / 3
        intensity = np.dot(normal, light_dir)
        shade = np.clip((intensity + 1) / 2, 0, 1)
        pts = perspective_projection(vertices[triangle], FOCAL_LENGTH)
        # pts = vertices[triangle][:, :2]
        pts = np.array(pts, np.int32)
        pts += shift

        
        color = colors[i]
        color = tuple([int(c * shade) for c in color])

        cv2.fillPoly(img, [pts], color)


mesh_vertices, mesh_triangles, is_rigid = generate_mesh(100, 100, 10)
mesh_colors = np.array([[0, 255, 0]] * len(mesh_triangles))
ps = ParticleSystem(mesh_vertices, np.array(
    [0, .91, 0], dtype=np.float64), .4, is_rigid)

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
light_dir = np.array([0, 0, 0])


start_time = t = time.time()
frame_count = 0

ball_radius = 10
ball_center = np.array([0, 0, -150], dtype=np.float64)
sphere_vertices, sphere_faces = create_sphere(ball_radius, 1, ball_center)
sphere_colors = np.array([[0, 0, 255]] * len(sphere_faces))


SHIFT = np.array([WIDTH // 2, HEIGHT // 2])

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

mp_hands = mp.solutions.hands
hand = mp_hands.Hands()
num_hands = 1
hands = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=num_hands,
                min_detection_confidence=0.65,
                min_tracking_confidence=0.65)
mp_drawing = mp.solutions.drawing_utils
timestamp = 0

while True:
    t = time.time()
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    np_array = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    if results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np_array)
            timestamp = timestamp + 1 # should be monotonically increasing, because in LIVE_STREAM mode
            if i == 0:
                tip_pos = hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y
                print(hand_landmarks.landmark[8].z)
                x = tip_pos[0] * WIDTH - SHIFT[0]
                y = tip_pos[1] * HEIGHT - SHIFT[1]
                z = -100 + hand_landmarks.landmark[8].z
                # coords = perspective_projection(np.array([x, y, z]), FOCAL_LENGTH)
                ps.x[0] = np.array([x, y, z])
            
    cloths_normals = calculate_normals(ps.x, mesh_triangles)
    sphere_normals = calculate_normals(sphere_vertices, sphere_faces)
    
    # choose whether to draw the mesh or the sphere
    # draw_mesh(img, ps.x, mesh_triangles, normals, light_dir, SHIFT)
    # draw_mesh(img, sphere_vertices, sphere_faces, np.zeros_like(sphere_vertices), light_dir, SHIFT)

    # draw all at once
    draw_mesh(frame, np.vstack((ps.x, sphere_vertices)), np.vstack((mesh_triangles, np.array(sphere_faces) +
              len(ps.x))), np.vstack((cloths_normals, sphere_normals)), light_dir, SHIFT, np.vstack((mesh_colors, sphere_colors)))

    ps.ball_collision(ball_radius, ball_center)

    cv2.putText(frame, f"FPS: {frame_count / (time.time() - start_time):.2f}",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"Ball Position: {
                ball_center}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imshow('img', frame)
    cv2.setMouseCallback('img', ps.mouseEvent, SHIFT)

    frame_count += 1
    ps.TimeStep((time.time() - t))

    cv2.imshow('img', frame)
    key = cv2.waitKey(1)
    
    if key == ord('a'):
        sphere_vertices[:, 0] -= 2
        ball_center[0] -= 2
    elif key == ord('d'):
        sphere_vertices[:, 0] += 2
        ball_center[0] += 2
    elif key == ord('w'):
        sphere_vertices[:, 2] -= 2
        ball_center[2] -= 2
    elif key == ord('s'):
        sphere_vertices[:, 2] += 2
        ball_center[2] += 2
    elif key == ord('q'):
        break
