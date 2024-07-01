import cv2
import numpy as np
import time
from particle_system import ParticleSystem
import mediapipe as mp
from cloth import Cloth
from ball import Ball
from enum import Enum

HEIGHT = 300
WIDTH = 300 * 16 // 9
FOCAL_LENGTH = 100
SHIFT = np.array([WIDTH // 2, HEIGHT // 2])
GRAVITY = np.array([0, 9.1, 0], np.float64)

# create enum for the different test


class Test(Enum):
    BALL = 0
    CAM = 1


def perspective_projection(vertices, f):
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    z = np.where(z == 0, 1, z)
    x_p = f * x / z
    y_p = f * y / z
    result = np.stack((-x_p, -y_p), axis=1)
    return np.array(result, np.int32)


def calculate_normals(vertices, mesh_triangles):
    a = vertices[mesh_triangles[:, 0]]
    b = vertices[mesh_triangles[:, 1]]
    c = vertices[mesh_triangles[:, 2]]
    ab = b - a
    ac = c - a
    n = np.cross(ab, ac)

    normals = np.zeros_like(vertices)
    np.add.at(normals, mesh_triangles[:, 0], n)
    np.add.at(normals, mesh_triangles[:, 1], n)
    np.add.at(normals, mesh_triangles[:, 2], n)

    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = -np.abs(normals)
    nonzero_mask = norms > 0
    normals[nonzero_mask[:, 0]] /= norms[nonzero_mask].reshape(-1, 1)

    return normals


def draw_mesh(img, vertices, mesh_triangles, normals, shift=(0, 0), colors=None, perspective=True):
    indices = np.argsort(np.array([np.mean(vertices[triangle], axis=0)[
                         2] for triangle in mesh_triangles]))[::-1]
    indices = indices[::-1]
    light_direction = np.array([0, 0, -.5])
    
    for i in indices:
        triangle = mesh_triangles[i]
        normal = (normals[triangle[0]] +
                  normals[triangle[1]] + normals[triangle[2]]) / 3
        shade = np.dot(normal, light_direction)
        shade = np.clip(shade, 0, 1)
        if perspective:
            pts = perspective_projection(vertices[triangle], FOCAL_LENGTH)
        else:
            pts = np.array(vertices[triangle][:, :2], np.int32)
        pts += shift

        color = colors[i]
        color = tuple([int(c * shade) for c in color])

        cv2.fillPoly(img, [pts], color)


def cam_test():
    cloth = Cloth(150, 150, 10)
    cloth.is_rigid[0] = True
    cloth.is_rigid[9] = True
    
    
    particle_system = ParticleSystem(
        cloth, GRAVITY, .4)

    constraints = []
    lengths = []

    for i in range(len(cloth.faces)):
        for j in range(3):
            for k in range(j + 1, 3):
                edge = [cloth.faces[i][j], cloth.faces[i][k]]
                if edge not in constraints and edge[::-1] not in constraints:
                    constraints.append(edge)
                    lengths.append(np.linalg.norm(
                        particle_system.x[edge[0]] - particle_system.x[edge[1]]))
    particle_system.setConstraint(constraints, lengths)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.65,
        min_tracking_confidence=0.65
    )
    frame_count = 0
    mp_drawing = mp.solutions.drawing_utils
    start_time = time.time()
    t = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                if i == 0:
                    tip_pos = hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y
                    x = tip_pos[0] * WIDTH - SHIFT[0] +30
                    y = tip_pos[1] * HEIGHT - SHIFT[1]+10
                    z = -100 + hand_landmarks.landmark[8].z
                    if not particle_system.is_rigid[0]:
                        particle_system.x = cloth.vertices.copy()
                        particle_system.oldx = cloth.vertices.copy()
                    particle_system.x[0] = np.array([x, y, z])
                    particle_system.is_rigid[0] = True
                if i == 1:
                    tip_pos = hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y
                    x = tip_pos[0] * WIDTH - SHIFT[0] +60
                    y = tip_pos[1] * HEIGHT - SHIFT[1]+10
                    z = -100 + hand_landmarks.landmark[8].z
                    if not particle_system.is_rigid[9]:
                        particle_system.x = cloth.vertices.copy()
                        particle_system.oldx = cloth.vertices.copy()
                    particle_system.x[9] = np.array([x, y, z])
                    particle_system.is_rigid[9] = True
        else:
            particle_system.is_rigid[0] = False
            particle_system.is_rigid[9] = False

        cloth_normals = calculate_normals(particle_system.x, cloth.faces)

        draw_mesh(frame, particle_system.x,
                  cloth.faces,
                  cloth_normals, SHIFT,
                  cloth.colors,
                  perspective=False)


        cv2.putText(frame, f"FPS: {frame_count / (time.time() - start_time):.2f}",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow('img', frame)
        cv2.setMouseCallback('img', particle_system.mouseEvent, SHIFT)

        frame_count += 1
        particle_system.TimeStep((time.time() - t))
        
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def ball_test():
    l = 12
    cloth = Cloth(400, 400, l)
    cloth.vertices[:, 2] -= 150
    cloth.is_rigid[0] = True
    cloth.is_rigid[11] = True
    ball = Ball(60, [0, 0, -150])
    particle_system = ParticleSystem(
        cloth, GRAVITY, .4)

    constraints = []
    lengths = []

    flatten_coord = lambda i, j: i * l + j
    for idx in range(len(cloth.vertices)):
        i = idx // l
        j = idx % l
        for (a, b) in [(i, j + 1), (i + 1, j), (i + 1, j + 1), (i - 1, j + 1)]:
            if 0 <= a < l and 0 <= b < l:
                constraints.append([idx, flatten_coord(a, b)])
                lengths.append(np.linalg.norm(
                    particle_system.x[idx] - particle_system.x[flatten_coord(a, b)]))
    
    particle_system.setConstraint(constraints, lengths)            
    

    start_time = time.time()
    frame_count = 0

    t = time.time()
    while True:
        frame = np.zeros((HEIGHT, WIDTH, 3), np.uint8)

        cloth_normals = calculate_normals(particle_system.x, cloth.faces)
        ball_normals = calculate_normals(ball.vertices, ball.faces)

        draw_mesh(frame, np.vstack((particle_system.x, ball.vertices)),
                  np.vstack((cloth.faces, np.array(
                      ball.faces) + len(particle_system.x))),
                  np.vstack((cloth_normals, ball_normals)), SHIFT,
                  np.vstack((cloth.colors, ball.colors)))

        particle_system.ball_collision(ball.radius, ball.origin)

        cv2.putText(frame, f"FPS: {frame_count / (time.time() - start_time):.2f}",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Ball Position: {ball.origin}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow('img', frame)
        cv2.setMouseCallback('img', particle_system.mouseEvent, SHIFT)

        frame_count += 1
        particle_system.TimeStep((time.time() - t))

        key = cv2.waitKey(1)
        if key == ord('a'):
            ball.vertices[:, 0] -= 2
            ball.origin[0] -= 2
        elif key == ord('d'):
            ball.vertices[:, 0] += 2
            ball.origin[0] += 2
        elif key == ord('w'):
            ball.vertices[:, 2] -= 2
            ball.origin[2] -= 2
        elif key == ord('s'):
            ball.vertices[:, 2] += 2
            ball.origin[2] += 2
        elif key == ord(' '):
            particle_system.spacebarEvent()
        elif key == ord('q'):
            break


def main(test):
    if test == Test.CAM:
        cam_test()
    elif test == Test.BALL:
        ball_test()
    


if __name__ == "__main__":
    test = Test.CAM
    # test = Test.BALL
    main(test)
