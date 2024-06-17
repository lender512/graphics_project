import numpy as np
import cv2

def create_cube(l, origin=(0, 0, 0)):
    vertices = np.array([
        [origin[0] - l / 2, origin[1] - l / 2, origin[2] - l / 2],
        [origin[0] + l / 2, origin[1] - l / 2, origin[2] - l / 2],
        [origin[0] + l / 2, origin[1] + l / 2, origin[2] - l / 2],
        [origin[0] - l / 2, origin[1] + l / 2, origin[2] - l / 2],
        [origin[0] - l / 2, origin[1] - l / 2, origin[2] + l / 2],
        [origin[0] + l / 2, origin[1] - l / 2, origin[2] + l / 2],
        [origin[0] + l / 2, origin[1] + l / 2, origin[2] + l / 2],
        [origin[0] - l / 2, origin[1] + l / 2, origin[2] + l / 2]
    ])
    
    faces = np.array([
        [0, 1, 2],
        [0, 2, 3],
        [1, 5, 6],
        [1, 6, 2],
        [5, 4, 7],
        [5, 7, 6],
        [4, 0, 3],
        [4, 3, 7],
        [3, 2, 6],
        [3, 6, 7],
        [4, 5, 1],
        [4, 1, 0]
    ])
    
    return vertices, faces


#create sphere of triangles
def create_sphere(radius, iterations=2, origin=(0, 0, 0)):
    vertices = []
    faces = []
    
    #create icosahedron
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
    
    #normalize vertices
    vertices = vertices / np.linalg.norm(vertices, axis=1)[:, None]
    
    #refine the sphere
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
        
    #scale the sphere
    vertices = vertices * radius + origin
    
    return vertices, faces

def perspective_projection(vertex, f):
    x, y, z = vertex
    # Avoid division by zero
    if z == 0:
        z = 1
    x_p = int(f * x / z)
    y_p = int(f * y / z)
    return np.array([x_p, y_p])


def draw_cube(vertices, faces, img, color=(0, 255, 0), shift=(0, 0)):
    for face in faces:
        points = np.array([vertices[i] for i in face])
        for i in range(len(points)):
            a = points[i] + shift
            b = points[(i + 1) % len(points)] + shift
            cv2.line(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), color, 2)

def draw_cube_perspective(vertices, faces, img, f, color=(0, 255, 0)):
    # vertices = vertices + shift
    projected_vertices = np.array([perspective_projection(v, f) for v in vertices])

    #projected_vertices[:, 0] += shift[0]
    # Shift vertices to the center of the imageq
    
    #applt the shift
    
    draw_cube(projected_vertices, faces, img, color, shift=(256, 256))

def rotate_cube(vertices, angle = [0, 0, 0], pivot=(0, 0, 0)):
    angle = np.radians(angle)
    
    #consider pivot
    vertices = vertices - pivot
    
    #rotation matrix
    rotation_matrix = np.array([
        [np.cos(angle[1]) * np.cos(angle[2]), np.cos(angle[1]) * np.sin(angle[2]), -np.sin(angle[1])],
        [np.sin(angle[0]) * np.sin(angle[1]) * np.cos(angle[2]) - np.cos(angle[0]) * np.sin(angle[2]), np.sin(angle[0]) * np.sin(angle[1]) * np.sin(angle[2]) + np.cos(angle[0]) * np.cos(angle[2]), np.sin(angle[0]) * np.cos(angle[1])],
        [np.cos(angle[0]) * np.sin(angle[1]) * np.cos(angle[2]) + np.sin(angle[0]) * np.sin(angle[2]), np.cos(angle[0]) * np.sin(angle[1]) * np.sin(angle[2]) - np.sin(angle[0]) * np.cos(angle[2]), np.cos(angle[0]) * np.cos(angle[1])]
    ])
    
    
    result = vertices @ rotation_matrix.T
    result = result + pivot

    return result

if __name__ == "__main__":
    img = np.zeros((512, 512, 3), np.uint8)
    # vertices, faces = create_cube(10, origin=(0, 0, -20))
    vertices, faces = create_sphere(10, origin=(0, 0, -20))
    
    # #rotate the cube
    
    # vertices = vertices @ rotation_matrix.T
    # draw_cube_perspective(vertices, faces, img, 100)
    
    
    # cv2.imshow("Cube", img)
    # cv2.waitKey(0)
    
    while True:
        img = np.zeros((512, 512, 3), np.uint8)
        draw_cube_perspective(vertices, faces, img, 100)
        vertices = rotate_cube(vertices, [0.00, 1, 0.00], np.mean(vertices, axis=0))
        cv2.imshow("Cube", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    
