import numpy as np

class Ball:
    def __init__(self, radius, origin):
        self.radius = radius
        self.origin = np.array(origin, dtype=np.float64)
        self.vertices, self.faces = self.create_sphere()
        self.colors = np.array([[0, 0, 255]] * len(self.faces))

    def create_sphere(self):
        vertices = []
        faces = []

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

        vertices = vertices / np.linalg.norm(vertices, axis=1)[:, None]

        for _ in range(2):
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

        vertices = vertices * self.radius + self.origin
        faces = np.array(faces)

        return vertices, faces