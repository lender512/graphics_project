import numpy as np

class Cloth:
    def __init__(self, width, height, num_particles):
        self.width = width
        self.height = height
        self.num_particles = num_particles
        self.vertices, self.faces, self.is_rigid = self.generate_cloth()
        self.colors = np.array([[0, 255, 0]] * len(self.faces))

    def generate_cloth(self):
        vertices = []
        faces = []
        for i in range(self.num_particles):
            row = []
            for j in range(self.num_particles):
                z = -90
                row.append([j * self.width / self.num_particles,
                            i * self.height / self.num_particles, z])
            vertices.append(row)

        vertices = np.array(vertices).reshape(-1, 3)
        is_rigid = np.zeros(len(vertices))

        for i in range(self.num_particles - 1):
            for j in range(self.num_particles - 1):
                faces.append([i * self.num_particles + j, i *
                              self.num_particles + j + 1, (i + 1) * self.num_particles + j])
                faces.append([i * self.num_particles + j + 1, (i + 1) *
                              self.num_particles + j + 1, (i + 1) * self.num_particles + j])


        for i in range(len(faces)):
            a = vertices[faces[i][0]]
            b = vertices[faces[i][1]]
            c = vertices[faces[i][2]]
            ab = b - a
            ac = c - a
            n = np.cross(ab, ac)
            if n[2] < 0:
                faces[i] = [faces[i][0], faces[i][2], faces[i][1]]

            faces[i] = faces[i][::-1]

        faces = np.array(faces)
        vertices[:, :2] -= np.mean(vertices[:, :2], axis=0)

        return vertices, faces, is_rigid