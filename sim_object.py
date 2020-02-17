class SimRect:
    def __init__(self, vertice):
        self.vertice = vertice

    def vertices_to_edges(self, vertices):
        def edge_direction(p0, p1):
            return (p1[0] - p0[0], p1[1] - p0[1])

        return [edge_direction(vertices[i], vertices[(i + 1) % len(vertices)]) \
                for i in range(len(vertices))]

    def project(self, vertices, axis):
        def dot(a, b):
            return a[0] * b[0] + a[1] * b[1]

        dots = [dot(vertex, axis) for vertex in vertices]
        return [min(dots), max(dots)]

    def overlap(self, a, b):
        def contains(n, range_):
            a = range_[0]
            b = range_[1]
            if b < a:
                a = range_[1]
                b = range_[0]
            return (n >= a) and (n <= b)

        if contains(a[0], b):  return True
        if contains(a[1], b):  return True
        if contains(b[0], a):  return True
        if contains(b[1], a):  return True
        return False

    def intersect(self, alien_rect):

        def normalize(v):
            from math import sqrt
            norm = sqrt(v[0] ** 2 + v[1] ** 2)
            return (v[0] / norm, v[1] / norm)

        def orthogonal(v):
            return (v[1], -v[0])

        edges_a = self.vertices_to_edges(self.vertice);
        edges_b = self.vertices_to_edges(alien_rect.vertice);

        edges = edges_a + edges_b
        axes = [normalize(orthogonal(edge)) for edge in edges]
        for i in range(len(axes)):
            projection_a = self.project(self.vertice, axes[i])
            projection_b = self.project(alien_rect.vertice, axes[i])
            overlapping = self.overlap(projection_a, projection_b)
            if not overlapping:
                return False
        return True


class CarColider:
    def __init__(self):
        pass


class WallColider:
    def __init__(self):
        pass


if __name__ == "__main__":

    a_vertices = [(0, 0), (70, 0), (0, 70)]
    b_vertices = [(70, 70), (150, 70), (70, 150)]
    c_vertices = [(30, 30), (150, 70), (70, 150)]

    s1 = SimRect(a_vertices)
    s2 = SimRect(b_vertices)
    s3 = SimRect(c_vertices)

    print(s1.intersect(s2))
    print(s1.intersect(s3))
    print(s3.intersect(s2))