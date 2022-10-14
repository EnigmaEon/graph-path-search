from abc import ABC, abstractmethod
import random
from collections import deque
import heapq


class Edge:
    def __init__(self, a: int, b: int, weight: int = 1):
        self._a = a
        self._b = b
        self._weight = weight

    def __str__(self):
        return f'({self._a}, {self._b}, {self._weight})'

    @property
    def a(self):
        return self._a

    @property
    def b(self):
        return self._b

    @property
    def weight(self):
        return self._weight


class Graph:
    def __init__(self):
        self._vertices = []
        self._edges = set()

    def __str__(self):
        ret = ""
        for edge in self._edges:
            ret += str(edge) + '\n'
        return ret

    @property
    def edges_count(self):
        return len(self._edges)

    @property
    def vertices_count(self):
        return len(self._vertices)

    def in_graph(self, edge: Edge) -> bool:
        return edge in self._edges

    def add_edge(self, edge: Edge, is_oriented: bool) -> None:
        while len(self._vertices) <= max(edge.a, edge.b):
            self._vertices.append([])

        self._vertices[edge.a].append(edge)
        if not is_oriented:
            inv_edge = Edge(edge.b, edge.a, edge.weight)
            self._vertices[edge.b].append(inv_edge)

        self._edges.add(edge)

    def neighbours(self, vertex: int) -> list:
        return self._vertices[vertex]


class GraphPath:
    def __init__(self, way):
        self._way = list(way)
        self._length = 0
        for edge in way:
            self._length += edge.weight

    def __str__(self):
        if len(self._way) == 0:
            return ""
        ret = str(self._way[0].a)
        for edge in self._way:
            ret += "->" + str(edge.b)
        return ret


def _restore_answer(start: int, end: int, prev: list) -> GraphPath:
    way = []
    cur = end
    while cur != start:
        way.append(prev[cur])
        cur = prev[cur].a
    return GraphPath(reversed(way))


class Finder(ABC):
    def __init__(self, graph: Graph):
        self.graph = graph

    @abstractmethod
    def find_path(self, start: int, end: int) -> GraphPath:
        pass


class BFSFinder(Finder):
    def find_path(self, start: int, end: int) -> GraphPath:
        assert 0 <= start < self.graph.vertices_count
        assert 0 <= end < self.graph.vertices_count

        dist = [10 ** 9 for _ in range(self.graph.vertices_count)]
        prev = [-1 for _ in range(self.graph.vertices_count)]

        queue = deque()
        queue.appendleft(start)
        dist[start] = 0
        while len(queue) > 0:
            v = queue.pop()
            for edge in self.graph.neighbours(v):
                if dist[v] + edge.weight < dist[edge.b]:
                    dist[edge.b] = dist[v] + edge.weight
                    prev[edge.b] = edge
                    queue.appendleft(edge.b)

        return _restore_answer(start, end, prev)


class DijkstraFinder(Finder):
    def find_path(self, start: int, end: int) -> GraphPath:
        assert 0 <= start < self.graph.vertices_count
        assert 0 <= end < self.graph.vertices_count

        dist = [10 ** 9 for _ in range(self.graph.vertices_count)]
        prev = [-1 for _ in range(self.graph.vertices_count)]

        queue = [(0, start)]
        dist[start] = 0

        while len(queue) > 0:
            cur, v = heapq.heappop(queue)
            if dist[v] < -cur:
                continue
            for edge in self.graph.neighbours(v):
                if dist[v] + edge.weight < dist[edge.b]:
                    dist[edge.b] = dist[v] + edge.weight
                    prev[edge.b] = edge
                    heapq.heappush(queue, (-dist[edge.b], edge.b))

        return _restore_answer(start, end, prev)


def generate_graph(n: int, is_weighted=False, is_oriented=False, density=0.2):
    """Return random connected graph with given density"""
    assert n >= 0, "The number of vertices must be non-negative"
    assert 0 <= density <= 1, "Density of graph must be in range [0;1]"

    edges_count = int(density * n * n)
    if edges_count < n - 1:
        raise ValueError("Graph density is too low")

    graph = generate_tree(n, is_weighted, is_oriented)

    while graph.edges_count < edges_count:
        edge = Edge(random.randint(0, n - 1), random.randint(0, n - 1),
                    random.randint(0, n * n) if is_weighted else 1)
        if edge.a == edge.b or graph.in_graph(edge):
            continue
        graph.add_edge(edge, is_oriented)

    return graph


def generate_tree(n: int, is_weighted=False, is_oriented=False) -> Graph:
    unused = [i for i in range(n)]
    random.shuffle(unused)
    used = [unused.pop()]
    tree = Graph()
    while tree.edges_count < n - 1:
        cur = used[random.randint(0, len(used) - 1)]
        used.append(unused.pop())
        tree.add_edge(Edge(cur, used[-1],
                           random.randint(0, n * n) if is_weighted else 1),
                      is_oriented)

    return tree
