import pathfinder as pf


if __name__ == "__main__":
    n = 8
    graph = pf.generate_graph(n, is_weighted=False, is_oriented=False, density=0.3)
    dijkstra = pf.BFSFinder(graph)
    print(dijkstra.find_path(0, graph.vertices_count - 1))
