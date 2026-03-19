import json
import sys
sys.path.append('..')

from mcf import max_concurrent_flow_split_per_commodity_graphs, lambda_max_concurrent_flow_split
import networkx as nx


def test_multi_graphs():

    """
    Test con 2 commodities y grafos distintos.

    Commodity 0:
        0 -> 1 -> 5
        0 -> 2 -> 5

    Commodity 1:
        0 -> 1 -> 5   (comparte enlace con commodity 0)

    El enlace (1,5) es el cuello de botella.
    Queremos ver que el segundo commodity no puede usarlo completamente
    si el primero ya lo ha usado.
    """


    G0 = nx.Graph()

    # Path A: 0-1-5
    G0.add_edge('0', '1', l=1, capacity=10)
    G0.add_edge('1', '5', l=1, capacity=10)

    # Path B: 0-2-5
    G0.add_edge('0', '2', l=1, capacity=10)
    G0.add_edge('2', '5', l=1, capacity=10)


    G1 = nx.Graph()

    # SOLO un path: 0-1-5 (comparte cuello de botella)
    G1.add_edge('0', '1', l=1, capacity=10)
    G1.add_edge('1', '5', l=1, capacity=10)

    Gs = [G0, G1]

    srcs = ['0', '0']
    tgts = ['5', '5']
    ds   = [6, 4]   # demandas

    eps = 0.1

    # usamos m total de edges únicos
    all_edges = set()
    for G in Gs:
        for u, v in G.edges():
            all_edges.add(tuple(sorted((u, v))))
    m = len(all_edges)

    delta = (m / (1 - eps)) ** (-1 / eps)

    f, paths = max_concurrent_flow_split_per_commodity_graphs(
        Gs,
        srcs,
        tgts,
        ds,
        delta,
        eps,
        c_label="capacity"
    )

    print("\n====== PATHS ======")
    for i in paths:
        for j in paths[i]:
            print(f"Phase {i}, commodity {j}")
            if paths[i][j] is None:
                continue
            for P, flow in paths[i][j]:
                print("  ", " -> ".join(P), "flow:", flow)

    lambda_star, lambdas, fitted_flow = lambda_max_concurrent_flow_split(
        G=Gs[0],   # usamos uno base (solo para evaluación)
        ds=ds,
        c_label="capacity",
        paths=paths
    )

    print("\n====== RESULT ======")
    print("λ global:", lambda_star)
    print("λ por commodity:", lambdas)
    print("fitted flow:", json.dumps(fitted_flow, indent=2))


if __name__ == '__main__':
    test_multi_graphs()