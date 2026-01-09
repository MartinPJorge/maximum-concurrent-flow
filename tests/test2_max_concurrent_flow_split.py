import sys
from mcf import min_cost, max_concurrent_flow_split, lambda_max_concurrent_flow_split
import networkx as nx
from math import log

def test_():

    """
    Test topology with three alternative paths between source and destination:
    0 -> 1 -> 5, 0 -> 1 -> 2 -> 5, and 0 -> 3 -> 4 -> 5.
    All links have unit cost and increased capacities.
    A single commodity with demand 10 is routed using the
    Maximum Concurrent Flow (splittable) algorithm.
    """

    G = nx.Graph()

    # Path P1: 0 -> 1 -> 5
    G.add_edge('0', '1', l=1, capacity=20)
    G.add_edge('1', '5', l=1, capacity=8)

    # Path P2: 0 -> 1 -> 2->5 
    G.add_edge('0', '1', l=1, capacity=20)
    G.add_edge('1', '2', l=1, capacity=8)
    G.add_edge('2', '5', l=1, capacity=8)

    # Path P3: 0 ->3->4-> 5 
    G.add_edge('0', '3', l=1, capacity=10)
    G.add_edge('3', '4', l=1, capacity=10)
    G.add_edge('4', '5', l=1, capacity=10)

    #commodities
    srcs = ['0']
    tgts = ['5']
    ds   = [10]
    eps=0.1
    m = len(G.edges)
    delta = (m / (1-eps))**(-1/eps)
    f, paths = max_concurrent_flow_split(G,
        srcs,
        tgts,
        ds,
        delta,
        eps,
        c_label="capacity")

    for i in paths:
        for j in paths[i]:
            print(f"Phase {i}, commodity {j}")
            for P, flow in paths[i][j]:
                print("  ", " -> ".join(P), "flow:", flow)


    lambda_star, lambdas, fitted_flow = lambda_max_concurrent_flow_split(
    G=G,
    ds=ds,
    c_label="capacity",
    paths=paths
)

    print("λ global:", lambda_star)
    print("λ por commodity:", lambdas)


if __name__ == '__main__':
    test_()
