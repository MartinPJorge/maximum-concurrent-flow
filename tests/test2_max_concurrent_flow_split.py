import sys

sys.path.append('..')
from mcf import  max_concurrent_flow_split, lambda_max_concurrent_flow_split,min_cost_noDigraph,flow_to_used_paths,min_cost_for_mcf
import networkx as nx
from math import log
import json

def test_():

    """
    Test topology with three alternative paths between source and destination:
    0 -> 1 -> 5, 0 -> 1 -> 2 -> 5, and 0 -> 3 -> 4 -> 5.
    All links have unit cost and increased capacities.
    A single commodity with demand 10 is routed using the
    Maximum Concurrent Flow (splittable) algorithm.
    """

    G = nx.Graph()

    G.add_edge('0', '1', l=1, capacity=20)
    G.add_edge('1', '5', l=1, capacity=8)

    # Path P2: 0 -> 1 -> 2 -> 5
    G.add_edge('1', '2', l=1, capacity=8)
    G.add_edge('2', '5', l=1, capacity=8)

    # Path P3: 0 -> 3 -> 4 -> 5
    G.add_edge('0', '3', l=1, capacity=10)
    G.add_edge('3', '4', l=1, capacity=10)
    G.add_edge('4', '5', l=1, capacity=10)

    # --- b: +10 sale de 0, -10 entra en 5 ---
    b = {'0': 10, '5': -10}

    # flow = min_cost_noDigraph(G, b)
    # print("Flow dict (solo aristas con flujo > 0):")
    # for u, nbrs in flow.items():
    #     for v, f in nbrs.items():
    #         if f > 0:
    #             print(f"  {u} -> {v}: {f}")

    # used_paths= flow_to_used_paths( flow, '0', '5', 1e-6)
    # print("Used paths and their flows:")
    # print(json.dumps(used_paths, indent=2))
    

    #commodities
    srcs = ['0']
    tgts = ['5']
    ds   = [10]
    eps=0.99
    m = len(G.edges)
    delta = (m / (1-eps))**(-1/eps)
    print("Delta:", delta)
    flow_ij, used_paths = min_cost_for_mcf(
                G, '0', '5', 10,
                c_label="capacity",
                l_label='l',
                delta=delta,
            )
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

    print('fitted_flow:', json.dumps(fitted_flow, indent=2))


if __name__ == '__main__':
    test_()
