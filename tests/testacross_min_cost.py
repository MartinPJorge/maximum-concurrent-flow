import sys
sys.path.append('..')
from mcf import min_cost_for_mcf, max_concurrent_flow_split, lambda_max_concurrent_flow_split
import networkx as nx
from math import log
import json

def test_():

    G = nx.Graph()

    G.add_edge("gNB", "rg", l=1, capacity=600)
    G.add_edge("ru", "UPF", l=1, capacity=600)

    G.add_edge("rg", "r5", l=1, capacity=200)
    G.add_edge("rg", "r6", l=1, capacity=200)

    G.add_edge("r5", "r4", l=1, capacity=200)
    G.add_edge("r5", "r6", l=1, capacity=200)

    G.add_edge("r4", "r1", l=1, capacity=200)
    G.add_edge("r6", "r2", l=1, capacity=200)

    G.add_edge("r2", "r1", l=1, capacity=200)
    G.add_edge("r1", "r3", l=1, capacity=200)
    G.add_edge("r2", "r3", l=1, capacity=200)

    G.add_edge("r1", "r7", l=1, capacity=200)
    G.add_edge("r7", "r8", l=1, capacity=200)
    G.add_edge("r8", "r9", l=1, capacity=200)

    G.add_edge("r3", "r9", l=1, capacity=200)

    G.add_edge("r8", "ru", l=1, capacity=200)
    G.add_edge("r9", "ru", l=1, capacity=200)
    b = {'gNB': 300, 'UPF': -300}

    srcs = ['gNB']
    tgts = ['UPF']
    ds   = [300]
    eps=0.99
    m = len(G.edges)
    delta = (m / (1-eps))**(-1/eps)
    print("Delta:", delta)
    flow_ij, used_paths = min_cost_for_mcf(
                G, 'gNB', 'UPF', 300,
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
