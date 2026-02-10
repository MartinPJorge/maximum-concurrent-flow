import sys
sys.path.append('..')
from mcf import min_cost_noDigraph, flow_to_used_paths, max_concurrent_flow_split
import networkx as nx
from math import log
def test_():


    G = nx.Graph()

    # Aristas del ejemplo
    G.add_edge("A", "B", capacity=400, l=400)
    G.add_edge("A", "C", capacity=400, l=300)
    G.add_edge("B", "C", capacity=400, l=400)

    # Balances (paper convention)
    b = {
        "A":  200,
        "B":    0,
        "C": -200
    }

    flow_dict = min_cost_noDigraph(G, b)
    print("Flow dict:", flow_dict)

    

    used_paths = flow_to_used_paths(flow_dict, s="A", t="C")

    for p, f in used_paths:
        print(p, f)
    # #commodities
    # srcs = ['A']
    # tgts = ['C']
    # ds   = [200]
    # eps=0.1
    # m = len(G.edges)
    # delta = (m / (1-eps))**(-1/eps)
    # f, paths = max_concurrent_flow_split(G,
    #     srcs,
    #     tgts,
    #     ds,
    #     delta,
    #     eps,
    #     c_label="capacity")

    # for i in paths:
    #     for j in paths[i]:
    #         print(f"Phase {i}, commodity {j}")
    #         for P, flow in paths[i][j]:
    #             print("  ", " -> ".join(P), "flow:", flow)


if __name__ == '__main__':
    test_()
