import sys
sys.path.append('..')
from mcf import min_cost_noDigraph, flow_to_used_paths
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
  


if __name__ == '__main__':
    test_()
