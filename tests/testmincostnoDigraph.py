import sys
sys.path.append('..')
from mcf import min_cost_noDigraph
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

    y_star, cost, y_hat = min_cost_noDigraph(G, b)

    print("problema auxiliar R)")
    for k, v in sorted(y_hat.items()):
        print(f"{k}: {v}")

    print("problema original no dirigido")
    for k, v in sorted(y_star.items()):
        print(f"{k}: {v}")

    print(cost)

  


if __name__ == '__main__':
    test_()
